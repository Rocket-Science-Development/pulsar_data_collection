"""ConsistencyProbe: repeated questions, normalized final answers, pairwise agreement, stubbed endpoints"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from itertools import cycle

import pytest

from pulsar_data_collection.genai import (
    ConsistencyProbe,
    ConsistencyQuestion,
    EmbeddingSimilarity,
    EntailmentSimilarity,
    LexicalSimilarity,
    normalize_answer,
    openai_chat_target,
)
from pulsar_data_collection.privacy import PiiPolicy

SIN = "130 692 544"  # fictitious, valid checksum


@pytest.mark.parametrize(
    "answer, kind, labels, expected",
    [
        ("Oui, il implique des renseignements personnels.", "yes_no", (), "yes"),
        ("**No.** It does not.", "yes_no", (), "no"),
        ("Réponse : non", "yes_no", (), "no"),
        ("I am not sure.", "yes_no", (), None),
        ("The total is 6,370 dollars.", "number", (), "6370"),
        ("6 370 $", "number", (), "6370"),
        ("Environ 3,5 %", "number", (), "3.5"),
        ("Answer: 12.50", "number", (), "12.5"),
        ("Il est en développement, et sera en production en 2027.", "label", ("production", "développement"), "développement"),
        ("Hors service.", "label", ("production", "hors service"), "hors service"),
        ("Aucune idée", "label", ("production", "retired"), None),
        ("  Winnipeg. ", "short", (), "winnipeg"),
        ("La Saskatchewan", "short", (), "saskatchewan"),
        ("Free text answer", "free", (), None),
        (None, "number", (), None),
    ],
)
def test_normalized_final_answer(answer, kind, labels, expected):
    assert normalize_answer(answer, kind, labels) == expected


def test_yes_no_ignores_a_hyphenated_non():
    assert normalize_answer("Les non-résidents paient-ils la taxe ? Oui.", "yes_no") == "yes"
    assert normalize_answer("Le no-show est facturé.", "yes_no") is None


@pytest.mark.parametrize(
    "answer, expected",
    [("1.234.567", "1234567"), ("1.234.567,5 €", "1234567.5"), ("COVID-19: 5 cases", "5"), ("A4: 3 pages", "3")],
)
def test_number_thousands_dots_and_numbers_inside_words(answer, expected):
    assert normalize_answer(answer, "number") == expected


@pytest.mark.parametrize(
    "answer, kind",
    [
        ("Sorry, no, I cannot answer that.", "yes_no"),
        ("I'm sorry, I can't say yes or no.", "yes_no"),
        ("Je suis désolé, je ne peux pas répondre: non.", "yes_no"),
        ("Sorry, I cannot give the 2026 figure.", "number"),
    ],
)
def test_a_refusal_is_not_read_as_a_categorical_answer(answer, kind):
    question = ConsistencyQuestion("q", "?", answer_type=kind)
    probe = ConsistencyProbe([question], repeats=2)
    assert probe._read(answer, question) == ("refusal", None)


def test_a_final_answer_marker_after_an_apology_is_read():
    question = ConsistencyQuestion("q", "?", answer_type="yes_no")
    assert ConsistencyProbe([question], repeats=2)._read("I'm sorry for the delay. Answer: no", question) == ("ok", "no")


def test_normalize_refuses_an_unknown_type():
    with pytest.raises(ValueError):
        normalize_answer("x", "essay")


QUESTIONS = [
    ConsistencyQuestion("q-cap", "Capital of Manitoba? One word.", answer_type="short", expected="Winnipeg"),
    ConsistencyQuestion("q-sum", "245 x 26? Number only.", answer_type="number", expected="6370"),
    ConsistencyQuestion("q-yn", "Is Ottawa the capital of Canada? Yes or no.", answer_type="yes_no", expected="yes"),
]


def test_consistent_target_agrees_fully_and_records_correctness():
    answers = {QUESTIONS[0].text: "Winnipeg.", QUESTIONS[1].text: "6370", QUESTIONS[2].text: "Yes."}
    result = ConsistencyProbe(QUESTIONS, repeats=3).run(lambda text: answers[text])
    q = result.questions.set_index("question_id")
    assert (q["n_answers"] == 3).all() and (q["n_pairs"] == 3).all()
    assert (q["agreement"] == 1.0).all() and q["full_agreement"].all()
    assert (q["correct_share"] == 1.0).all() and (q["agreement_method"] == "exact").all()
    assert q.loc["q-sum", "majority_answer"] == "6370" and q.loc["q-sum", "majority_share"] == 1.0
    assert len(result.answers) == 9 and set(result.answers["normalized"]) == {"winnipeg", "6370", "yes"}
    # the question texts are not stored
    assert not result.answers.isin([qq.text for qq in QUESTIONS]).any().any()


def test_inconsistent_target_scores_pairs_and_keeps_errors_apart():
    replies = {
        QUESTIONS[0].text: cycle(["Winnipeg", "Regina", "Winnipeg"]),
        QUESTIONS[1].text: cycle(["6370", "6 370", "boom"]),
        QUESTIONS[2].text: cycle(["Yes", "I'm sorry, I cannot answer.", "I'm sorry, I can't."]),
    }

    def ask(text):
        reply = next(replies[text])
        if reply == "boom":
            raise TimeoutError("no answer")
        return reply

    result = ConsistencyProbe(QUESTIONS, repeats=3, seed=1).run(ask)
    q = result.questions.set_index("question_id")
    assert q.loc["q-cap", "agreement"] == pytest.approx(1 / 3) and not q.loc["q-cap", "full_agreement"]
    assert q.loc["q-cap", "correct_share"] == pytest.approx(2 / 3)
    # the failed call is not an answer: 2 answers, 1 pair, both read 6370
    assert q.loc["q-sum", "n_errors"] == 1 and q.loc["q-sum", "n_answers"] == 2 and q.loc["q-sum", "agreement"] == 1.0
    assert (result.answers["answer_status"] == "error").sum() == 1
    # two refusals agree with each other, not with the answer
    assert q.loc["q-yn", "n_refusals"] == 2 and q.loc["q-yn", "agreement"] == pytest.approx(1 / 3)
    assert sorted(q.loc["q-yn", "pair_scores"]) == [0.0, 0.0, 1.0]


def test_unparsed_answers_agree_with_nothing():
    question = ConsistencyQuestion("q", "Yes or no?", answer_type="yes_no")
    result = ConsistencyProbe([question], repeats=2).run(lambda text: "Perhaps.")
    row = result.questions.iloc[0]
    assert row["n_unparsed"] == 2 and row["agreement"] == 0.0 and not row["full_agreement"]


def test_paraphrases_are_scored_apart_and_requests_spread_in_rounds():
    question = ConsistencyQuestion(
        "q", "Capital of Canada?", paraphrases=("Which city is Canada's capital?",), answer_type="short"
    )
    probe = ConsistencyProbe([question], repeats=3, paraphrase_repeats=2)
    plan = probe.plan()
    assert probe.planned_calls == len(plan) == 5
    assert [r.repeat for r in plan] == sorted(r.repeat for r in plan)  # round after round
    result = probe.run(lambda text: "Ottawa" if text.startswith("Capital") else "Toronto")
    row = result.questions.iloc[0]
    assert row["agreement_repeats"] == 1.0 and row["agreement_paraphrases"] == 0.0
    assert row["n_pairs"] == 10 and row["agreement"] == pytest.approx(4 / 10)


def test_free_text_uses_the_similarity_and_its_threshold():
    question = ConsistencyQuestion("q", "Explain a grant.")
    answers = cycle(["A grant is money you do not repay.", "A grant is money that you do not repay."])
    result = ConsistencyProbe([question], repeats=2, similarity=LexicalSimilarity()).run(lambda text: next(answers))
    row = result.questions.iloc[0]
    assert row["agreement_method"] == "lexical" and 0.5 < row["agreement"] < 1 and row["full_agreement"]
    strict = ConsistencyProbe([question], repeats=2, agreement_threshold=0.99).run(lambda text: next(answers))
    assert not strict.questions.iloc[0]["full_agreement"]


class FakeNli:
    def entailment(self, premise, hypothesis):
        return 0.9 if "repay" in premise and "repay" in hypothesis else 0.1


def test_entailment_similarity_is_bidirectional():
    sim = EntailmentSimilarity(FakeNli())
    assert sim("money to repay", "no repay needed") == 0.9 and sim("money to repay", "a gift") == 0.1
    assert EntailmentSimilarity(FakeNli(), aggregate="mean")("repay", "gift") == pytest.approx(0.1)
    with pytest.raises(TypeError):
        EntailmentSimilarity(object())


def test_keep_answers_false_and_pii_masking():
    question = ConsistencyQuestion("q", "Write a note.", answer_type="short")
    reply = f"Le NAS est {SIN}"
    kept = ConsistencyProbe([question], repeats=2, pii_policy=PiiPolicy()).run(lambda text: reply)
    assert not kept.answers["answer"].str.contains(SIN).any()
    assert "[SIN]" in kept.answers["answer"].iloc[0]
    assert SIN.replace(" ", "") not in str(kept.questions["majority_answer"].iloc[0]).replace(" ", "")
    assert kept.questions.iloc[0]["full_agreement"]  # scored on the texts as received
    dropped = ConsistencyProbe([question], repeats=2, keep_answers=False).run(lambda text: reply)
    assert "answer" not in dropped.answers


def test_invalid_settings_are_refused():
    with pytest.raises(ValueError):
        ConsistencyProbe(QUESTIONS, repeats=1)
    with pytest.raises(ValueError):
        ConsistencyProbe([QUESTIONS[0], QUESTIONS[0]])
    with pytest.raises(ValueError):
        ConsistencyProbe([ConsistencyQuestion("q", "?", answer_type="label")])
    with pytest.raises(ValueError):
        ConsistencyProbe([])


def test_workers_keep_one_row_per_call():
    answers = {QUESTIONS[0].text: "Winnipeg", QUESTIONS[1].text: "6370", QUESTIONS[2].text: "yes"}
    seen = []
    result = ConsistencyProbe(QUESTIONS, repeats=4).run(lambda t: answers[t], workers=2, progress=lambda i, n, r: seen.append(i))
    assert len(result.answers) == 12 and seen == list(range(1, 13)) and result.questions["full_agreement"].all()


# ------------------------------------------------------------------------------------- stubbed HTTP endpoint
class _Handler(BaseHTTPRequestHandler):
    requests: list = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append((self.path, body, self.headers.get("Authorization")))
        if self.path.endswith("/chat/completions"):
            out = {"choices": [{"message": {"role": "assistant", "content": "Winnipeg"}}]}
        else:
            vectors = {"a": [1.0, 0.0], "b": [0.6, 0.8], "c": [-1.0, 0.0]}
            out = {"data": [{"index": i, "embedding": vectors[t[-1]]} for i, t in enumerate(body["input"])]}
        payload = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def endpoint():
    _Handler.requests = []
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()


def test_openai_chat_target_sends_the_settings(endpoint):
    ask = openai_chat_target(endpoint, "llama3.2:3b", temperature=0.7, max_tokens=32, api_key="test-key", seed=3)
    assert ask("Capital of Manitoba?") == "Winnipeg"
    path, body, auth = _Handler.requests[-1]
    assert path == "/v1/chat/completions" and auth == "Bearer test-key"
    assert body["temperature"] == 0.7 and body["max_tokens"] == 32 and body["seed"] == 3
    assert body["messages"] == [{"role": "user", "content": "Capital of Manitoba?"}]
    with pytest.raises(ValueError):
        openai_chat_target("file:///etc", "m")("x")


def test_embedding_similarity_is_a_cached_cosine(endpoint):
    sim = EmbeddingSimilarity(endpoint, model="nomic-embed-text", prefix="clustering: ")
    assert sim("text a", "text b") == pytest.approx(0.6)
    assert sim("text a", "text c") == 0.0  # negative cosine counts as 0
    assert sim("text a", "text b") == pytest.approx(0.6)
    assert len(_Handler.requests) == 2  # "text a" and "text b" embedded once
    assert _Handler.requests[0][1]["input"][0].startswith("clustering: ")
    question = ConsistencyQuestion("q", "Explain.")
    replies = cycle(["text a", "text b"])
    row = ConsistencyProbe([question], repeats=2, similarity=sim).run(lambda t: next(replies)).questions.iloc[0]
    assert (
        row["agreement"] == pytest.approx(0.6) and not row["full_agreement"] and row["agreement_method"].startswith("embedding")
    )


def test_score_reads_the_answers_again_with_its_own_questions():
    short = ConsistencyQuestion("q", "Capital of Manitoba?", answer_type="short")
    replies = cycle(["Winnipeg.", "The capital is Winnipeg"])
    answers = ConsistencyProbe([short], repeats=2).run(lambda t: next(replies)).answers
    assert ConsistencyProbe([short], repeats=2).score(answers).iloc[0]["agreement"] == 0.0
    free = ConsistencyProbe([ConsistencyQuestion("q", "Capital of Manitoba?")], repeats=2, similarity=LexicalSimilarity())
    row = free.score(answers).iloc[0]
    assert row["agreement_method"] == "lexical" and 0 < row["agreement"] < 1  # not the stored exact comparison
