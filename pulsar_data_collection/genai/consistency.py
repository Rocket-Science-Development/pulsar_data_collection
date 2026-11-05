"""Consistency probe: the same question asked several times (and in other words), and how much the answers agree.

A system that gives another answer each time the same question is asked cannot be trusted on any single answer. The
probe asks each question `repeats` times (and each paraphrase supplied by the user `paraphrase_repeats` times), at the
temperature of the deployment, and records per question:

- the answers (optional, `keep_answers`; masked with a `PiiPolicy` when given) and the **normalized final answer**
  when the question expects a short one: a yes / no, a number, one of a list of labels, or a short text;
- the **pairwise agreement** of the answers: exact match of the normalized answers for short answers (1 or 0), a
  semantic score in [0, 1] for free text (`EmbeddingSimilarity`: cosine of the embeddings from an OpenAI-compatible
  endpoint; `EntailmentSimilarity`: bidirectional entailment with the NLI model of `FaithfulnessScorer`; default:
  word overlap, a weak lexical proxy);
- per question: the mean agreement, whether all the answers agree (`full_agreement`), the majority answer and its
  share, and the share of correct answers when the expected answer is known.

The per-question frame is analysed with `pulsar_metrics.genai.consistency` (window metric with bootstrap intervals)
and `consistency_drift` (one-sided test of a drop against a reference). The question texts are not stored: rows hold
the `question_id` and the variant ("original", "paraphrase-1", ...).

Refusals are recorded as such (`answer_status="refusal"`): two refusals of the same question agree, a refusal and an
answer do not. An answer from which no final answer can be read (`"unparsed"`) agrees with nothing. Failed calls
(`"error"`) are not answers: they are counted apart and left out of the agreement.
"""

import json
import math
import random  # reproducible ordering of the requests, never used for security
import re
import time
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable, Optional, Sequence

import pandas as pd

from .probes import ANSWER_ERROR, ANSWER_OK, ANSWER_REFUSAL, ANSWER_UNPARSED, is_refusal

ANSWER_TYPES = ("free", "short", "label", "number", "yes_no")
CATEGORICAL_TYPES = ("short", "label", "number", "yes_no")
ORIGINAL = "original"
REFUSAL_VALUE = "<refusal>"  # normalized value of a refusal: two refusals agree

_MARKER = re.compile(r"(?:final answer|answer|réponse finale|reponse finale|réponse|reponse)\s*[:：]", re.IGNORECASE)
# a whole word: the "non" of "non-résidents" or the "no" of "no-show" is not an answer
_YES_NO = re.compile(r"\b(yes|oui|no|non|true|vrai|false|faux)\b(?![-‐]\w)")
_YES = {"yes", "oui", "true", "vrai"}
# a sign only outside a word and no number glued to a word ("COVID-19", "A4"); thousands separated by spaces, commas,
# or at least two dots ("1.234.567")
_NUMBER = re.compile(
    r"(?:(?<!\w)[-−])?(?<![\w.,])(?<!\w[-−])"
    r"(?:\d{1,3}(?:[ \u00a0\u202f,]\d{3})+(?:[.,]\d+)?(?!\d)|\d{1,3}(?:\.\d{3}){2,}(?:,\d+)?(?![\d.])|\d+(?:[.,]\d+)?)"
)
_ARTICLE = re.compile(r"^(?:the|a|an|le|la|les|l|un|une)\s+")


def fold(text: str) -> str:
    """Lower case, accents removed, whitespace collapsed"""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(text.lower().split())


def _final_segment(text: str) -> str:
    """The text after the last "Answer:" / "Réponse :" marker, or the whole text"""
    parts = _MARKER.split(text)
    return parts[-1] if len(parts) > 1 and parts[-1].strip() else text


def parse_number(text: str) -> Optional[float]:
    """First number of the text: "6 370", "6,370", "1.234.567", "6370.5", "3,5" (French decimal comma) and "-2" are
    read; a number inside a word ("COVID-19", "A4") is not"""
    match = _NUMBER.search(text)
    if not match:
        return None
    raw = re.sub(r"[ \u00a0\u202f]", "", match.group(0)).replace("−", "-")
    if raw.count(".") > 1:  # 1.234.567 or 1.234.567,5: dots separate thousands
        raw = raw.replace(".", "").replace(",", ".")
    elif "," in raw and "." in raw:
        raw = raw.replace(",", "") if raw.rfind(".") > raw.rfind(",") else raw.replace(".", "").replace(",", ".")
    elif "," in raw:
        # 6,370 or 1,234,567: thousands; 3,5 or 12,75: decimal comma
        raw = raw.replace(",", "") if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+", raw) else raw.replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def _number_text(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") if math.isfinite(value) else str(value)


def normalize_answer(answer: Any, answer_type: str = "short", labels: Sequence[str] = ()) -> Optional[str]:
    """Normalized final answer, or None when none can be read

    "yes_no": "yes" or "no" (first yes / oui / no / non / true / vrai / false / faux); "number": the first number, as
    text ("6370", "3.5"); "label": the label of `labels` named first (accent- and case-insensitive, whole words);
    "short": the folded text without punctuation nor leading article; "free": None (agreement is semantic). The text
    after a final "Answer:" or "Réponse :" marker is read when there is one.
    """
    if answer_type not in ANSWER_TYPES:
        raise ValueError(f"answer_type must be one of {ANSWER_TYPES}")
    if not isinstance(answer, str) or not answer.strip() or answer_type == "free":
        return None
    text = fold(_final_segment(answer))
    if answer_type == "yes_no":
        match = _YES_NO.search(text)
        return None if not match else ("yes" if match.group(1) in _YES else "no")
    if answer_type == "number":
        value = parse_number(text)
        return None if value is None else _number_text(value)
    if answer_type == "label":
        found = []
        for label in labels:
            match = re.search(r"(?<!\w)" + re.escape(fold(label)) + r"(?!\w)", text)
            if match:
                found.append((match.start(), -len(label), label))
        return min(found)[2] if found else None
    words = re.sub(r"[^\w\s'-]", " ", text).replace("'", " ")
    words = _ARTICLE.sub("", " ".join(words.split()))
    return words or None


# ------------------------------------------------------------------------------------------- semantic agreement
def _words(text: str) -> set:
    return set(re.findall(r"\w+", fold(text)))


class LexicalSimilarity:
    """Jaccard overlap of the words of two answers: a cheap, weak proxy (two paraphrases may share few words)"""

    name = "lexical"
    threshold = 0.5

    def __call__(self, a: str, b: str) -> float:
        wa, wb = _words(a), _words(b)
        return 1.0 if not wa and not wb else len(wa & wb) / len(wa | wb)


def _post_json(url: str, payload: dict, timeout: float, api_key: Optional[str]) -> dict:
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"only http(s) endpoints are supported, got {url!r}")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    # the scheme is checked above
    with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
        return json.loads(response.read().decode("utf-8"))


class EmbeddingSimilarity:
    """Cosine similarity of the embeddings of two answers, from an OpenAI-compatible `/embeddings` endpoint

    Parameters
    ----------
    base_url : str, optional
        Base URL of the API (default the local Ollama, "http://127.0.0.1:11434/v1": on Windows, "localhost" first
        tries IPv6 and each call waited about 20 s before falling back)
    model : str, optional
        Embedding model (default "nomic-embed-text"; "granite-embedding:278m" is multilingual)
    timeout : float, optional
        Seconds per request (default 60)
    api_key : str, optional
    prefix : str, optional
        Text put before each answer (some models expect a task prefix, e.g. "clustering: " for nomic-embed-text)
    threshold : float, optional
        Score from which two free-text answers are taken as agreeing (`full_agreement`); default 0.9. The cosine of
        unrelated texts is well above 0 with most models: fit it on answers known to agree and to differ

    Embeddings are cached per text; negative cosines count as 0.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434/v1",
        model: str = "nomic-embed-text",
        timeout: float = 60.0,
        api_key: Optional[str] = None,
        prefix: str = "",
        threshold: float = 0.9,
    ):
        self.base_url, self.model, self.timeout, self.api_key = base_url.rstrip("/"), model, timeout, api_key
        self.prefix, self.threshold = prefix, threshold
        self.name = f"embedding:{model}"
        self._cache: dict = {}

    def embed(self, texts: Sequence[str]) -> list:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            payload = {"model": self.model, "input": [self.prefix + t for t in missing]}
            data = _post_json(f"{self.base_url}/embeddings", payload, self.timeout, self.api_key)["data"]
            for item in sorted(data, key=lambda d: d.get("index", 0)):
                self._cache[missing[item.get("index", 0)]] = [float(x) for x in item["embedding"]]
        return [self._cache[t] for t in texts]

    def __call__(self, a: str, b: str) -> float:
        ea, eb = self.embed([a, b])
        na, nb = math.sqrt(sum(x * x for x in ea)), math.sqrt(sum(x * x for x in eb))
        if not na or not nb:
            return 0.0
        return max(0.0, min(1.0, sum(x * y for x, y in zip(ea, eb)) / (na * nb)))


class EntailmentSimilarity:
    """Bidirectional entailment of two answers with an NLI model: each answer as premise of the other

    `scorer` is any object with an `entailment(premise, hypothesis) -> float` method, e.g. a
    `FaithfulnessScorer` (multilingual mDeBERTa NLI). `aggregate` "min" (default: both directions must hold, the
    answers say the same thing) or "mean". Threshold of `full_agreement` default 0.5.
    """

    name = "nli"

    def __init__(self, scorer, aggregate: str = "min", threshold: float = 0.5):
        if aggregate not in ("min", "mean"):
            raise ValueError('aggregate must be "min" or "mean"')
        if not callable(getattr(scorer, "entailment", None)):
            raise TypeError("scorer needs an entailment(premise, hypothesis) method (e.g. FaithfulnessScorer)")
        self.scorer, self.aggregate, self.threshold = scorer, aggregate, threshold

    def __call__(self, a: str, b: str) -> float:
        forward, backward = float(self.scorer.entailment(a, b)), float(self.scorer.entailment(b, a))
        return min(forward, backward) if self.aggregate == "min" else (forward + backward) / 2


def openai_chat_target(
    base_url: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = 128,
    timeout: float = 120.0,
    api_key: Optional[str] = None,
    system: Optional[str] = None,
    **params,
) -> Callable[[str], str]:
    """`ask(prompt) -> answer` for an OpenAI-compatible `/chat/completions` endpoint (Ollama, vLLM, a provider...)

    `max_tokens` caps the answer (default 128) and `timeout` each call (default 120 s); `params` go into the request
    body (e.g. top_p, seed).
    """
    url = base_url.rstrip("/") + "/chat/completions"

    def ask(prompt: str) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        payload = {"model": model, "messages": messages, "temperature": temperature, **params}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        data = _post_json(url, payload, timeout, api_key)
        return data["choices"][0]["message"]["content"] or ""

    return ask


# ------------------------------------------------------------------------------------------------------- probe
@dataclass(frozen=True)
class ConsistencyQuestion:
    """A question asked several times

    question_id : str
        Stable identifier (the text is not stored in the results)
    text : str
    paraphrases : tuple of str, optional
        Other wordings of the same question
    answer_type : str, optional
        "free" (default), "short", "label", "number" or "yes_no"
    labels : tuple of str, optional
        The allowed answers of a "label" question
    expected : str, optional
        Known answer, normalized like the answers (for the share of correct answers)
    """

    question_id: str
    text: str
    paraphrases: tuple = ()
    answer_type: str = "free"
    labels: tuple = ()
    expected: Optional[str] = None
    context: dict = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class ConsistencyRequest:
    """One call of the probe"""

    question_id: str
    variant: str  # "original" or "paraphrase-<k>"
    repeat: int
    text: str


@dataclass
class ConsistencyResult:
    """`answers`: one row per call; `questions`: one row per question (the input of pulsar_metrics.genai.consistency)"""

    answers: pd.DataFrame
    questions: pd.DataFrame


class ConsistencyProbe:
    """Ask each question several times against a target and score the agreement of the answers

    Parameters
    ----------
    questions : sequence of ConsistencyQuestion or dict
        The questions (dicts are passed to ConsistencyQuestion)
    repeats : int, optional
        Answers per question in its original wording (default 3, at least 2 without paraphrases)
    paraphrase_repeats : int, optional
        Answers per paraphrase (default 1)
    similarity : callable, optional
        `similarity(a, b) -> [0, 1]` for free-text questions: EmbeddingSimilarity, EntailmentSimilarity, or your own
        (default LexicalSimilarity, a weak proxy). Its `threshold` attribute, if any, is the default
        `agreement_threshold`
    agreement_threshold : float, optional
        Pair score from which two free-text answers agree, for `full_agreement` (default the similarity's threshold,
        else 0.5). Short answers agree when their normalized answers are equal
    keep_answers : bool, optional
        Keep the answer texts in `answers` (default True, as CanaryProbe); False keeps the normalized answers only
    pii_policy : PiiPolicy, optional
        Applied to the answer texts and the normalized answers before they are returned (mask or flag); the scores are
        computed on the texts as received
    normalize : callable, optional
        `normalize(answer, question) -> str or None`, replacing `normalize_answer`
    seed : int, optional

    The calls are made round after round (each round asks every question and paraphrase once, in a random order), so
    that the repetitions of a question are spread in time rather than drawn back to back.
    """

    def __init__(
        self,
        questions: Sequence,
        repeats: int = 3,
        paraphrase_repeats: int = 1,
        similarity: Optional[Callable[[str, str], float]] = None,
        agreement_threshold: Optional[float] = None,
        keep_answers: bool = True,
        pii_policy=None,
        normalize: Optional[Callable[[Any, ConsistencyQuestion], Optional[str]]] = None,
        seed: int = 0,
    ):
        self.questions = [q if isinstance(q, ConsistencyQuestion) else ConsistencyQuestion(**q) for q in questions]
        if not self.questions:
            raise ValueError("at least one question is required")
        ids = [q.question_id for q in self.questions]
        if len(set(ids)) != len(ids):
            raise ValueError("question_id must be unique")
        for q in self.questions:
            if q.answer_type not in ANSWER_TYPES:
                raise ValueError(f"{q.question_id}: answer_type must be one of {ANSWER_TYPES}")
            if q.answer_type == "label" and not q.labels:
                raise ValueError(f"{q.question_id}: a label question needs its labels")
        if repeats < 1 or paraphrase_repeats < 0:
            raise ValueError("repeats must be at least 1 and paraphrase_repeats at least 0")
        if repeats < 2 and not (paraphrase_repeats and any(q.paraphrases for q in self.questions)):
            raise ValueError("repeats must be at least 2 (agreement needs two answers)")
        self.repeats, self.paraphrase_repeats = repeats, paraphrase_repeats
        self.similarity = similarity or LexicalSimilarity()
        if agreement_threshold is None:
            agreement_threshold = getattr(self.similarity, "threshold", 0.5)
        self.agreement_threshold = float(agreement_threshold)
        self.keep_answers, self.pii_policy, self.normalize, self.seed = keep_answers, pii_policy, normalize, seed

    @property
    def planned_calls(self) -> int:
        return sum(self.repeats + self.paraphrase_repeats * len(q.paraphrases) for q in self.questions)

    def plan(self) -> list:
        """The requests, round after round, each round in a random order"""
        rng = random.Random(self.seed)  # nosec B311
        rounds = max(self.repeats, self.paraphrase_repeats)
        out = []
        for r in range(rounds):
            batch = []
            for q in self.questions:
                if r < self.repeats:
                    batch.append(ConsistencyRequest(q.question_id, ORIGINAL, r, q.text))
                if r < self.paraphrase_repeats:
                    batch += [ConsistencyRequest(q.question_id, f"paraphrase-{k}", r, p) for k, p in enumerate(q.paraphrases, 1)]
            rng.shuffle(batch)
            out += batch
        return out

    def _read(self, answer: Any, question: ConsistencyQuestion) -> tuple:
        """(answer_status, normalized)

        With the default normalization, a categorical answer that starts with a refusal is a refusal even when a yes /
        no or a number can be read in it ("Sorry, no, I cannot answer that."), unless an "Answer:" marker gives the
        answer after it.
        """
        if self.normalize is None and question.answer_type in CATEGORICAL_TYPES and is_refusal(answer):
            if _final_segment(answer) is answer:  # no marker followed by an answer
                return ANSWER_REFUSAL, None
        if question.answer_type in CATEGORICAL_TYPES or self.normalize is not None:
            value = (
                self.normalize(answer, question)
                if self.normalize is not None
                else normalize_answer(answer, question.answer_type, question.labels)
            )
            if value is not None:
                return ANSWER_OK, value
            return (ANSWER_REFUSAL, None) if is_refusal(answer) else (ANSWER_UNPARSED, None)
        if is_refusal(answer):
            return ANSWER_REFUSAL, None
        return (ANSWER_OK, None) if isinstance(answer, str) and answer.strip() else (ANSWER_UNPARSED, None)

    def run(
        self,
        ask: Callable[[str], str],
        workers: int = 1,
        progress: Optional[Callable[[int, int, dict], None]] = None,
    ) -> ConsistencyResult:
        """Send every request to `ask(text) -> answer` (`workers` calls at a time) and score the agreement

        A failed call is kept with its error (`answer_status="error"`) and left out of the agreement.
        """
        requests = self.plan()
        by_id = {q.question_id: q for q in self.questions}

        def call(request: ConsistencyRequest) -> dict:
            started, answer, error = time.perf_counter(), None, None
            try:
                answer = ask(request.text)
            except Exception as exc:  # noqa: BLE001 - one failed request must not stop the probe
                error = f"{type(exc).__name__}: {exc}"
            status, normalized = (ANSWER_ERROR, None) if error else self._read(answer, by_id[request.question_id])
            return {
                "question_id": request.question_id,
                "variant": request.variant,
                "repeat": request.repeat,
                "answer": answer if isinstance(answer, str) else None,
                "normalized": normalized,
                "answer_status": status,
                "error": error,
                "latency_s": time.perf_counter() - started,
            }

        rows = []
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for i, row in enumerate(pool.map(call, requests), start=1):
                rows.append(row)
                if progress:
                    progress(i, len(requests), row)
        answers = pd.DataFrame(rows, columns=_ANSWER_COLUMNS)
        questions = self.score(answers)
        if not self.keep_answers:
            answers = answers.drop(columns=["answer"])
        if self.pii_policy is not None:
            columns = [c for c in ("answer", "normalized") if c in answers]
            answers = self.pii_policy.model_copy(update={"columns": columns}).apply(answers)
            questions = self.pii_policy.model_copy(update={"columns": ["majority_answer"], "count_column": None}).apply(questions)
        return ConsistencyResult(answers=answers, questions=questions)

    def _pair(self, a: dict, b: dict, categorical: bool) -> float:
        statuses = (a["answer_status"], b["answer_status"])
        if ANSWER_UNPARSED in statuses:
            return 0.0
        if ANSWER_REFUSAL in statuses:
            return float(statuses == (ANSWER_REFUSAL, ANSWER_REFUSAL))
        if categorical:
            return float(a["normalized"] == b["normalized"])
        return float(self.similarity(a["answer"], b["answer"]))

    def score(self, answers: pd.DataFrame) -> pd.DataFrame:
        """One row per question from the `answers` frame of `run` (it needs the `answer` column for free text)

        Answers with their text are read again with this probe's questions (answer type, labels, `normalize`), so a
        frame collected for short answers can be scored as free text by another probe; without the text, the stored
        `answer_status` and `normalized` are used.

        Columns: question_id, answer_type, n_answers (calls that answered, refusals and unparsed included), n_errors,
        n_refusals, n_unparsed, n_pairs, agreement (mean pair score), agreement_repeats (pairs of the original
        wording), agreement_paraphrases (pairs original x paraphrase), min_pair, full_agreement (every pair agrees),
        majority_answer and majority_share (short answers), correct_share (when `expected` is known),
        agreement_method, pair_scores (list).
        """
        out = []
        for q in self.questions:
            rows = answers[answers["question_id"] == q.question_id].to_dict("records") if len(answers) else []
            usable = [r for r in rows if r["answer_status"] != ANSWER_ERROR]
            for r in usable:
                if isinstance(r.get("answer"), str):
                    # read again with this probe's question: a frame can be scored under another answer type
                    r["answer_status"], r["normalized"] = self._read(r["answer"], q)
            categorical = q.answer_type in CATEGORICAL_TYPES or self.normalize is not None
            pairs = []
            for a, b in combinations(usable, 2):
                kind = "repeats" if a["variant"] == b["variant"] == ORIGINAL else "paraphrases"
                if ORIGINAL not in (a["variant"], b["variant"]):
                    kind = "other"
                pairs.append((kind, self._pair(a, b, categorical)))
            scores = [s for _, s in pairs]
            threshold = 1.0 if categorical else self.agreement_threshold
            values = [REFUSAL_VALUE if r["answer_status"] == ANSWER_REFUSAL else r["normalized"] for r in usable]
            counted = pd.Series([v for v in values if v is not None], dtype=object).value_counts()
            expected = q.expected
            if expected is not None and q.answer_type in CATEGORICAL_TYPES:
                expected = normalize_answer(expected, q.answer_type, q.labels) or expected

            def mean(kind=None):
                chosen = [s for k, s in pairs if kind is None or k == kind]
                return sum(chosen) / len(chosen) if chosen else None

            out.append(
                {
                    "question_id": q.question_id,
                    "answer_type": q.answer_type,
                    "n_answers": len(usable),
                    "n_errors": len(rows) - len(usable),
                    "n_refusals": sum(r["answer_status"] == ANSWER_REFUSAL for r in usable),
                    "n_unparsed": sum(r["answer_status"] == ANSWER_UNPARSED for r in usable),
                    "n_pairs": len(scores),
                    "agreement": mean(),
                    "agreement_repeats": mean("repeats"),
                    "agreement_paraphrases": mean("paraphrases"),
                    "min_pair": min(scores) if scores else None,
                    "full_agreement": (min(scores) >= threshold) if scores else None,
                    "majority_answer": counted.index[0] if categorical and len(counted) else None,
                    "majority_share": (int(counted.iloc[0]) / len(usable)) if categorical and len(counted) else None,
                    "correct_share": (
                        sum(v == expected for v in values) / len(usable) if expected is not None and usable else None
                    ),
                    "agreement_method": "exact" if categorical else getattr(self.similarity, "name", "custom"),
                    "pair_scores": scores,
                }
            )
        return pd.DataFrame(out, columns=_QUESTION_COLUMNS)


_ANSWER_COLUMNS = ["question_id", "variant", "repeat", "answer", "normalized", "answer_status", "error", "latency_s"]
_QUESTION_COLUMNS = [
    "question_id",
    "answer_type",
    "n_answers",
    "n_errors",
    "n_refusals",
    "n_unparsed",
    "n_pairs",
    "agreement",
    "agreement_repeats",
    "agreement_paraphrases",
    "min_pair",
    "full_agreement",
    "majority_answer",
    "majority_share",
    "correct_share",
    "agreement_method",
    "pair_scores",
]
