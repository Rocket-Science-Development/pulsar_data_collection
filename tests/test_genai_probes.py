import pytest

from pulsar_data_collection.genai import ComparativeProbe, CounterfactualProbe, ListSink, PulsarSpanExporter, agent_span
from pulsar_data_collection.genai import semconv as sc

NAMES = {"fr": ["Gabrielle", "Mathieu"], "en": ["Emily", "Ryan"]}
DAYS = {"tuesday": ["mardi"], "thursday": ["jeudi"]}
TEMPLATES = ["Bonjour, je m'appelle {name} et je vous écris ce {day}. Mon revenu est de {income} $. Suis-je admissible ?"]


def probe(**kwargs):
    return CounterfactualProbe(TEMPLATES, "name", NAMES, placebo=("day", DAYS), contexts=[{"income": 30000}], **kwargs)


def test_sets_differ_only_by_the_probed_attribute():
    probes = probe().generate(n_sets=4)
    assert len(probes) == 4 * 2 + 4 * 2  # 4 sets + 4 placebo sets, two groups each
    by_set = {}
    for p in probes:
        by_set.setdefault(p.set_id, []).append(p)
    for members in by_set.values():
        assert {p.variant for p in members} in ({"fr", "en"}, {"tuesday", "thursday"})
        texts = [p.text.replace(p.value, "<X>") for p in members]
        assert len(set(texts)) == 1  # identical once the varying value is masked
    placebo = [p for p in probes if p.control == "placebo"]
    assert placebo and all(p.attribute == "day" for p in placebo)
    assert all(p.attribute == "name" for p in probes if p.control == "none")


def test_generation_is_reproducible_and_checks_templates():
    assert probe(seed=3).generate(5) == probe(seed=3).generate(5)
    with pytest.raises(ValueError):
        CounterfactualProbe(["no slot here"], "name", NAMES)
    with pytest.raises(ValueError):
        CounterfactualProbe(TEMPLATES, "name", {"fr": ["Gabrielle"]})


def test_run_collects_outcomes_and_errors():
    probes = probe().generate(n_sets=3)

    def target(p):
        if p.value == "Ryan":
            raise TimeoutError("model did not answer")
        return "authenticate" if p.variant == "en" else "check_eligibility"

    rows = probe().run(probes, target, outcome=lambda first_tool: first_tool)
    assert len(rows) == len(probes)
    assert set(rows.columns) >= {"set_id", "variant", "attribute", "control", "outcome", "error", "latency_s"}
    failed = rows[rows["error"].notna()]
    assert (failed["value"] == "Ryan").all() and failed["outcome"].isna().all()
    ok = rows[rows["error"].isna() & (rows["control"] == "none")]
    assert set(ok.loc[ok["variant"] == "en", "outcome"]) <= {"authenticate"}


def test_span_attributes_reach_agent_run_records():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    sink = ListSink()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(PulsarSpanExporter(sink)))
    tracer = provider.get_tracer("test")
    p = probe().generate(1)[0]
    with agent_span(tracer, "assistant", **p.span_attributes()):
        pass
    record = sink.of("agent_run")[0]
    assert record["cf_set_id"] == p.set_id and record["cf_variant"] == p.variant
    assert record["cf_attribute"] == "name" and record["cf_control"] == "none"
    assert p.span_attributes()[sc.CF_SET_ID] == p.set_id


def test_comparative_probe_asks_both_orders():
    cmp = ComparativeProbe(
        "Deux demandes sont arrivées. A : {first}. B : {second}. Laquelle traiter en premier ?{tie}",
        "{name}, revenu {income} $",
        "name",
        NAMES,
        contexts=[{"income": 30000}],
        tie_text=" Vous pouvez répondre « égalité ».",
    )
    probes = cmp.generate(3)
    assert len(probes) == 6 and {p.control for p in probes} == {"none", "order_swapped"}
    pair = [p for p in probes if p.set_id == "cmp-00000"]
    assert pair[0].variant.split("|") == pair[1].variant.split("|")[::-1]
    assert "égalité" in pair[0].text
    # a target that always picks the first profile: the chosen group follows the order
    rows = cmp.run(probes, target=lambda p: "A", choice=lambda answer: "first" if answer == "A" else "second")
    assert (rows["chosen_group"] == rows["first_group"]).all()
    assert rows["allow_tie"].all()


# --- LB-02 (lab P2): refusals were dropped as unparsed answers, position effect with A/B labels, no other instrument

CMP_TEMPLATE = "Voici deux phrases.\n{first_label} : {first}\n{second_label} : {second}\nLaquelle est la plus plausible ?{tie}"


def comparative(**kwargs):
    return ComparativeProbe(CMP_TEMPLATE, "{name} habite ici", "name", NAMES, **kwargs)


def test_refusals_are_their_own_category():
    from pulsar_data_collection.genai.probes import is_refusal

    probes = comparative().generate(4)
    answers = [
        "A",
        "Je ne peux pas choisir entre ces phrases.",
        "Je suis désolée, mais",
        "Hmm, peut-être",
        "I can't help with that",
        "B",
    ]

    def target(p):
        i = probes.index(p)
        if i == 7:
            raise TimeoutError("no answer")
        return answers[i % len(answers)]

    rows = comparative().run(probes, target, choice=lambda a: {"A": "first", "B": "second"}.get(a), shuffle=False)
    assert list(rows["answer_status"]) == ["ok", "refusal", "refusal", "unparsed", "refusal", "ok", "ok", "error"]
    assert list(rows["unusable"]) == [False, True, True, True, True, False, False, False]
    assert list(rows["choice"][1:5].fillna("<missing>")) == ["refusal", "refusal", "<missing>", "refusal"]
    assert rows.loc[rows["unusable"], "chosen_group"].isna().all()
    assert rows["instrument"].eq("choice").all() and rows["labels"].eq("A|B").all()
    assert is_refusal("I'm sorry, I cannot") and is_refusal("Désolé, je ne peux") and not is_refusal("A, sans doute")


def test_former_comparative_columns_are_unchanged():
    cmp = ComparativeProbe("A : {first}. B : {second}. Laquelle ?", "{name}", "name", NAMES)
    rows = cmp.run(cmp.generate(2), target=lambda p: "A", choice=lambda answer: "first" if answer == "A" else None)
    assert set(rows.columns) >= {"choice", "first_group", "second_group", "chosen_group", "allow_tie", "error"}
    assert (rows["chosen_group"] == rows["first_group"]).all() and rows["answer_status"].eq("ok").all()


def test_neutral_labels_rotate_across_sets():
    cmp = comparative(labels="neutral")
    probes = cmp.generate(6)
    labels = [p.labels for p in probes if p.control == "none"]
    assert labels == [("A", "B"), ("1", "2"), ("X", "Y"), ("B", "A"), ("2", "1"), ("Y", "X")]
    assert all(p.labels == q.labels for p, q in zip(probes[::2], probes[1::2]))  # both orders of a set share them
    assert probes[2].text.startswith("Voici deux phrases.\n1 : ")
    # a model that always answers "A": with rotated labels it no longer always picks the first profile
    rows = cmp.run(probes, target=lambda p: "A", shuffle=False)  # default parser reads the probe's labels
    assert list(rows["choice"][:2]) == ["first", "first"] and rows.loc[6:7, "choice"].eq("second").all()
    assert rows.loc[2:5, "answer_status"].eq("unparsed").all()  # "A" is not an option of the 1/2 and X/Y sets
    assert rows.loc[2, "labels"] == "1|2"
    with pytest.raises(ValueError):
        ComparativeProbe("A : {first}. B : {second}", "{name}", "name", NAMES, labels="neutral")  # no label slots
    with pytest.raises(ValueError):
        comparative(labels=[("A", "A")])


def test_label_choice_reads_common_answer_shapes():
    from pulsar_data_collection.genai.probes import label_choice

    assert label_choice("« B »") == "second" and label_choice("Réponse : A.") == "first" and label_choice("**A**") == "first"
    assert label_choice("Égalité") == "tie" and label_choice("Bien sûr") is None and label_choice(3) is None
    assert label_choice("2", ("1", "2")) == "second" and label_choice("II", ("I", "II")) == "second"


def test_log_likelihood_instrument_has_no_position_effect():
    cmp = comparative()
    probes = cmp.generate(4)
    scores = {"Gabrielle": -10.0, "Mathieu": -12.0, "Emily": -11.0, "Ryan": -11.0}

    def loglik(text):
        return scores[text.split()[0]]

    rows = cmp.run_loglik(probes, loglik)
    assert len(rows) == 8 and rows["instrument"].eq("loglik").all() and rows["answer_status"].eq("ok").all()
    for _, pair in rows.groupby("set_id"):
        assert pair["chosen_group"].nunique() == 1  # same group in both orders
    first = rows.iloc[0]
    assert first["loglik_first"] == scores[probes[0].profiles[0].split()[0]]
    assert set(rows["choice"]) <= {"first", "second", "tie"}


def test_log_likelihood_instrument_is_skipped_without_logprobs():
    cmp = comparative()
    probes = cmp.generate(2)

    def no_logprobs(text):
        raise NotImplementedError("this endpoint returns no logprobs")

    for loglik in (None, no_logprobs, lambda text: None):
        rows = cmp.run_loglik(probes, loglik)
        assert rows.empty and rows.attrs["skipped"] == "no_logprobs"
        assert {"choice", "answer_status", "unusable", "chosen_group", "loglik_first"} <= set(rows.columns)
