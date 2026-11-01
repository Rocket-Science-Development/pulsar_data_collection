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
