import pytest

from pulsar_data_collection.genai import (
    AbstentionDetector,
    NliAbstentionFallback,
    SupportThresholds,
    answer_support,
    calibrate_support_thresholds,
    record_answer_support,
)
from pulsar_data_collection.genai.abstention import normalize

# synthetic answers written for these tests
ABSTENTIONS = [
    "Je n'ai trouvé aucune information sur la date d'ouverture du musée.",
    "Je suis désolé, mais les passages renvoyés ne contiennent pas d'informations sur ce traité.",
    "Il n'y a pas d'informations disponibles sur la population de cette ville.",
    "Les informations demandées ne sont pas disponibles dans les documents trouvés.",
    "Le nom du premier directeur n’est pas mentionné dans les passages retournés.",
    "Il n'est pas possible de trouver d'informations sur ce programme dans les documents fournis.",
    "Le système d'IA « Outil de tri » n'est pas dans le registre.",
    "Désolé. Je ne peux pas répondre à cette question.",
    "Il semble que les documents de recherche n'ont pas trouvé d'informations sur ce sujet. Puis-je vous aider ?",
    "I could not find any information about the opening date of the museum.",
    'The register does not contain information on a system named "Sorting tool".',
    'The current status of the AI system "Sorting tool" is not in the register.',
    "Sorry. I cannot answer this question from the provided documents.",
    "There is no information about this treaty in the passages. Please rephrase your question.",
    # an abbreviation does not end the sentence: "Outil Inc. »" is not read as content
    "Puisque aucun résultat n'a été trouvé pour « Outil Inc. », je dois répondre :\n\n"
    "Le système « Outil Inc. » n'est pas dans le registre.",
]
ANSWERS = [
    "Le musée a ouvert ses portes en 1923.",
    "Non.",
    "Le système d'IA « Outil de tri » n'implique pas de renseignements personnels.",
    "Ce lac n'est pas un réservoir artificiel, mais un lac glaciaire.",  # denies a fact, does not report missing data
    "Il n'y a pas d'oxygène au fond du lac, ce qui ralentit la décomposition.",
    'The AI system "Sorting tool" does not involve personal information.',
    "The museum opened in 1923, according to the register.",
    "Le système est hors service, car il n'y a pas d'enregistrement récent dans le registre.",  # answer with its reason
    "Le traité a été signé par trois pays, mais il n'y a pas de mention de sa ratification dans les documents.",
]
HEDGED = [
    "La date n'est pas explicitement mentionnée dans les passages. Cependant, on peut en déduire qu'il a ouvert vers 1920.",
    "Je n'ai pas trouvé d'information précise sur ce traité. Cependant, je peux vous dire qu'il a été signé en 1815.",
    "Il semble que les documents de recherche n'ont pas trouvé d'informations sur le sujet. Le lac mesure 12 km de long.",
    "Il n'est pas possible de déterminer le nombre exact de membres dans les documents, mais ils étaient environ trois cents.",
    "I could not find this in the documents. However, the museum most likely opened in the 1920s.",
]


@pytest.mark.parametrize("answer", ABSTENTIONS)
def test_abstentions_are_detected_in_french_and_english(answer):
    result = AbstentionDetector().detect(answer)
    assert result.abstained and result.kind == "abstention" and result.pattern is not None


@pytest.mark.parametrize("answer", ANSWERS)
def test_answers_are_not_abstentions(answer):
    result = AbstentionDetector().detect(answer)
    assert not result.abstained and result.kind == "answer"


@pytest.mark.parametrize("answer", HEDGED)
def test_an_answer_given_after_saying_nothing_was_found_is_hedged(answer):
    result = AbstentionDetector().detect(answer)
    assert not result.abstained and result.kind == "hedged"


def test_no_answer_text_gives_no_verdict():
    detector = AbstentionDetector()
    assert detector.detect(None) is None and detector.detect("  ") is None
    assert detector.score("") == 0.0


def test_normalize_removes_accents_case_and_typographic_apostrophes():
    assert normalize("Je n’ai TROUVÉ  aucune") == "je n'ai trouve aucune"


def test_extra_patterns_extend_the_rules():
    answer = "Cette question sort du périmètre des fiches."
    assert not AbstentionDetector().detect(answer).abstained
    detector = AbstentionDetector(extra_patterns=[("fr_out_of_scope", r"\bsort du perimetre\b")])
    assert detector.detect(answer).pattern == "fr_out_of_scope"


def test_fallback_scores_only_what_the_rules_leave_undecided():
    calls = []

    def fallback(answer):
        calls.append(answer)
        return 0.8

    detector = AbstentionDetector(fallback=fallback, fallback_threshold=0.5)
    assert detector.detect(ABSTENTIONS[0]).pattern == "fr_none_found" and not calls
    result = detector.detect("Rien de tel dans ce que j'ai sous les yeux.")
    assert result.abstained and result.pattern == "fallback" and result.score == 0.8 and len(calls) == 1
    assert detector.score("Rien de tel dans ce que j'ai sous les yeux.") == 0.8


def test_nli_fallback_takes_the_best_entailment_of_its_hypotheses():
    seen = []

    def entailment(premise, hypothesis):
        seen.append((premise, hypothesis))
        return 0.9 if hypothesis.startswith("The") else 0.2

    fallback = NliAbstentionFallback(entailment)
    assert fallback("Nothing like that here.") == 0.9
    assert [h for _, h in seen] == list(NliAbstentionFallback.HYPOTHESES)

    class Scorer:
        def entailment(self, premise, hypothesis):
            return 0.3

    assert NliAbstentionFallback.from_scorer(Scorer(), hypotheses=["x"])("y") == 0.3


def test_thresholds_are_lower_quantiles_of_the_reference():
    thresholds = calibrate_support_thresholds(list(range(1, 102)), [0.1 * i for i in range(11)] + [None], quantile=0.1)
    assert thresholds.retrieval_top == pytest.approx(11.0)
    assert thresholds.faithfulness == pytest.approx(0.1)
    assert calibrate_support_thresholds().retrieval_top is None
    with pytest.raises(ValueError):
        calibrate_support_thresholds([1.0], quantile=1.0)


def test_an_answer_is_unsupported_when_retrieval_or_faithfulness_fails():
    thresholds = SupportThresholds(retrieval_top=5.0, faithfulness=0.3)
    assert answer_support(False, retrieval_empty=True, thresholds=thresholds).reasons == ("empty_retrieval",)
    low = answer_support(False, retrieval_empty=False, retrieval_top=2.0, faithfulness=0.1, thresholds=thresholds)
    assert low.unsupported and low.reasons == ("low_retrieval_score", "low_faithfulness")
    ok = answer_support(False, retrieval_empty=False, retrieval_top=9.0, faithfulness=float("nan"), thresholds=thresholds)
    assert ok.unsupported is False and ok.reasons == ()  # a missing faithfulness is not evidence
    abstained = answer_support(True, retrieval_empty=True, thresholds=thresholds)
    assert abstained.unsupported is False and abstained.reasons == ("empty_retrieval",)
    assert answer_support(None, retrieval_empty=True).unsupported is None
    # the application's own relevance check, e.g. the entity named in the question is absent from the passages
    irrelevant = answer_support(False, retrieval_empty=False, retrieval_top=9.0, retrieval_relevant=False, thresholds=thresholds)
    assert irrelevant.unsupported and irrelevant.reasons == ("irrelevant_retrieval",)
    # without thresholds, only an empty retrieval counts
    assert answer_support(False, retrieval_empty=False, retrieval_top=0.0, faithfulness=0.0).unsupported is False


def exporter():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    from pulsar_data_collection.genai import ListSink, PulsarSpanExporter

    sink = ListSink()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(PulsarSpanExporter(sink)))
    return sink, provider.get_tracer("test")


def test_record_answer_support_reaches_the_agent_run_record():
    from pulsar_data_collection.genai import agent_span

    sink, tracer = exporter()
    thresholds = SupportThresholds(retrieval_top=5.0)
    with agent_span(tracer, "assistant") as run:
        support = record_answer_support(
            run, "Le musée a ouvert en 1923.", retrieval_empty=False, retrieval_top=1.0, thresholds=thresholds
        )
    with agent_span(tracer, "assistant") as run:
        record_answer_support(run, ABSTENTIONS[0], retrieval_empty=True, thresholds=thresholds)
    with agent_span(tracer, "assistant") as run:
        record_answer_support(run, None, retrieval_empty=True)
    answered, abstained, empty = sink.of("agent_run")
    assert support.unsupported and support.abstention.kind == "answer"
    assert answered["det_abstention"] == 0.0 and answered["det_unsupported_answer"] == 1.0
    assert abstained["det_abstention"] == 1.0 and abstained["det_unsupported_answer"] == 0.0
    assert "det_abstention" not in empty and "det_unsupported_answer" not in empty


def test_detector_record_writes_the_abstention_score():
    from pulsar_data_collection.genai import agent_span

    sink, tracer = exporter()
    with agent_span(tracer, "assistant") as run:
        AbstentionDetector().record(run, ABSTENTIONS[9])
    assert sink.of("agent_run")[0]["det_abstention"] == 1.0
