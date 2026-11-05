"""Collection for LLM applications and agents, from OpenTelemetry GenAI spans (requires the `genai` extra)."""

from ..fingerprint import EndpointFingerprint
from .abstention import (
    AbstentionDetector,
    AnswerSupport,
    NliAbstentionFallback,
    SupportThresholds,
    answer_support,
    calibrate_support_thresholds,
    record_answer_support,
)
from .canary import CanaryProbe
from .consistency import (
    ConsistencyProbe,
    ConsistencyQuestion,
    ConsistencyResult,
    EmbeddingSimilarity,
    EntailmentSimilarity,
    LexicalSimilarity,
    normalize_answer,
    openai_chat_target,
)
from .detectors import (
    ContentSafetyDetector,
    FaithfulnessScorer,
    InjectionDetector,
    ModelPiiDetector,
    ScoreCalibrator,
    TextClassifierDetector,
    calibrate_threshold,
    french_benign_calibration_set,
    guess_language,
    record_score,
)
from .exporter import ListSink, OpenRunTracker, PulsarSpanExporter, PulseSink
from .instrument import agent_span, llm_span, record_event, record_llm_response, record_tool_result, tool_span
from .probes import ComparativeProbe, CounterfactualProbe, Probe
from .semconv import SCHEMA_VERSION
from .trajectory import TrajectoryAssembler

__all__ = [
    "AbstentionDetector",
    "AnswerSupport",
    "NliAbstentionFallback",
    "SupportThresholds",
    "answer_support",
    "calibrate_support_thresholds",
    "record_answer_support",
    "CanaryProbe",
    "ConsistencyProbe",
    "ConsistencyQuestion",
    "ConsistencyResult",
    "EmbeddingSimilarity",
    "EndpointFingerprint",
    "EntailmentSimilarity",
    "LexicalSimilarity",
    "ContentSafetyDetector",
    "FaithfulnessScorer",
    "InjectionDetector",
    "ModelPiiDetector",
    "ScoreCalibrator",
    "TextClassifierDetector",
    "ComparativeProbe",
    "CounterfactualProbe",
    "ListSink",
    "OpenRunTracker",
    "PulsarSpanExporter",
    "Probe",
    "PulseSink",
    "SCHEMA_VERSION",
    "TrajectoryAssembler",
    "agent_span",
    "calibrate_threshold",
    "french_benign_calibration_set",
    "guess_language",
    "llm_span",
    "normalize_answer",
    "openai_chat_target",
    "record_event",
    "record_llm_response",
    "record_score",
    "record_tool_result",
    "tool_span",
]
