"""Detectors of LLM applications and agents, run in-process: only their scores reach the spans, never the text.

- `InjectionDetector`: probability that a text is a prompt injection or a jailbreak; default model
  protectai/deberta-v3-base-prompt-injection-v2 (Apache 2.0).
- `ModelPiiDetector`: personal information the rules miss (names, street addresses), merged with the rule-based
  findings; default model urchade/gliner_multi_pii-v1 (Apache 2.0).
- `FaithfulnessScorer`: how much an answer is supported by the retrieved passages (multilingual NLI entailment);
  default model MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 (MIT).

The models are downloaded once from the Hugging Face hub and run on CPU or GPU (extra `detectors`:
`pip install "pulsar-data-collection[detectors]"`). Any other classifier can be plugged in: a detector is an object
with a `name` and a `score(text)` (or `score(answer, context)`) method; `record_score(span, detector_name, value)`
writes `pulsar.detector.<name>` on a span, and the `agent_run`, `llm_call` and `tool_call` records gain the field
`det_<name>` (maximum over the run). Prompt Guard 2 (Meta, Llama licence, gated) fits the same interface.
"""

import re
from typing import Optional, Sequence

from ..privacy import Finding, PiiDetector

DETECTOR_PREFIX = "pulsar.detector."


def record_score(span, name: str, value: float) -> None:
    """Write a detector score on a span; the highest value wins if the span is scored several times"""
    key = DETECTOR_PREFIX + name
    current = (getattr(span, "attributes", None) or {}).get(key)
    if current is None or value > current:
        span.set_attribute(key, float(value))


def _pipeline(task: str, model: str, device=None, **kwargs):
    try:
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError('this detector needs the "detectors" extra: pip install "pulsar-data-collection[detectors]"') from exc
    return pipeline(task, model=model, device=device, **kwargs)


class InjectionDetector:
    """Probability that a text tries to override the instructions (prompt injection, jailbreak)"""

    name = "injection"

    def __init__(self, model: str = "protectai/deberta-v3-base-prompt-injection-v2", device=None, max_length: int = 512):
        self._classify = _pipeline("text-classification", model, device=device, truncation=True, max_length=max_length)

    def score(self, text: str) -> float:
        if not text:
            return 0.0
        result = self._classify(text)[0]
        label = result["label"].upper()
        return float(result["score"] if label in ("INJECTION", "JAILBREAK", "LABEL_1", "MALICIOUS") else 1 - result["score"])


class ModelPiiDetector:
    """Rule-based findings (validated identifiers) plus a named-entity model for names and addresses"""

    LABELS = {"person": "person_name", "street address": "street_address", "date of birth": "date_of_birth"}

    def __init__(self, model: str = "urchade/gliner_multi_pii-v1", threshold: float = 0.5, rules: Optional[PiiDetector] = None):
        try:
            from gliner import GLiNER
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError('ModelPiiDetector needs the "detectors" extra') from exc
        self._model = GLiNER.from_pretrained(model)
        self.threshold = threshold
        self.rules = rules or PiiDetector()

    def scan(self, text: str) -> list:
        findings = self.rules.scan(text)
        entities = self._model.predict_entities(text, list(self.LABELS), threshold=self.threshold) if text else []
        for e in entities:
            candidate = Finding(self.LABELS[e["label"]], e["start"], e["end"], float(e["score"]))
            if all(candidate.end <= f.start or candidate.start >= f.end for f in findings):
                findings.append(candidate)
        return sorted(findings, key=lambda f: f.start)


_SENTENCES = re.compile(r"(?<=[.!?])\s+")


class FaithfulnessScorer:
    """Share of the answer supported by the context: mean, over the answer's sentences, of the probability that the
    retrieved passages (joined) entail the sentence (1 = every sentence is supported)"""

    name = "faithfulness"

    def __init__(self, model: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", device=None):
        self._nli = _pipeline("text-classification", model, device=device, truncation=True, max_length=512, top_k=None)

    def entailment(self, premise: str, hypothesis: str) -> float:
        scores = self._nli({"text": premise, "text_pair": hypothesis})
        return float(next((s["score"] for s in scores if s["label"].lower().startswith("entail")), 0.0))

    def score(self, answer: str, context: Sequence[str], max_sentences: int = 6) -> Optional[float]:
        """Mean entailment of the answer's sentences (at most `max_sentences`) by the retrieved passages, joined"""
        sentences = [s for s in _SENTENCES.split(answer.strip()) if len(s.split()) >= 3][:max_sentences]
        premise = " ".join(p for p in context if p)
        if not sentences or not premise:
            return None  # nothing to judge: no claim, or nothing retrieved
        outputs = self._nli([{"text": premise, "text_pair": s} for s in sentences], batch_size=16)
        if outputs and isinstance(outputs[0], dict):
            outputs = [outputs]  # a single pair comes back unwrapped
        entail = [float(next((x["score"] for x in out if x["label"].lower().startswith("entail")), 0.0)) for out in outputs]
        return sum(entail) / len(entail)
