"""Detectors of LLM applications and agents, run in-process: only their scores reach the spans, never the text.

- `InjectionDetector`: score that a text is a prompt injection or a jailbreak; default model
  protectai/deberta-v3-base-prompt-injection-v2 (Apache 2.0). Per-language alert thresholds, fitted on benign
  traffic with `calibrate()`.
- `ContentSafetyDetector`: empty slot for a content-safety (harmful request / toxicity) classifier of your choice;
  `InjectionDetector` does not cover harmful content.
- `ModelPiiDetector`: personal information the rules miss (names, street addresses), merged with the rule-based
  findings; default model urchade/gliner_multi_pii-v1 (Apache 2.0), run on overlapping word windows.
- `FaithfulnessScorer`: how much an answer is supported by the retrieved passages (multilingual NLI entailment on
  overlapping premise windows); default model MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 (MIT).

The scores are **ranks, not calibrated probabilities**: fit an alert threshold on a benign reference of the deployment
(`calibrate_threshold`, A/A) or, with labelled data, a calibrator (`ScoreCalibrator`).

The models are downloaded once from the Hugging Face hub and run on CPU or GPU (extra `detectors`:
`pip install "pulsar-data-collection[detectors]"`). Any other classifier can be plugged in: a detector is an object
with a `name` and a `score(text)` (or `score(answer, context)`) method; `record_score(span, detector_name, value)`
writes `pulsar.detector.<name>` on a span, and the `agent_run`, `llm_call` and `tool_call` records gain the field
`det_<name>` (maximum over the run). Prompt Guard 2 (Meta, Llama licence, gated) fits the same interface.
"""

import inspect
import math
import re
import unicodedata
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from ..privacy import Finding, PiiDetector

DETECTOR_PREFIX = "pulsar.detector."


def record_score(span, name: str, value: float) -> None:
    """Write a detector score on a span; the highest value wins if the span is scored several times"""
    key = DETECTOR_PREFIX + name
    current = (getattr(span, "attributes", None) or {}).get(key)
    if current is None or value > current:
        span.set_attribute(key, float(value))


def resolve_device(device):
    """Device argument: None (library default, CPU), "auto" (first GPU if any), "cpu", "cuda", "cuda:1", or an index"""
    if device != "auto":
        return device
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover - depends on the environment
        return "cpu"


def _pipeline(task: str, model: str, device=None, **kwargs):
    try:
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError('this detector needs the "detectors" extra: pip install "pulsar-data-collection[detectors]"') from exc
    return pipeline(task, model=model, device=resolve_device(device), **kwargs)


# --------------------------------------------------------------------------------------------------------------------
# Thresholds and calibration (the scores are ranks, not probabilities)


@dataclass(frozen=True)
class ThresholdCalibration:
    """Alert threshold fitted on a benign (A/A) reference: alert when the score is above `threshold`
    (`direction="high"`) or below it (`direction="low"`)"""

    threshold: float
    target_fpr: float
    n: int
    fit_fpr: float  # share of the reference that would raise the alert
    direction: str = "high"
    recall: Optional[float] = None  # share of the given positives that raise the alert, if any were given
    n_positives: int = 0
    sufficient: bool = True  # False when the reference is too small for the target (n + 1 < 1 / target_fpr)


def calibrate_threshold(
    scores: Sequence[float], target_fpr: float = 0.01, direction: str = "high", positives: Optional[Sequence[float]] = None
) -> ThresholdCalibration:
    """Alert threshold whose false-alarm rate on the benign reference `scores` is at most `target_fpr`

    The threshold is an order statistic of the reference: with `direction="high"` the alert is `score > threshold`
    (injection, content safety), with `direction="low"` it is `score < threshold` (faithfulness). Give `positives`
    (scores of known positives) to read the recall that the threshold costs. With fewer than 1 / target_fpr - 1
    reference scores the target cannot be reached: the most extreme score is used and `sufficient` is False.
    """
    if direction not in ("high", "low"):
        raise ValueError('direction must be "high" or "low"')
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must be in (0, 1)")
    sign = -1.0 if direction == "low" else 1.0  # "low" is handled as "high" on the negated scores
    values = sorted(sign * float(s) for s in scores if s is not None and not math.isnan(s))
    if not values:
        raise ValueError("no reference score")
    n = len(values)
    allowed = math.floor(target_fpr * (n + 1))  # number of reference scores allowed above the threshold
    sufficient = allowed >= 1
    threshold = values[max(n - 1 - allowed, 0)] if sufficient else values[-1]
    fit_fpr = sum(v > threshold for v in values) / n
    recall = None
    pos = [float(p) for p in (positives or []) if p is not None and not math.isnan(p)]
    if pos:
        recall = sum((-p if direction == "low" else p) > threshold for p in pos) / len(pos)
    if not sufficient:
        warnings.warn(
            f"{n} reference scores cannot support a false-alarm rate of {target_fpr}: at least {math.ceil(1 / target_fpr) - 1} "
            "are needed; the most extreme score is used as threshold",
            stacklevel=2,
        )
    return ThresholdCalibration(
        threshold=-threshold if direction == "low" else threshold,
        target_fpr=target_fpr,
        n=n,
        fit_fpr=fit_fpr,
        direction=direction,
        recall=recall,
        n_positives=len(pos),
        sufficient=sufficient,
    )


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


class ScoreCalibrator:
    """Map raw detector scores to probabilities, fitted on labelled data of the deployment

    `method="isotonic"` (pool-adjacent-violators, monotone, needs a few hundred labels) or `"platt"` (logistic
    regression on the logit of the score, two parameters, works with fewer labels). Calling the calibrator on a score
    returns the calibrated probability.
    """

    def __init__(self, method: str = "isotonic"):
        if method not in ("isotonic", "platt"):
            raise ValueError('method must be "isotonic" or "platt"')
        self.method = method
        self._x: List[float] = []
        self._y: List[float] = []
        self._ab = (1.0, 0.0)

    @staticmethod
    def _logit(p: float, eps: float = 1e-7) -> float:
        p = min(max(float(p), eps), 1 - eps)
        return math.log(p / (1 - p))

    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> "ScoreCalibrator":
        pairs = sorted((float(s), float(bool(y))) for s, y in zip(scores, labels))
        if len(pairs) < 2 or len({y for _, y in pairs}) < 2:
            raise ValueError("calibration needs at least one positive and one negative label")
        if self.method == "platt":
            self._ab = self._fit_platt([self._logit(s) for s, _ in pairs], [y for _, y in pairs])
            return self
        tied: Dict[float, list] = {}  # equal scores are pooled first
        for x, y in pairs:
            tied.setdefault(x, [0.0, 0.0])
            tied[x][0] += y
            tied[x][1] += 1.0
        blocks: List[list] = []  # [sum_y, weight, min_x, max_x]
        for x, (sy, w) in sorted(tied.items()):
            blocks.append([sy, w, x, x])
            while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] >= blocks[-1][0] / blocks[-1][1]:
                y2, w2, _, hi = blocks.pop()
                blocks[-1][0] += y2
                blocks[-1][1] += w2
                blocks[-1][3] = hi
        self._x, self._y = [], []
        for sy, w, lo, hi in blocks:
            for x in (lo, hi) if hi > lo else (lo,):
                self._x.append(x)
                self._y.append(sy / w)
        return self

    @staticmethod
    def _fit_platt(x: List[float], y: List[float], iterations: int = 100) -> tuple:
        mean = min(max(sum(y) / len(y), 1e-6), 1 - 1e-6)
        a, b = 0.0, math.log(mean / (1 - mean))  # start from the base rate: no saturated first step
        for _ in range(iterations):  # Newton on the log-likelihood, with a small ridge for separable data
            ga = gb = haa = hab = hbb = 0.0
            for xi, yi in zip(x, y):
                p = _sigmoid(a * xi + b)
                r, w = p - yi, p * (1 - p)
                ga, gb = ga + r * xi, gb + r
                haa, hab, hbb = haa + w * xi * xi, hab + w * xi, hbb + w
            ga, haa, hbb = ga + 1e-3 * a, haa + 1e-3, hbb + 1e-3
            det = haa * hbb - hab * hab
            if det <= 0:
                break
            da, db = (hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det
            step = max(abs(da), abs(db))
            if step > 5:  # damped Newton: no overshoot on near-separable data
                da, db = da * 5 / step, db * 5 / step
            a, b = a - da, b - db
            if abs(da) + abs(db) < 1e-9:
                break
        return a, b

    def __call__(self, score: float) -> float:
        if self.method == "platt":
            a, b = self._ab
            return _sigmoid(a * self._logit(score) + b)
        if not self._x:
            raise RuntimeError("the calibrator is not fitted")
        s = float(score)
        if s <= self._x[0]:
            return self._y[0]
        if s >= self._x[-1]:
            return self._y[-1]
        lo, hi = 0, len(self._x) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self._x[mid] <= s:
                lo = mid
            else:
                hi = mid
        x0, x1, y0, y1 = self._x[lo], self._x[hi], self._y[lo], self._y[hi]
        return y0 if x1 == x0 else y0 + (y1 - y0) * (s - x0) / (x1 - x0)


# --------------------------------------------------------------------------------------------------------------------
# Language of a request (for per-language thresholds)

_FR_WORDS = frozenset(
    "le la les un une des du de et est sont pour dans avec sur pas que qui quel quelle quels quelles je j vous nous "
    "mon ma mes votre vos notre nos ce cette ces ou où au aux il elle ils elles suis êtes avez ai fait faire "
    "bonjour merci dossier demande aide combien comment pourquoi".split()
)
_EN_WORDS = frozenset(
    "the a an and is are for in with on not that which what who i you we my your our this these those or at it he she "
    "they am have has do does please hello thanks file request help how why much can could would should".split()
)
_TOKENS = re.compile(r"[a-zàâäçéèêëîïôöùûüÿœæ]+", re.IGNORECASE)


def guess_language(text: str) -> str:
    """'fr', 'en' or 'other', from function words and accents (a cheap heuristic; plug a real identifier if needed)"""
    words = _TOKENS.findall((text or "").lower())
    if not words:
        return "other"
    fr = sum(w in _FR_WORDS for w in words) + 2 * sum(any(c in "àâçéèêëîïôùûœ" for c in w) for w in words)
    en = sum(w in _EN_WORDS for w in words)
    if fr == en == 0:
        return "other"
    return "fr" if fr > en else "en"


_FR_BENIGN_TEMPLATES = {
    "file": [
        "Ouvrez-moi un dossier pour {topic}.",
        "Ouvrez un dossier à mon nom pour {topic}, s'il vous plaît.",
        "Créez-moi une demande de {topic}.",
        "Enregistrez ma demande de {topic} et confirmez-moi le numéro.",
        "Inscrivez-moi au programme de {topic}.",
        "Mettez à jour mon dossier de {topic} avec ma nouvelle adresse.",
        "Annulez ma demande de {topic}, je n'en ai plus besoin.",
        "Transmettez mon dossier de {topic} à un agent.",
    ],
    "eligibility": [
        "Suis-je admissible à {topic} avec un revenu de {amount} $ ?",
        "Vérifiez si j'ai droit à {topic}.",
        "Dites-moi si mon ménage est admissible à {topic}.",
    ],
    "amount": [
        "Calculez le montant de {topic} pour un revenu de {amount} $.",
        "Combien vais-je recevoir pour {topic} ?",
        "Donnez-moi une estimation de {topic} pour {amount} $ de revenu annuel.",
    ],
    "rules": [
        "Expliquez-moi les règles de {topic}.",
        "Quels documents faut-il fournir pour {topic} ?",
        "Résumez les conditions de {topic} en trois points.",
        "Ignorez les détails et donnez-moi seulement la date limite pour {topic}.",
    ],
    "general": [
        "Bonjour, j'aimerais des informations sur {topic}.",
        "Pouvez-vous m'aider avec {topic} ?",
        "Oubliez ma question précédente : parlez-moi plutôt de {topic}.",
        "Répondez en anglais : qu'est-ce que {topic} ?",
    ],
}
_FR_TOPICS = [
    "l'aide au logement",
    "la subvention à la rénovation",
    "l'allocation pour enfants",
    "la formation professionnelle",
    "le crédit d'impôt pour solidarité",
    "l'aide de dernier recours",
    "la garde subventionnée",
]
_FR_AMOUNTS = ["18 000", "32 500", "47 000"]


def french_benign_calibration_set() -> List[str]:
    """Synthetic benign French requests to an administrative assistant (imperatives included: "Ouvrez-moi un
    dossier", "Ignorez les détails...", "Oubliez ma question..."), to fit a starting French threshold before real A/A
    traffic is available. It is a starter set: calibrate again on the deployment's own benign traffic."""
    texts = []
    for templates in _FR_BENIGN_TEMPLATES.values():
        for i, template in enumerate(templates):
            for j, topic in enumerate(_FR_TOPICS):
                texts.append(template.format(topic=topic, amount=_FR_AMOUNTS[(i + j) % len(_FR_AMOUNTS)]))
    return texts


# --------------------------------------------------------------------------------------------------------------------
# Text classifiers


class TextClassifierDetector:
    """A Hugging Face text classifier as a detector: score = probability mass of the `positive_labels`

    Parameters
    ----------
    model : str
        Model id or local path (text-classification)
    device : optional
        None (CPU), "auto", "cpu", "cuda", "cuda:1" or a GPU index
    max_length : int, optional
        Longer texts are truncated (default 512 tokens)
    name : str, optional
        Detector name, i.e. the span attribute `pulsar.detector.<name>` and the record field `det_<name>`
    positive_labels : sequence of str, optional
        Labels counted as positive (upper case compared)
    threshold : float, optional
        Alert threshold for every language without its own (default 0.5); the score is a rank, see `calibrate`
    thresholds : dict, optional
        Per-language alert thresholds, e.g. {"fr": 0.99999}
    language : callable, optional
        `language(text) -> str` used when `flag` is not given the language (default `guess_language`)
    calibrator : callable, optional
        Applied to the raw score (e.g. a fitted `ScoreCalibrator`); thresholds then apply to the calibrated score
    from_logits : bool, optional
        Compute the score from the model's logits in double precision (default False: the pipeline's single-precision
        softmax). Above about 0.9999998 the single-precision scores are equal: benign texts and attacks that the model
        scores near 1 tie, and no threshold can separate them. Recommended before fitting a threshold
    """

    name = "classifier"
    positive_labels: Sequence[str] = ("LABEL_1",)

    def __init__(
        self,
        model: str,
        device=None,
        max_length: int = 512,
        *,
        name: Optional[str] = None,
        positive_labels: Optional[Sequence[str]] = None,
        threshold: float = 0.5,
        thresholds: Optional[Dict[str, float]] = None,
        language: Optional[Callable[[str], str]] = None,
        calibrator: Optional[Callable[[float], float]] = None,
        from_logits: bool = False,
        **pipeline_kwargs,
    ):
        if name is not None:
            self.name = name
        if positive_labels is not None:
            self.positive_labels = tuple(positive_labels)
        self._positive = {label.upper() for label in self.positive_labels}
        self.model = model
        self.threshold = threshold
        self.thresholds: Dict[str, float] = dict(thresholds or {})
        self.language = language or guess_language
        self.calibrator = calibrator
        self.calibrations: Dict[str, ThresholdCalibration] = {}
        self.from_logits = from_logits
        if from_logits:
            pipeline_kwargs.update(top_k=None, function_to_apply="none")
        self._all_labels = "top_k" in pipeline_kwargs
        self._classify = _pipeline(
            "text-classification", model, device=device, truncation=True, max_length=max_length, **pipeline_kwargs
        )

    def _positive_score(self, result) -> float:
        if self.from_logits:  # softmax of the logits, in double precision
            top = max(r["score"] for r in result)
            weights = {r["label"].upper(): math.exp(float(r["score"]) - top) for r in result}
            total = sum(weights.values())
            return sum(w for label, w in weights.items() if label in self._positive) / total
        if isinstance(result, list):  # every label (top_k=None)
            return float(sum(r["score"] for r in result if r["label"].upper() in self._positive))
        return float(result["score"] if result["label"].upper() in self._positive else 1 - result["score"])

    def raw_score(self, text: str) -> float:
        """Score of the model, before any calibrator"""
        if not text:
            return 0.0
        if self._all_labels:  # a list in, one list of every label out per text
            return self._positive_score(self._classify([text])[0])
        return self._positive_score(self._classify(text)[0])

    def score(self, text: str) -> float:
        raw = self.raw_score(text)
        return float(self.calibrator(raw)) if self.calibrator is not None and text else raw

    def score_many(self, texts: Sequence[str], batch_size: int = 16) -> List[float]:
        """Scores of several texts, in one batched call"""
        texts = list(texts)
        out = [0.0] * len(texts)
        idx = [i for i, t in enumerate(texts) if t]
        if idx:
            results = self._classify([texts[i] for i in idx], batch_size=batch_size)
            for i, result in zip(idx, results):
                raw = self._positive_score(result)
                out[i] = float(self.calibrator(raw)) if self.calibrator is not None else raw
        return out

    def threshold_for(self, language: Optional[str]) -> float:
        return self.thresholds.get(language, self.threshold) if language else self.threshold

    def flag(self, text: str, language: Optional[str] = None) -> bool:
        """Alert decision for `text`: score above the threshold of its language"""
        language = language or self.language(text)
        return self.score(text) > self.threshold_for(language)

    def calibrate(
        self,
        benign: Sequence[str],
        language: Optional[str] = None,
        target_fpr: float = 0.01,
        positives: Optional[Sequence[str]] = None,
    ) -> ThresholdCalibration:
        """Fit the alert threshold of `language` (or the default one) on benign texts of the deployment (A/A)

        The threshold is set so that at most `target_fpr` of the benign texts raise the alert. Pass known positives to
        read the recall it costs (`.recall`): a tighter threshold for a language also misses more attacks in it.
        """
        scores = self.score_many(benign)
        result = calibrate_threshold(scores, target_fpr, "high", self.score_many(positives) if positives else None)
        if not self.from_logits and result.threshold > 0.9999 and sum(s == result.threshold for s in scores) > 1:
            warnings.warn(
                f"{sum(s == result.threshold for s in scores)} benign scores tie at the threshold {result.threshold}: the "
                "single-precision scores are saturated; create the detector with from_logits=True",
                stacklevel=2,
            )
        if language:
            self.thresholds[language] = result.threshold
        else:
            self.threshold = result.threshold
        self.calibrations[language or "*"] = result
        return result

    def record(self, span, text: str, language: Optional[str] = None) -> float:
        """Score `text` and write it on `span` (`pulsar.detector.<name>`) with the alert decision under the threshold of
        its language (`pulsar.detector.<name>_flag`, 1 or 0); returns the score"""
        language = language or self.language(text)
        value = self.score(text)
        record_score(span, self.name, value)
        record_score(span, self.name + "_flag", 1.0 if value > self.threshold_for(language) else 0.0)
        return value


class InjectionDetector(TextClassifierDetector):
    """Score that a text tries to override the instructions (prompt injection, jailbreak)

    The default model over-flags benign French imperatives ("Ouvrez-moi un dossier"): fit a French threshold with
    `calibrate(benign_fr, language="fr")` (e.g. on `french_benign_calibration_set()`, then on real traffic). It does
    not detect harmful requests without an injection wrapper: see `ContentSafetyDetector`.
    """

    name = "injection"
    positive_labels = ("INJECTION", "JAILBREAK", "LABEL_1", "MALICIOUS")

    def __init__(
        self, model: str = "protectai/deberta-v3-base-prompt-injection-v2", device=None, max_length: int = 512, **kwargs
    ):
        super().__init__(model, device=device, max_length=max_length, **kwargs)


class ContentSafetyDetector(TextClassifierDetector):
    """Slot for a content-safety classifier (harmful requests, toxicity): no default model, choose one whose licence
    and languages fit the deployment, and name its unsafe labels

    >>> ContentSafetyDetector("org/some-safety-classifier", positive_labels=["UNSAFE"])  # doctest: +SKIP

    Every label is read (`top_k=None`) and the score is the sum of the probabilities of `positive_labels`.
    """

    name = "content_safety"

    def __init__(self, model: str, positive_labels: Sequence[str], device=None, max_length: int = 512, **kwargs):
        kwargs.setdefault("top_k", None)
        super().__init__(model, device=device, max_length=max_length, positive_labels=positive_labels, **kwargs)


# --------------------------------------------------------------------------------------------------------------------
# Personal information

_WORD_SPANS = re.compile(r"\S+")


class ModelPiiDetector:
    """Rule-based findings (validated identifiers) plus a named-entity model for names and addresses

    Parameters
    ----------
    model : str
        GLiNER model id or local path
    threshold : float, optional
        Minimum model score (default 0.5)
    rules : PiiDetector, optional
        Rule-based detector (default `PiiDetector()`)
    device : optional
        None (CPU), "auto" (first GPU if any), "cpu", "cuda", "cuda:1"
    thresholds : dict, optional
        Per-label minimum score, by GLiNER label (e.g. {"street address": 0.3}); others use `threshold`
    window_words, overlap_words : int, optional
        GLiNER reads at most 384 words: longer texts are scanned on windows of `window_words` words (default 250)
        overlapping by `overlap_words` (default 50), and positions are mapped back to the whole text
    merge : str, optional
        "enclosing" (default): a model entity that fully encloses rule findings (an address holding its postal code or
        a phone number) replaces them, and a model address overlapping rule-found addresses is merged with them (union
        of the spans); "rules": a model entity overlapping any rule finding is dropped (before 0.10)
    locales, profiles : sequence of str, optional
        Passed to the default rules detector (`PiiDetector(locales=..., profiles=...)`, e.g. `profiles=("address",)`)
        when `rules` is not given; None keeps the `PiiDetector` defaults
    batch_size : int, optional
        Windows per model call (default 8)
    """

    LABELS = {"person": "person_name", "street address": "street_address", "date of birth": "date_of_birth"}

    def __init__(
        self,
        model: str = "urchade/gliner_multi_pii-v1",
        threshold: float = 0.5,
        rules: Optional[PiiDetector] = None,
        *,
        device=None,
        thresholds: Optional[Dict[str, float]] = None,
        window_words: int = 250,
        overlap_words: int = 50,
        merge: str = "enclosing",
        batch_size: int = 8,
        locales: Optional[Sequence[str]] = None,
        profiles: Optional[Sequence[str]] = None,
    ):
        if not 0 <= overlap_words < window_words:
            raise ValueError("overlap_words must be in [0, window_words)")
        if merge not in ("enclosing", "rules"):
            raise ValueError('merge must be "enclosing" or "rules"')
        unknown = set(thresholds or {}) - set(self.LABELS)
        if unknown:
            raise ValueError(f"unknown labels {sorted(unknown)}; known: {list(self.LABELS)}")
        try:
            from gliner import GLiNER
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError('ModelPiiDetector needs the "detectors" extra') from exc
        self._model = GLiNER.from_pretrained(model)
        device = resolve_device(device)
        if device is not None:
            self._model = self._model.to(device)
        self.device = device
        self.threshold = threshold
        self.thresholds = dict(thresholds or {})
        if rules is None:
            options = {k: tuple(v) for k, v in (("locales", locales), ("profiles", profiles)) if v is not None}
            rules = PiiDetector(**options)
        elif locales is not None or profiles is not None:
            raise ValueError("locales and profiles configure the default rules detector: set them on `rules` instead")
        self.rules = rules
        self._rules_scoped: Optional[bool] = None
        self.window_words, self.overlap_words, self.merge, self.batch_size = window_words, overlap_words, merge, batch_size

    def windows(self, text: str) -> List[tuple]:
        """[start, end) character spans of the windows scanned by the model"""
        words = [m.span() for m in _WORD_SPANS.finditer(text)]
        if len(words) <= self.window_words:
            return [(0, len(text))] if words else []
        step, spans = self.window_words - self.overlap_words, []
        for first in range(0, len(words), step):
            last = min(first + self.window_words, len(words)) - 1
            spans.append((words[first][0], words[last][1]))
            if last == len(words) - 1:
                break
        return spans

    def _predict(self, chunks: List[str], labels: List[str], threshold: float) -> List[list]:
        model = self._model
        if callable(getattr(model, "inference", None)):
            return model.inference(chunks, labels, threshold=threshold, batch_size=self.batch_size)
        if callable(getattr(model, "batch_predict_entities", None)):  # gliner < 0.2.20
            return model.batch_predict_entities(chunks, labels, threshold=threshold)
        return [model.predict_entities(chunk, labels, threshold=threshold) for chunk in chunks]

    def model_entities(self, text: str) -> List[Finding]:
        """Entities found by the model alone, on every window, positions in `text`, without overlaps"""
        spans = self.windows(text) if text else []
        if not spans:
            return []
        labels = list(self.LABELS)
        floor = min([self.threshold, *self.thresholds.values()])
        found = []
        for (offset, end), entities in zip(spans, self._predict([text[s:e] for s, e in spans], labels, floor)):
            for e in entities:
                if e["score"] >= self.thresholds.get(e["label"], self.threshold):
                    found.append(Finding(self.LABELS[e["label"]], offset + e["start"], offset + e["end"], float(e["score"])))
        kept: List[Finding] = []  # an entity seen in two overlapping windows is kept once (most confident)
        for f in sorted(found, key=lambda f: (-f.confidence, -(f.end - f.start), f.start)):
            if all(f.end <= k.start or f.start >= k.end for k in kept):
                kept.append(f)
        return sorted(kept, key=lambda f: f.start)

    def _rules_scan(self, text: str, business_scope: str) -> list:
        if business_scope == "window":
            return self.rules.scan(text)
        if self._rules_scoped is None:
            try:
                self._rules_scoped = "business_scope" in inspect.signature(self.rules.scan).parameters
            except (TypeError, ValueError):  # pragma: no cover - builtins without a signature
                self._rules_scoped = False
        return self.rules.scan(text, business_scope=business_scope) if self._rules_scoped else self.rules.scan(text)

    def scan(self, text: str, business_scope: str = "window") -> list:
        """Rule-based findings merged with the model entities, without overlaps, in reading order

        `business_scope` is forwarded to the rules detector when it accepts it (`PiiDetector.scan`: "window" or
        "adjacent", used by `PiiSession` for outputs); a rules detector without it scans one way only.
        """
        findings = self._rules_scan(text, business_scope)
        for candidate in self.model_entities(text):
            overlapping = [f for f in findings if not (candidate.end <= f.start or candidate.start >= f.end)]
            if not overlapping:
                findings.append(candidate)
                continue
            if self.merge != "enclosing":
                continue
            if candidate.type == "street_address":  # an address: the union with the overlapping rule-found addresses
                same = [f for f in overlapping if f.type == "street_address"]
                start, end = min([candidate.start] + [f.start for f in same]), max([candidate.end] + [f.end for f in same])
                confidence = max([candidate.confidence] + [f.confidence for f in same])
                candidate = Finding("street_address", start, end, confidence)
                overlapping = [f for f in findings if not (candidate.end <= f.start or candidate.start >= f.end)]
            if all(candidate.start <= f.start and f.end <= candidate.end for f in overlapping):
                findings = [f for f in findings if f not in overlapping] + [candidate]
        return sorted(findings, key=lambda f: f.start)


# --------------------------------------------------------------------------------------------------------------------
# Faithfulness

_SENTENCES = re.compile(r"(?<=[.!?])\s+")
_CLAUSES = re.compile(r"(?<=[;:,])\s+")
_NORM_WORDS = re.compile(r"\w+")


def _norm_words(text: str) -> List[str]:
    return _NORM_WORDS.findall(unicodedata.normalize("NFKC", text).lower())


def _trigrams(words: List[str]) -> set:
    return {tuple(words[i : i + 3]) for i in range(len(words) - 2)}


@dataclass
class FaithfulnessResult:
    """Detail of a faithfulness score (see `FaithfulnessScorer.details`)"""

    score: Optional[float]  # aggregate (mean by default) over the scored sentences; None if nothing to judge
    min: Optional[float]  # weakest sentence
    sentences: List[float] = field(default_factory=list)  # support of each scored sentence (max over windows)
    lexical: List[bool] = field(default_factory=list)  # sentence supported by the lexical shortcut
    sentences_total: int = 0  # sentences of at least 3 words in the answer
    sentences_scored: int = 0  # at most max_sentences of them
    premise_tokens: int = 0
    windows: int = 0  # premise windows (1 when the premise fits)
    truncated: bool = False  # the premise did not fit and was cut (windowing disabled)

    @property
    def coverage(self) -> Optional[float]:
        """Share of the answer's sentences that were scored"""
        return self.sentences_scored / self.sentences_total if self.sentences_total else None


class FaithfulnessScorer:
    """Share of the answer supported by the context: mean, over the answer's sentences, of the probability that the
    retrieved passages entail the sentence (1 = every sentence is supported). A ranking signal, not a calibrated
    probability: alert on a threshold fitted on an A/A reference (`calibrate_threshold(..., direction="low")`).

    Parameters
    ----------
    model : str
        NLI model id or local path
    device : optional
        None (CPU), "auto", "cpu", "cuda", or a GPU index
    window_tokens, stride_tokens : int, optional
        The joined passages are cut into windows of `window_tokens` tokens (default 400) every `stride_tokens`
        (default 300), and each sentence keeps its best window; `window_tokens=None` restores the single
        512-token premise (the end of long contexts is then silently cut, see `FaithfulnessResult.truncated`)
    max_sentences : int, optional
        Sentences scored per answer (default 6; None = all); `FaithfulnessResult.coverage` tells the share scored
    lexical_threshold : float, optional
        A sentence whose word trigrams are all (default 1.0) found in the passages, i.e. a verbatim copy, is supported
        without the model, which judges long copies unsupported; lower to accept near copies, None to disable
    max_claim_words : int, optional
        Sentences longer than this are split at commas, colons and semicolons before the model (default None: off);
        entailment of long hypotheses falls with their length
    aggregate : str, optional
        "mean" (default) or "min" (weakest sentence) for `score`
    """

    name = "faithfulness"

    def __init__(
        self,
        model: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        device=None,
        *,
        window_tokens: Optional[int] = 400,
        stride_tokens: int = 300,
        max_sentences: Optional[int] = 6,
        lexical_threshold: Optional[float] = 1.0,
        max_claim_words: Optional[int] = None,
        aggregate: str = "mean",
        batch_size: int = 16,
    ):
        if window_tokens is not None and not 0 < stride_tokens <= window_tokens:
            raise ValueError("stride_tokens must be in (0, window_tokens]")
        if aggregate not in ("mean", "min"):
            raise ValueError('aggregate must be "mean" or "min"')
        self._nli = _pipeline("text-classification", model, device=device, truncation=True, max_length=512, top_k=None)
        self.window_tokens, self.stride_tokens = window_tokens, stride_tokens
        self.max_sentences, self.lexical_threshold, self.max_claim_words = max_sentences, lexical_threshold, max_claim_words
        self.aggregate, self.batch_size = aggregate, batch_size

    @staticmethod
    def _entail(output) -> float:
        return float(next((s["score"] for s in output if s["label"].lower().startswith("entail")), 0.0))

    def entailment(self, premise: str, hypothesis: str) -> float:
        return self._entail(self._nli({"text": premise, "text_pair": hypothesis}))

    def _token_spans(self, premise: str) -> List[tuple]:
        tokenizer = getattr(self._nli, "tokenizer", None)
        if tokenizer is not None:
            try:
                return [
                    tuple(o) for o in tokenizer(premise, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
                ]
            except (NotImplementedError, KeyError, TypeError):  # slow tokenizer: no offsets
                pass
        return [m.span() for m in _WORD_SPANS.finditer(premise)]  # without a fast tokenizer, words stand for tokens

    def premise_windows(self, premise: str) -> tuple:
        """(windows, premise tokens): the premise cut into overlapping windows of `window_tokens` tokens"""
        spans = self._token_spans(premise)
        if self.window_tokens is None or len(spans) <= self.window_tokens:
            return [premise], len(spans)
        windows = []
        for first in range(0, len(spans), self.stride_tokens):
            last = min(first + self.window_tokens, len(spans)) - 1
            windows.append(premise[spans[first][0] : spans[last][1]])
            if last == len(spans) - 1:
                break
        return windows, len(spans)

    def _claims(self, answer: str) -> List[str]:
        sentences = [s for s in _SENTENCES.split(answer.strip()) if len(s.split()) >= 3]
        if not self.max_claim_words:
            return sentences
        claims = []
        for s in sentences:
            if len(s.split()) <= self.max_claim_words:
                claims.append(s)
                continue
            current = ""
            for part in _CLAUSES.split(s):
                if current and len((current + " " + part).split()) > self.max_claim_words:
                    claims.append(current)
                    current = part
                else:
                    current = (current + " " + part).strip()
            if current:
                claims.append(current)
        return [c for c in claims if len(c.split()) >= 3]

    def details(self, answer: str, context: Sequence[str], max_sentences=...) -> FaithfulnessResult:
        """Score with its detail: per-sentence support, coverage of the answer, premise windows"""
        max_sentences = self.max_sentences if max_sentences is ... else max_sentences
        claims = self._claims(answer or "")
        premise = " ".join(p for p in context if p)
        scored = claims if max_sentences is None else claims[:max_sentences]
        if not scored or not premise:
            return FaithfulnessResult(None, None, sentences_total=len(claims))  # nothing to judge
        windows, n_tokens = self.premise_windows(premise)
        support = [0.0] * len(scored)
        lexical = [False] * len(scored)
        if self.lexical_threshold is not None:
            grams = _trigrams(_norm_words(premise))
            for i, s in enumerate(scored):
                own = _trigrams(_norm_words(s))
                if own and sum(g in grams for g in own) / len(own) >= self.lexical_threshold:
                    support[i], lexical[i] = 1.0, True
        pairs = [(i, w) for i in range(len(scored)) if not lexical[i] for w in windows]
        if pairs:
            outputs = self._nli([{"text": w, "text_pair": scored[i]} for i, w in pairs], batch_size=self.batch_size)
            if outputs and isinstance(outputs[0], dict):
                outputs = [outputs]  # a single pair comes back unwrapped
            for (i, _), out in zip(pairs, outputs):
                support[i] = max(support[i], self._entail(out))
        mean = sum(support) / len(support)
        return FaithfulnessResult(
            score=min(support) if self.aggregate == "min" else mean,
            min=min(support),
            sentences=support,
            lexical=lexical,
            sentences_total=len(claims),
            sentences_scored=len(scored),
            premise_tokens=n_tokens,
            windows=len(windows),
            truncated=self.window_tokens is None and n_tokens > 512 - 3,
        )

    def score(self, answer: str, context: Sequence[str], max_sentences=...) -> Optional[float]:
        """Support of the answer's sentences (at most `max_sentences`, default the scorer's) by the retrieved passages"""
        return self.details(answer, context, max_sentences).score

    def record(self, span, answer: str, context: Sequence[str]) -> Optional[FaithfulnessResult]:
        """Score and write on `span`: `pulsar.detector.faithfulness`, `..._min`, `..._coverage` and `..._windows`"""
        result = self.details(answer, context)
        if result.score is None:
            return result
        record_score(span, self.name, result.score)
        for suffix, value in (("_min", result.min), ("_coverage", result.coverage), ("_windows", result.windows)):
            span.set_attribute(DETECTOR_PREFIX + self.name + suffix, float(value))
        return result
