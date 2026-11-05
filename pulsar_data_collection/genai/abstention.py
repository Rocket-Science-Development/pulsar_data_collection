"""Abstention and answers without support of a RAG agent, per run: the hallucination proxy of a retrieval agent.

- `AbstentionDetector`: the answer says the information is not found, not in the sources, or that it cannot answer
  (French and English). Rules first (`NOT_FOUND_PATTERNS`, `CANNOT_ANSWER_PATTERNS`, `FILLER_PATTERNS`, documented
  below); an optional fallback (e.g. `NliAbstentionFallback`, NLI entailment through `FaithfulnessScorer.entailment`)
  scores the answers the rules leave undecided.
- `answer_support`: an answer without support is an answer (no abstention) given although the retrieval found nothing
  relevant (empty retrieval, best retrieval score under a threshold, or the application's own relevance check failed)
  or although the answer is not supported by the passages (faithfulness under a threshold). Thresholds are fitted on
  a reference with `calibrate_support_thresholds`.
- `record_answer_support(span, answer, ...)` writes `pulsar.detector.abstention` and
  `pulsar.detector.unsupported_answer` (1.0 or 0.0) on a span: the `agent_run` record gains `det_abstention` and
  `det_unsupported_answer`, whose rates per window are the two signals (in pulsar_metrics:
  `DriftTestMetric("rate_increase", "det_unsupported_answer")`, one-sided exact test against the reference).

Only the verdicts reach the spans, never the text.

How the rules decide (on the answer casefolded, accents and typographic apostrophes removed, cut into sentences):

1. leading sentences that only apologise ("Je suis désolé.", "Sorry.") are skipped;
2. the first remaining sentence must state that the information is missing: a `NOT_FOUND_PATTERNS` match in a
   sentence that names a source (`SOURCE_WORDS`: passages, documents, information, register...), or a
   `CANNOT_ANSWER_PATTERNS` match. A statement introduced as a justification ("..., car il n'y a pas d'information",
   "because ...") does not count: the sentence answers and explains why;
3. what follows must not answer: a later clause after a contrast ("mais", "cependant", "however"...) or a later
   sentence that is neither another statement of missing information nor filler (`FILLER_PATTERNS`: an offer of help,
   an invitation to rephrase, related items) makes the answer *hedged*: the agent said it found nothing, then answered
   anyway. A hedged answer is an answer (`abstained` False, `kind` "hedged"), and so can be an answer without support.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from .detectors import record_score

ABSTENTION = "abstention"
UNSUPPORTED_ANSWER = "unsupported_answer"

# Nouns naming where the information should have come from; a NOT_FOUND statement counts only next to one of them
# ("Moab n'est pas un désert" denies a fact, it does not report missing information)
SOURCE_WORDS = re.compile(
    r"\b(?:informations?|passages?|documents?|sources?|contexte?|donnees|data|resultats?|results?|recherches?|search"
    r"|corpus|registre|register|registry|fiches?|enregistrements?|records?|entry|entries|database|base de donnees"
    r"|mentions?|references?|extraits?|excerpts?|renseignements? disponibles?)\b"
)

# Statements that information is missing (casefolded, accents removed). Each entry: (name, pattern)
NOT_FOUND_PATTERNS = (
    # "il n'y a pas d'information", "je n'ai pas trouvé de document", "il n'existe aucune mention"
    (
        "fr_none_found",
        r"\b(?:il n'y a|il n'existe|je n'ai (?:pas )?trouve|je n'ai rien trouve|n'ai trouve|je ne trouve"
        r"|je n'arrive pas a trouver)\b[^.,;]{0,60}?\b(?:informations?|passages?|documents?|donnees|resultats?|mentions?"
        r"|mentionne\w*|reponses?|sources?|references?|traces?|enregistrements?|fiches?|precisions?|details?"
        r"|renseignements?|rien)\b",
    ),
    # "aucune information", "aucun document pertinent"
    (
        "fr_none",
        r"\baucune? (?:\w+ )?(?:informations?|passages?|documents?|donnees|resultats?|mentions?|reponses?|sources?"
        r"|references?|traces?|enregistrements?|fiches?)\b",
    ),
    # "les passages ne contiennent pas", "les informations fournies ne traitent pas", "ne mentionnent aucune"
    (
        "fr_source_lacks",
        r"\bne (?:contien\w*|content\w*|mentionn\w*|fourni\w*|trait\w*|comport\w*|precis\w*|indiqu\w*|parl\w*"
        r"|repond\w*|permet\w*|donn\w*|concern\w*|port\w*|abord\w*) (?:pas|aucun\w*|rien)\b",
    ),
    # "n'est pas (explicitement) mentionné / indiqué / précisé / disponible dans les passages"
    (
        "fr_not_stated",
        r"\b(?:n'est|ne sont|n'etait|n'etaient|ne soi(?:en)?t|n'a ete|n'ont ete) pas (?:\w+ )?(?:mentionne|indique"
        r"|precise|specifie|present|disponible|fourni|trouve|reference|repertorie|liste|inscrit|aborde|traite"
        r"|documente|couvert|repondu)\w*",
    ),
    # "n'est pas dans le registre", "ne figure pas parmi les résultats", "absent du registre"
    (
        "fr_not_in",
        r"\b(?:n'(?:est|sont|apparai\w*|figure\w*) pas|ne (?:figure|apparai)\w* pas) (?:\w+ )?(?:dans|au|aux|parmi)\b"
        r"|\babsente?s? d[eu]s? (?:registre|passages?|documents?|resultats?|sources?)\b",
    ),
    # "il n'est pas possible de trouver / déterminer", "impossible de savoir"
    ("fr_cannot_find", r"\b(?:pas possible|impossible) de (?:trouver|determiner|savoir|identifier|preciser|dire)\b"),
    # "les documents n'ont pas trouvé / fourni d'informations"
    (
        "fr_search_failed",
        r"\bn'(?:a|ont|avaient|avait) (?:pas|rien) (?:trouve|fourni|donne|renvoye|retourne|contenu)\b"
        r"|\bn'(?:a|ont) pas (?:de|d') (?:reponse|information)",
    ),
    # "there is no information", "I could not find any", "no relevant documents"
    (
        "en_none_found",
        r"\b(?:there (?:is|are|was|were) no|i (?:could not|couldn't|cannot|can't|was unable to|am unable to|did not"
        r"|didn't) find|no (?:relevant |specific |such |further |additional )?(?:information|data|records?|results?"
        r"|mentions?|entr(?:y|ies)|passages?|documents?|details))\b",
    ),
    # "the register does not contain / list", "the passages don't mention"
    (
        "en_source_lacks",
        r"\b(?:does|do|did) not (?:contain|list|include|mention|provide|cover|specify|state|have)\b|\b(?:doesn't|don't"
        r"|didn't) (?:contain|list|include|mention|provide|cover|specify|state|have)\b",
    ),
    # "is not in the register", "is not found / listed / mentioned in the documents", "not available"
    (
        "en_not_in",
        r"\b(?:is|are|was|were) not (?:\w+ )?(?:in|found|listed|included|mentioned|available|present|specified|stated"
        r"|provided|covered|part of)\b",
    ),
)

# Statements that the agent cannot answer, valid without a source word
CANNOT_ANSWER_PATTERNS = (
    (
        "fr_cannot_answer",
        r"\b(?:je ne (?:peux|pourrai\w*|suis pas en mesure d\w*) pas|je ne peux|impossible d\w*|pas possible d\w*"
        r"|je ne suis pas en mesure d\w*) ?(?:vous )?(?:repondre|fournir (?:une|de|la) (?:\w+ )?reponse"
        r"|donner (?:une|de|la) (?:\w+ )?reponse|apporter (?:une|de) reponse)",
    ),
    ("fr_dont_know", r"\bje ne (?:sais|connais) pas\b|\bje n'ai pas (?:la|de) reponse\b"),
    (
        "en_cannot_answer",
        r"\b(?:i (?:cannot|can't|am unable to|am not able to|could not|couldn't|won't be able to)|unable to) "
        r"(?:answer|provide (?:an |a |the |any )?(?:\w+ )?answer|give (?:an |a )?answer|determine|say)\b",
    ),
    ("en_dont_know", r"\bi (?:do not|don't) know\b"),
)

# What may follow a statement of missing information without answering: help, rephrasing, related items
FILLER_PATTERNS = (
    (
        "help",
        r"\b(?:puis-je|(?:je peux|je pourrais) vous (?:aider|suggerer d|recommander d|proposer d|orienter|inviter)|can i help"
        r"|i can help|let me know|n'hesitez|feel free|si vous (?:avez|souhaitez|pouvez|voulez)|if you (?:have|want|can"
        r"|could|would)|please|veuillez|pourriez-vous|could you|would you like|pour obtenir|to get|to obtain)\b",
    ),
    (
        "rephrase",
        r"\b(?:reformul\w*|rephras\w*|preciser votre|clarify|essayez|try|consult\w*|verifi\w*|check"
        r"|je (?:vous )?(?:recommande|suggere|conseille)|nous (?:vous )?(?:recommandons|suggerons|conseillons)"
        r"|i (?:recommend|suggest|advise)|autre chose|anything else)\b",
    ),
    # what was found instead: "les passages renvoyés semblent aborder la musique traditionnelle"
    (
        "sources_described",
        r"\b(?:les |the )?(?:passages?|documents?|resultats?|results|informations?) (?:\w+ ){0,3}?(?:semblent|semble"
        r"|portent|parlent|traitent|abordent|concernent|seem|appear|relate|cover|discuss)\b",
    ),
    (
        "related",
        r"\b(?:similaires?|similar|related|apparentes?|autres? systemes?|another system|other systems?"
        r"|il est possible que (?:la question|votre question|l'information)|it is possible that (?:the question|your"
        r" question|the information))\b",
    ),
)

APOLOGY = re.compile(
    r"^(?:je suis |nous sommes )?(?:desole|navre|malheureusement|helas|sorry|i'm sorry|i am sorry|unfortunately)\W*$"
)
# a contrast that may introduce an answer after the statement of missing information
CONTRAST = re.compile(r"(?:[,;:]|\s-)\s*(?:mais|cependant|toutefois|neanmoins|pourtant|but|however|yet|though|although)\b")
# a statement of missing information introduced as a justification of an answer ("..., car il n'y a pas")
JUSTIFICATION = re.compile(r"\b(?:car|parce qu\w*|puisqu\w*|vu qu\w*|etant donne qu\w*|because|since|as)\s*$")
# a sentence ends at . ! ? followed by a capital or an opening quote ("Thaimis Inc. »" does not end one), or at a newline
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[\"«“(]?[A-ZÀ-Ý])|\n+")


def normalize(text: str) -> str:
    """Casefolded text without accents, with straight apostrophes and single spaces (what the patterns read)"""
    text = unicodedata.normalize("NFKD", text.replace("’", "'").replace("ʼ", "'").replace("`", "'"))
    text = "".join(c for c in text if not unicodedata.combining(c)).casefold()
    return re.sub(r"[ \t ]+", " ", text).strip()


def _compile(patterns) -> tuple:
    return tuple((name, re.compile(pattern)) for name, pattern in patterns)


@dataclass
class AbstentionResult:
    """Verdict of AbstentionDetector on one answer

    `kind` is "abstention", "hedged" (the agent said it found nothing, then answered anyway: an answer) or "answer";
    `pattern` names the rule that matched (None when none did), or "fallback" when the fallback decided; `score` is
    1.0 / 0.0 for a rule verdict, the fallback score otherwise.
    """

    abstained: bool
    kind: str
    pattern: Optional[str] = None
    score: float = 0.0


class AbstentionDetector:
    """Does an answer abstain (information not found, not in the sources, cannot answer)? French and English rules

    Parameters
    ----------
    extra_patterns : sequence of (name, regex), optional
        More statements of missing information, read on the normalized text (`normalize`: casefolded, no accents);
        like `NOT_FOUND_PATTERNS`, they count only in a sentence that names a source (`SOURCE_WORDS`)
    extra_cannot_answer : sequence of (name, regex), optional
        More statements that the agent cannot answer, valid without a source word
    fallback : callable, optional
        `fallback(answer) -> float` in [0, 1], called only when the rules find no statement of missing information
        (e.g. `NliAbstentionFallback`); the answer abstains when the score reaches `fallback_threshold`
    fallback_threshold : float
        Default 0.5
    """

    name = ABSTENTION

    def __init__(
        self,
        extra_patterns: Sequence[tuple] = (),
        extra_cannot_answer: Sequence[tuple] = (),
        fallback: Optional[Callable[[str], float]] = None,
        fallback_threshold: float = 0.5,
    ):
        self.not_found = _compile(NOT_FOUND_PATTERNS) + _compile(extra_patterns)
        self.cannot_answer = _compile(CANNOT_ANSWER_PATTERNS) + _compile(extra_cannot_answer)
        self.filler = _compile(FILLER_PATTERNS)
        self.fallback, self.fallback_threshold = fallback, fallback_threshold

    def _statement(self, sentence: str) -> Optional[tuple]:
        """(pattern name, end of the match) of the first statement of missing information in a sentence, or None"""
        found = []
        sourced = SOURCE_WORDS.search(sentence) is not None
        for name, pattern in self.cannot_answer + (self.not_found if sourced else ()):
            match = pattern.search(sentence)
            if match:
                found.append((match.start(), match.end(), name))
        if not found:
            return None
        start, end, name = min(found)
        prefix = sentence[:start]
        justification = JUSTIFICATION.search(prefix)
        if justification is not None and self._answers(prefix[: justification.start()]):
            return None  # "X est hors service, car il n'y a pas d'enregistrement": an answer with its reason
        contrast = CONTRAST.search(prefix)
        if contrast is not None and self._answers(prefix[: contrast.start()]):
            return None  # "X pourrait être refusée par Y, mais il n'y a pas de mention...": answered first
        return name, end

    def _answers(self, text: str) -> bool:
        """True when a clause or sentence gives content: neither missing information nor filler"""
        text = text.strip(" ,;:.-")
        if len(re.findall(r"\w+", text)) < 3 or APOLOGY.match(text):
            return False
        if self._statement(text) is not None:
            return False
        return not any(pattern.search(text) for _, pattern in self.filler)

    def detect(self, answer: Optional[str]) -> Optional[AbstentionResult]:
        """Verdict on one answer; None when there is no answer text"""
        if answer is None or not str(answer).strip():
            return None
        sentences = [normalize(s) for s in _SENTENCE.split(str(answer).strip()) if s.strip()]
        while len(sentences) > 1 and APOLOGY.match(sentences[0]):
            sentences = sentences[1:]
        first = self._statement(sentences[0])
        if first is None:
            if self.fallback is not None:
                score = float(self.fallback(str(answer)))
                abstained = score >= self.fallback_threshold
                return AbstentionResult(abstained, "abstention" if abstained else "answer", "fallback", score)
            return AbstentionResult(False, "answer")
        name, end = first
        rest = sentences[0][end:]
        contrast = CONTRAST.search(rest)
        hedged = (contrast is not None and self._answers(rest[contrast.end() :])) or any(self._answers(s) for s in sentences[1:])
        if hedged:
            return AbstentionResult(False, "hedged", name, 0.0)
        return AbstentionResult(True, "abstention", name, 1.0)

    def score(self, text: str) -> float:
        """1.0 when the answer abstains, else 0.0 (the fallback score when the fallback decided); 0.0 without text"""
        result = self.detect(text)
        return 0.0 if result is None else (result.score if result.pattern == "fallback" else float(result.abstained))

    def record(self, span, answer: Optional[str]) -> Optional[AbstentionResult]:
        """Write pulsar.detector.abstention (1.0 / 0.0) on a span; nothing without answer text"""
        result = self.detect(answer)
        if result is not None:
            record_score(span, self.name, 1.0 if result.abstained else 0.0)
        return result


class NliAbstentionFallback:
    """Fallback of AbstentionDetector: highest entailment, by the answer, of hypotheses stating that nothing was found

    Parameters
    ----------
    entailment : callable
        `entailment(premise, hypothesis) -> float`, e.g. `FaithfulnessScorer().entailment` (multilingual NLI)
    hypotheses : sequence of str, optional
        Default: one French and one English statement that the information is not available
    """

    HYPOTHESES = (
        "L'information demandée n'a pas été trouvée.",
        "The requested information is not available.",
    )

    def __init__(self, entailment: Callable[[str, str], float], hypotheses: Optional[Sequence[str]] = None):
        self.entailment = entailment
        self.hypotheses = tuple(hypotheses or self.HYPOTHESES)

    @classmethod
    def from_scorer(cls, scorer, hypotheses: Optional[Sequence[str]] = None) -> "NliAbstentionFallback":
        """From a FaithfulnessScorer (or any object with `entailment(premise, hypothesis)`)"""
        return cls(scorer.entailment, hypotheses)

    def __call__(self, answer: str) -> float:
        return max(float(self.entailment(answer, h)) for h in self.hypotheses)


@dataclass
class SupportThresholds:
    """Thresholds under which a run counts as without support (None: criterion not used)

    retrieval_top : best retrieval score of the run (e.g. the `retrieval_top` score); under it, nothing relevant was
        found. Its scale is the retriever's (BM25, cosine...): fit it on a reference of the deployment
    faithfulness : faithfulness of the answer to the passages (FaithfulnessScorer); under it, the answer is not
        supported by what was retrieved
    """

    retrieval_top: Optional[float] = None
    faithfulness: Optional[float] = None


def _lower_quantile(values, quantile: float) -> Optional[float]:
    finite = sorted(float(v) for v in values if v is not None and v == v and abs(float(v)) != float("inf"))
    if not finite:
        return None
    position = quantile * (len(finite) - 1)  # linear interpolation, as numpy's default
    low = int(position)
    high = min(low + 1, len(finite) - 1)
    return finite[low] + (finite[high] - finite[low]) * (position - low)


def calibrate_support_thresholds(
    retrieval_top: Sequence[Optional[float]] = (), faithfulness: Sequence[Optional[float]] = (), quantile: float = 0.05
) -> SupportThresholds:
    """Thresholds at the lower `quantile` (default 5 %) of a reference of the deployment

    Pass the best retrieval scores of the reference runs whose retrieval returned something, and the faithfulness of
    the reference runs that answered (abstentions excluded). About `quantile` of the reference runs then fall under
    each threshold: the rate of answers without support of the reference is the base rate a window is compared with
    (one-sided exact test, pulsar_metrics `rate_increase`). Missing values are ignored; an empty list gives None.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    return SupportThresholds(_lower_quantile(retrieval_top, quantile), _lower_quantile(faithfulness, quantile))


@dataclass
class AnswerSupport:
    """Abstention and support of one run

    `unsupported` is True when the agent answered (no abstention) although `reasons` is not empty: "empty_retrieval",
    "low_retrieval_score", "irrelevant_retrieval", "low_faithfulness". None when there is no answer to judge.
    """

    abstained: Optional[bool]
    unsupported: Optional[bool]
    reasons: tuple = ()
    abstention: Optional[AbstentionResult] = None


def _below(value, threshold) -> bool:
    return threshold is not None and value is not None and value == value and float(value) < threshold


def answer_support(
    abstained: Optional[bool],
    *,
    retrieval_empty: Optional[bool] = None,
    retrieval_top: Optional[float] = None,
    faithfulness: Optional[float] = None,
    thresholds: Optional[SupportThresholds] = None,
    retrieval_relevant: Optional[bool] = None,
) -> AnswerSupport:
    """Is an answer without support? `retrieval_empty`: no passage came back (or no search was made);
    `retrieval_relevant`: the application's own check that the passages are about the question, e.g. that the entity
    named in the question appears in them (False adds "irrelevant_retrieval"). A missing score (None or NaN) is not
    evidence of anything and is ignored"""
    thresholds = thresholds or SupportThresholds()
    reasons = []
    if retrieval_empty:
        reasons.append("empty_retrieval")
    elif _below(retrieval_top, thresholds.retrieval_top):
        reasons.append("low_retrieval_score")
    if not retrieval_empty and retrieval_relevant is False:
        reasons.append("irrelevant_retrieval")
    if not retrieval_empty and _below(faithfulness, thresholds.faithfulness):
        reasons.append("low_faithfulness")
    unsupported = None if abstained is None else (not abstained and bool(reasons))
    return AnswerSupport(abstained, unsupported, tuple(reasons))


def record_answer_support(
    span,
    answer: Optional[str],
    *,
    retrieval_empty: Optional[bool] = None,
    retrieval_top: Optional[float] = None,
    faithfulness: Optional[float] = None,
    thresholds: Optional[SupportThresholds] = None,
    detector: Optional[AbstentionDetector] = None,
    retrieval_relevant: Optional[bool] = None,
) -> AnswerSupport:
    """Judge one run and write pulsar.detector.abstention and pulsar.detector.unsupported_answer (1.0 / 0.0) on a span

    Call it on the agent span when the answer is known (e.g. in the `on_finish` callback of the demo agent): the
    `agent_run` record gains `det_abstention` and `det_unsupported_answer`. Without answer text nothing is written.
    """
    detector = detector or AbstentionDetector()
    verdict = detector.detect(answer)
    support = answer_support(
        None if verdict is None else verdict.abstained,
        retrieval_empty=retrieval_empty,
        retrieval_top=retrieval_top,
        faithfulness=faithfulness,
        thresholds=thresholds,
        retrieval_relevant=retrieval_relevant,
    )
    support.abstention = verdict
    if verdict is not None:
        record_score(span, ABSTENTION, 1.0 if support.abstained else 0.0)
        record_score(span, UNSUPPORTED_ANSWER, 1.0 if support.unsupported else 0.0)
    return support
