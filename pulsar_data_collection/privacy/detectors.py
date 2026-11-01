"""Rule-based detection of personal information in text, for Canada (French and English).

Every recognizer is a pattern plus a validation (checksum, valid date, issuer prefix) and a confidence that the
context raises (a keyword such as "NAS" or "téléphone" just before) or lowers (a keyword such as "dossier" or
"commande", which announces a business number). Findings carry the type, the position and the confidence, never the
value itself.
"""

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

CONTEXT_CHARS = 40

# Words announcing a business identifier rather than personal information
BUSINESS_WORDS = (
    "dossier",
    "référence",
    "reference",
    "réf",
    "ref",
    "commande",
    "order",
    "facture",
    "invoice",
    "billet",
    "ticket",
    "produit",
    "product",
    "sku",
    "lot",
    "transaction",
    "contrat",
    "contract",
    "file",
    "case",
)


# A currency right after a number makes it an amount
AMOUNT_AFTER = re.compile(r"\s?(?:\$|\$ ?CA|CAD|dollars?|€|EUR|USD)(?!\w)", re.IGNORECASE)


@dataclass(frozen=True)
class Finding:
    """Personal information found in a text: type, position [start, end) and confidence in [0, 1]"""

    type: str
    start: int
    end: int
    confidence: float


@dataclass(frozen=True)
class Recognizer:
    type: str
    label: str  # replacement text when masking, e.g. "[SIN]"
    pattern: re.Pattern
    confidence: float  # without context
    context_words: tuple = ()  # raise the confidence to `context_confidence`
    context_confidence: float = 0.99
    validate: Optional[Callable[[re.Match], bool]] = None
    requires_context: bool = False  # only reported when a context word is present (e.g. dates of birth)
    business_sensitive: bool = True  # lowered when a business word precedes


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def luhn(number: str) -> bool:
    digits = [int(d) for d in number]
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _sin(match: re.Match) -> bool:
    number = _digits(match.group(0))
    return number[0] not in "08" and luhn(number)  # 0 and 8 are not issued


def _card(match: re.Match) -> bool:
    number = _digits(match.group(0))
    if not 13 <= len(number) <= 19 or not luhn(number):
        return False
    return bool(re.match(r"4|5[1-5]|2(2[2-9]|[3-6]\d|7[01]|720)|3[47]|6011|65|35", number))


def _ramq(match: re.Match) -> bool:
    month, day = int(match.group(3)), int(match.group(4))
    return (1 <= month <= 12 or 51 <= month <= 62) and 1 <= day <= 31  # women: month + 50


def _date(match: re.Match) -> bool:
    numbers = [int(n) for n in re.findall(r"\d+", match.group(0))]
    return any(1900 <= n <= 2100 for n in numbers)


_MONTHS = (
    r"janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre|"
    r"january|february|march|april|may|june|july|august|september|october|november|december"
)

RECOGNIZERS = [
    Recognizer(
        "email",
        "[EMAIL]",
        re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"),
        0.95,
        business_sensitive=False,
    ),
    Recognizer(
        "credit_card",
        "[CREDIT_CARD]",
        re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])"),
        0.95,
        context_words=("carte", "card", "visa", "mastercard", "crédit", "credit"),
        validate=_card,
        business_sensitive=False,
    ),
    Recognizer(
        "sin",
        "[SIN]",
        re.compile(r"(?<![\d-])\d{3}[ -]?\d{3}[ -]?\d{3}(?![\d-])"),
        0.6,
        context_words=("nas", "sin", "assurance sociale", "social insurance", "numéro d'assurance"),
        validate=_sin,
    ),
    Recognizer(
        "ramq",
        "[RAMQ]",
        re.compile(r"\b([A-Z]{4}) ?(\d{2})(\d{2}) ?(\d{2})(\d{2})\b"),
        0.9,
        context_words=("ramq", "assurance maladie", "carte soleil", "health card", "health insurance"),
        validate=_ramq,
    ),
    Recognizer(
        "phone",
        "[PHONE]",
        re.compile(r"(?<![\d-])(?:\+?1[ .-]?)?(?:\(\s*[2-9]\d{2}\s*\)|[2-9]\d{2})[ .-]?[2-9]\d{2}[ .-]?\d{4}(?![\d-])"),
        0.6,
        context_words=(
            "tél",
            "tel",
            "téléphone",
            "telephone",
            "phone",
            "cell",
            "cellulaire",
            "mobile",
            "appeler",
            "appelez",
            "appelle",
            "joindre",
            "rejoindre",
            "contacter",
            "contactez",
            "call",
            "reach",
            "text",
            "numéro",
            "number",
        ),
        context_confidence=0.9,
    ),
    Recognizer(
        "postal_code",
        "[POSTAL_CODE]",
        re.compile(r"\b[ABCEGHJ-NPRSTVXY]\d[ABCEGHJ-NPRSTV-Z][ -]?\d[ABCEGHJ-NPRSTV-Z]\d\b", re.IGNORECASE),
        0.85,
        context_words=("code postal", "postal code", "adresse", "address"),
    ),
    Recognizer(
        "drivers_licence_qc",
        "[DRIVERS_LICENCE]",
        re.compile(r"\b[A-Z]\d{4}[ -]?\d{6}[ -]?\d{2}\b"),
        0.9,
        context_words=("permis", "licence", "license", "saaq"),
    ),
    Recognizer(
        "passport",
        "[PASSPORT]",
        re.compile(r"\b[A-Z]{2}\d{6}\b"),
        0.4,
        context_words=("passeport", "passport"),
        context_confidence=0.95,
    ),
    Recognizer(
        "date_of_birth",
        "[DATE_OF_BIRTH]",
        re.compile(
            rf"\b(?:\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}[/.-]\d{{1,2}}[/.-]\d{{4}}|\d{{1,2}}(?:er)? (?:{_MONTHS}) \d{{4}}|"
            rf"(?:{_MONTHS}) \d{{1,2}},? \d{{4}})\b",
            re.IGNORECASE,
        ),
        0.0,
        context_words=("né le", "née le", "né(e) le", "date de naissance", "naissance", "dob", "date of birth", "born", "birth"),
        context_confidence=0.9,
        validate=_date,
        requires_context=True,
        business_sensitive=False,
    ),
]
TYPES = tuple(r.type for r in RECOGNIZERS)
LABELS = {r.type: r.label for r in RECOGNIZERS}


def _has(words: Iterable[str], before: str) -> bool:
    return any(re.search(rf"(?<!\w){re.escape(w)}(?!\w)", before) for w in words)


class PiiDetector:
    """Detect personal information in text with validated rules

    Parameters
    ----------
    types : sequence of str, optional
        Types to look for (default: all, see `TYPES`)
    min_confidence : float, optional
        Findings below are dropped (default 0.5)
    """

    def __init__(self, types: Optional[Iterable[str]] = None, min_confidence: float = 0.5):
        wanted = set(types) if types is not None else set(TYPES)
        unknown = wanted - set(TYPES)
        if unknown:
            raise ValueError(f"unknown personal information types {sorted(unknown)}; known: {TYPES}")
        self.recognizers = [r for r in RECOGNIZERS if r.type in wanted]
        self.min_confidence = min_confidence

    def scan(self, text: str) -> list:
        """Findings in `text`, without overlaps (the most confident, then the longest, wins), in reading order"""
        if not text:
            return []
        candidates = []
        lowered = text.lower()
        for r in self.recognizers:
            for match in r.pattern.finditer(text):
                if r.validate is not None and not r.validate(match):
                    continue
                before = lowered[max(0, match.start() - CONTEXT_CHARS) : match.start()]
                if r.context_words and _has(r.context_words, before):
                    confidence = max(r.confidence, r.context_confidence)
                elif r.requires_context:
                    continue
                else:
                    confidence = r.confidence
                    if r.business_sensitive and _has(BUSINESS_WORDS, before):
                        confidence = min(confidence, 0.3)
                if r.type in ("phone", "sin", "credit_card") and AMOUNT_AFTER.match(text, match.end()):
                    confidence = min(confidence, 0.3)  # "514 555 0199 $" is an amount
                if confidence >= self.min_confidence:
                    candidates.append(Finding(r.type, match.start(), match.end(), confidence))
        kept = []
        for f in sorted(candidates, key=lambda f: (-f.confidence, -(f.end - f.start), f.start)):
            if all(f.end <= k.start or f.start >= k.end for k in kept):
                kept.append(f)
        return sorted(kept, key=lambda f: f.start)

    def mask(self, text: str, findings: Optional[list] = None) -> str:
        """`text` with every finding replaced by its label, e.g. "[SIN]" """
        findings = self.scan(text) if findings is None else findings
        out, last = [], 0
        for f in sorted(findings, key=lambda f: f.start):
            out.append(text[last : f.start])
            out.append(LABELS[f.type])
            last = f.end
        out.append(text[last:])
        return "".join(out)


def normalize(finding: Finding, text: str) -> str:
    """Comparable form of a finding's value (digits only, or lower case for e-mails and postal codes), used in memory"""
    value = text[finding.start : finding.end]
    if finding.type in ("email", "postal_code", "ramq", "drivers_licence_qc", "passport", "date_of_birth"):
        return f"{finding.type}:{re.sub(r'[ -]', '', value).lower()}"
    digits = _digits(value)
    if finding.type == "phone" and len(digits) == 11:
        digits = digits[1:]
    return f"{finding.type}:{digits}"
