"""Rule-based detection of personal information in text, for Canada (French and English) and, on request, other
countries and sensitive financial identifiers.

Every recognizer is a pattern plus a validation (checksum, valid date, issuer prefix) and a confidence that the
context raises (a keyword such as "NAS" or "téléphone" just before) or lowers (a keyword such as "dossier" or
"commande", which announces a business number, or a word proper to a type, such as "client" or "confirmation" before
a number shaped like a SIN). Findings carry the type, the position and the confidence, never the value itself.

The Canadian rules are on by default. Other countries are opt-in (`locales`), and so are the groups of rules that
are not identifiers of a person in the strict sense or that are more prone to false positives (`profiles`):
"finance" (IBAN, SWIFT/BIC, bank account, transit and routing numbers, card security code) and "address" (street
addresses: civic number and street type, in French and English).
"""

import re
from dataclasses import dataclass
from functools import lru_cache
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

# Where the business words (and the words proper to a type) are looked for: anywhere in the CONTEXT_CHARS before the
# value ("window", default), or only immediately before it, possibly followed by a label such as "no", "n°", "#",
# "numéro" or ":" ("adjacent": "Votre dossier 130 692 544", "Customer no. 130 692 544", but not "Le dossier voisin
# indique aussi 130 692 544")
BUSINESS_SCOPES = ("window", "adjacent")
_FILLER = r"(?:n[o°º](?!\w)\.?|#|num[ée]ro(?!\w)|number(?!\w)|num(?!\w)\.?|id(?!\w)|[:=-])"

LOCALES = ("ca", "us", "fr", "uk", "intl")
PROFILES = ("finance", "address")

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
    negative_words: tuple = ()  # proper to this type, right before the value: something else ("client" for a SIN)
    weak_context_words: tuple = ()  # context words that an adjacent negative word overrides ("numéro de membre")
    negative_prefix: Optional[re.Pattern] = None  # a format cue just before the value (e.g. "#") lowers it
    adjacent_context: bool = False  # the context word must be immediately before (short values, e.g. a CVV)
    base_confidence: Optional[Callable[[re.Match], float]] = None  # confidence without context, by format
    locale: str = "ca"  # one of LOCALES, or "any"
    profile: str = "core"  # "core" (always on) or one of PROFILES


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


def _ramq_confidence(match: re.Match) -> float:
    # Printed in capitals on the card; in lower case, any 4-letter word followed by 8 digits could match
    return 0.9 if match.group(1).isupper() else 0.55


def _date(match: re.Match) -> bool:
    numbers = [int(n) for n in re.findall(r"\d+", match.group(0))]
    return any(1900 <= n <= 2100 for n in numbers)


def _e164(match: re.Match) -> bool:
    return 8 <= len(_digits(match.group(0))) <= 15


def _nir(match: re.Match) -> bool:
    """French social security number: 13 characters and a key, 97 - (number mod 97); Corsica 2A = 19, 2B = 18"""
    value = re.sub(r"[ .]", "", match.group(0)).upper()
    body, key = value[:13], value[13:]
    department = {"2A": "19", "2B": "18"}.get(body[5:7], body[5:7])
    body = body[:5] + department + body[7:]
    return body.isdigit() and int(key) == 97 - int(body) % 97


def _iban(match: re.Match) -> bool:
    value = re.sub(r"\s", "", match.group(0)).upper()
    if not 15 <= len(value) <= 34:
        return False
    rearranged = value[4:] + value[:4]
    return int("".join(str(int(c, 36)) for c in rearranged)) % 97 == 1


# ISO 3166-1 alpha-2, for the country of a SWIFT/BIC code
_COUNTRIES = set(
    "AD AE AF AG AI AL AM AO AQ AR AS AT AU AW AX AZ BA BB BD BE BF BG BH BI BJ BL BM BN BO BQ BR BS BT BV BW BY BZ CA "
    "CC CD CF CG CH CI CK CL CM CN CO CR CU CV CW CX CY CZ DE DJ DK DM DO DZ EC EE EG EH ER ES ET FI FJ FK FM FO FR GA "
    "GB GD GE GF GG GH GI GL GM GN GP GQ GR GS GT GU GW GY HK HM HN HR HT HU ID IE IL IM IN IO IQ IR IS IT JE JM JO JP "
    "KE KG KH KI KM KN KP KR KW KY KZ LA LB LC LI LK LR LS LT LU LV LY MA MC MD ME MF MG MH MK ML MM MN MO MP MQ MR MS "
    "MT MU MV MW MX MY MZ NA NC NE NF NG NI NL NO NP NR NU NZ OM PA PE PF PG PH PK PL PM PN PR PS PT PW PY QA RE RO RS "
    "RU RW SA SB SC SD SE SG SH SI SJ SK SL SM SN SO SR SS ST SV SX SY SZ TC TD TF TG TH TJ TK TL TM TN TO TR TT TV TW "
    "TZ UA UG UM US UY UZ VA VC VE VG VI VN VU WF WS XK YE YT ZA ZM ZW".split()
)


def _bic(match: re.Match) -> bool:
    return match.group(0)[4:6] in _COUNTRIES


def _aba(match: re.Match) -> bool:
    d = [int(c) for c in _digits(match.group(0))]
    return (3 * (d[0] + d[3] + d[6]) + 7 * (d[1] + d[4] + d[7]) + (d[2] + d[5] + d[8])) % 10 == 0


def _not_a_date(match: re.Match) -> bool:
    value = match.group(0)
    return not re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}-\d{1,2}-\d{4}|\d{4}", value)


_MONTHS = (
    r"janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre|"
    r"january|february|march|april|may|june|july|august|september|october|november|december"
)

_PHONE_WORDS = (
    "tél",
    "tel",
    "téléphone",
    "telephone",
    "phone",
    "cell",
    "cellulaire",
    "mobile",
    "portable",
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
)
# Words proper to a type: they announce another identifier with the same shape
_SIN_NEGATIVE = (
    "client",
    "customer",
    "membre",
    "member",
    "confirmation",
    "abonné",
    "abonne",
    "subscriber",
    "réservation",
    "reservation",
    "booking",
    "suivi",
    "tracking",
    "routing",
    "transit",
    "aba",
)
_PHONE_NEGATIVE = (
    "membre",
    "member",
    "matricule",
    "confirmation",
    "réservation",
    "reservation",
    "booking",
    "suivi",
    "tracking",
)

# Street addresses: civic number, then the street type before the name (French) or after it (English)
_NUM = r"(?<![\w.,-])\d{1,6}(?:-\d{1,6})?[A-Za-z]?(?:\s?(?:bis|ter))?"
_NAMEWORD = r"(?:[A-ZÀ-ÖØ-Þ][\w'’-]*(?:\.[\w'’-]+)*|\d{1,3}(?:e|er|re|ème|eme|st|nd|rd|th)(?!\w))"
_CONNECTOR = r"(?:de|du|des|la|le|les|aux|au|sur|en|et|of|the)"
_FR_STREET = (
    r"ruelle|rue|avenue|av\.|boulevard|boul\.?|bd|blvd|chemin|ch\.|route|rte|rang|place|pl\.|allée|allee|impasse|quai|"
    r"cours|montée|montee|côte|cote|croissant|promenade|sentier|square|passage|voie|esplanade|terrasse|carré|autoroute"
)
_EN_STREET = (
    r"street|st\.?|avenue|ave\.?|road|rd\.?|boulevard|blvd\.?|drive|dr\.?|lane|ln\.?|court|ct\.?|place|pl\.?|way|"
    r"crescent|cres\.?|terrace|parkway|pkwy\.?|highway|hwy\.?|circle|cir\.?|trail|square|sq\.?|close|row|grove|gardens"
)
_DIRECTION = r"(?:\s+(?:N|S|E|W|NE|NW|SE|SW|North|South|East|West|Nord|Sud|Est|Ouest)(?!\w)\.?)?"
_UNIT = r"(?:,?\s+(?:(?i:apt|app|appartement|apartment|unit|unité|suite|bureau|ste|local)\.?|#)\s*\d+[A-Za-z]?)?"
_ADDRESS_WORDS = (
    "adresse",
    "address",
    "habite",
    "j'habite",
    "live at",
    "lives at",
    "living at",
    "domicile",
    "demeure",
    "résidence",
    "residence",
    "located at",
    "ship to",
    "livrer",
    "livraison",
    "delivery",
    "mailing",
)

RECOGNIZERS = [
    Recognizer(
        "email",
        "[EMAIL]",
        # the lookbehind starts a match only at the start of a run: linear on long runs without "@"
        re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"),
        0.95,
        business_sensitive=False,
        locale="any",
    ),
    Recognizer(
        "credit_card",
        "[CREDIT_CARD]",
        re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])"),
        0.95,
        context_words=("carte", "card", "visa", "mastercard", "crédit", "credit"),
        validate=_card,
        business_sensitive=False,
        locale="any",
    ),
    Recognizer(
        "sin",
        "[SIN]",
        re.compile(r"(?<![\d-])\d{3}[ -]?\d{3}[ -]?\d{3}(?![\d-])"),
        0.6,
        context_words=("nas", "sin", "assurance sociale", "social insurance", "numéro d'assurance"),
        validate=_sin,
        negative_words=_SIN_NEGATIVE,
        negative_prefix=re.compile(r"#\s*$"),  # "#130692544" is a code, a SIN is not written so
    ),
    Recognizer(
        "ramq",
        "[RAMQ]",
        re.compile(r"\b([A-Za-z]{4})[ -]?(\d{2})(\d{2})[ -]?(\d{2})(\d{2})\b"),
        0.9,
        context_words=("ramq", "assurance maladie", "carte soleil", "health card", "health insurance"),
        validate=_ramq,
        base_confidence=_ramq_confidence,
    ),
    Recognizer(
        "phone",
        "[PHONE]",
        re.compile(r"(?<![\d-])(?:\+?1[ .-]?)?(?:\(\s*[2-9]\d{2}\s*\)|[2-9]\d{2})[ .-]?[2-9]\d{2}[ .-]?\d{4}(?![\d-])"),
        0.6,
        context_words=_PHONE_WORDS,
        context_confidence=0.9,
        negative_words=_PHONE_NEGATIVE,
        weak_context_words=("numéro", "number"),
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
        locale="any",
    ),
    # ---- United States
    Recognizer(
        "ssn_us",
        "[SSN]",
        re.compile(r"(?<![\d-])(?!000|666|9\d\d)\d{3}([- ])(?!00)\d{2}\1(?!0000)\d{4}(?![\d-])"),
        0.85,
        context_words=("ssn", "social security"),
        locale="us",
    ),
    Recognizer(
        "ssn_us",
        "[SSN]",
        re.compile(r"(?<![\d-])(?!000|666|9\d\d)\d{3}(?!00)\d{2}(?!0000)\d{4}(?![\d-])"),
        0.0,
        context_words=("ssn", "social security"),
        requires_context=True,
        locale="us",
    ),
    # ---- France
    Recognizer(
        "phone",
        "[PHONE]",
        re.compile(r"(?<![\w+])(?:\+33[ .-]?(?:\(0\)[ .-]?)?|0)[1-9](?:[ .-]?\d{2}){4}(?!\d)"),
        0.7,
        context_words=_PHONE_WORDS + ("fixe",),
        context_confidence=0.9,
        negative_words=_PHONE_NEGATIVE,
        weak_context_words=("numéro", "number"),
        locale="fr",
    ),
    Recognizer(
        "nir_fr",
        "[NIR]",
        re.compile(
            r"(?<!\w)[12][ .]?\d{2}[ .]?(?:0[1-9]|1[0-2]|[2-9]\d)[ .]?(?:\d{2}|2[AB])[ .]?\d{3}[ .]?\d{3}[ .]?\d{2}(?!\w)",
            re.IGNORECASE,
        ),
        0.9,
        context_words=("sécurité sociale", "securite sociale", "nir", "sécu", "secu", "carte vitale", "insee"),
        validate=_nir,
        locale="fr",
    ),
    # ---- United Kingdom
    Recognizer(
        "phone",
        "[PHONE]",
        re.compile(
            r"(?<![\w+])(?:\+44[ ]?(?:\(0\)[ ]?)?|0)(?:7\d{3}[ ]?\d{6}|20[ ]?\d{4}[ ]?\d{4}|1\d{2,3}[ ]?\d{3}[ ]?\d{3,4})(?!\d)"
        ),
        0.65,
        context_words=_PHONE_WORDS,
        context_confidence=0.9,
        negative_words=_PHONE_NEGATIVE,
        weak_context_words=("numéro", "number"),
        locale="uk",
    ),
    Recognizer(
        "nino_uk",
        "[NINO]",
        re.compile(r"(?<!\w)(?!BG|GB|KN|NK|NT|TN|ZZ)[A-CEGHJ-PR-TW-Z][A-CEGHJ-NPR-TW-Z] ?\d{2} ?\d{2} ?\d{2} ?[A-D](?!\w)"),
        0.8,
        context_words=("national insurance", "nino", "ni number"),
        locale="uk",
    ),
    # ---- International formats
    Recognizer(
        "phone",
        "[PHONE]",
        re.compile(r"(?<![\w+])\+[1-9]\d{0,2}(?:[ .-]?\(?\d{1,4}\)?){2,6}(?!\d)"),
        0.85,
        context_words=_PHONE_WORDS,
        context_confidence=0.95,
        validate=_e164,
        locale="intl",
    ),
    Recognizer(
        "passport",
        "[PASSPORT]",
        re.compile(r"(?<!\w)(?=[A-Z0-9]*\d)[A-Z0-9]{6,9}(?!\w)"),
        0.0,
        context_words=("passeport", "passport"),
        context_confidence=0.85,
        validate=_not_a_date,
        requires_context=True,
        locale="intl",
    ),
    Recognizer(
        "drivers_licence",
        "[DRIVERS_LICENCE]",
        re.compile(r"(?<![\w-])(?=[A-Z0-9-]*\d)(?=(?:[A-Z0-9]-?){5})[A-Z0-9][A-Z0-9-]{3,18}[A-Z0-9](?![\w-])"),
        0.0,
        context_words=(
            "driver's license",
            "driver's licence",
            "driver’s license",
            "driver’s licence",
            "drivers license",
            "driver license",
            "driving licence",
            "driving license",
            "permis de conduire",
            "numéro de permis",
            "license number",
            "licence number",
            "dl",
            "dln",
        ),
        context_confidence=0.85,
        validate=_not_a_date,
        requires_context=True,
        locale="intl",
    ),
    # ---- Profile "finance"
    Recognizer(
        "iban",
        "[IBAN]",
        re.compile(r"(?<!\w)[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,4})?(?!\w)"),
        0.95,
        context_words=("iban",),
        validate=_iban,
        business_sensitive=False,
        locale="any",
        profile="finance",
    ),
    Recognizer(
        "bic",
        "[BIC]",
        re.compile(r"(?<!\w)[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?(?!\w)"),
        0.0,
        context_words=("swift", "bic"),
        context_confidence=0.9,
        validate=_bic,
        requires_context=True,
        locale="any",
        profile="finance",
    ),
    Recognizer(
        "bank_account",
        "[BANK_ACCOUNT]",
        re.compile(r"(?<![\d-])\d(?:[ -]?\d){4,16}(?![\d-])"),
        0.0,
        context_words=(
            "compte bancaire",
            "numéro de compte",
            "no de compte",
            "n° de compte",
            "no. de compte",
            "compte no",
            "compte n°",
            "compte #",
            "compte chèque",
            "compte chèques",
            "compte épargne",
            "account number",
            "account no",
            "account #",
            "acct",
            "bank account",
            "checking account",
            "savings account",
            "folio",
            "bban",
        ),
        context_confidence=0.9,
        requires_context=True,
        business_sensitive=False,
        locale="any",
        profile="finance",
    ),
    Recognizer(
        "bank_account",
        "[BANK_ACCOUNT]",
        re.compile(r"(?<![\d-])\d{5}-\d{3}-\d{7,12}(?![\d-])"),  # cheque order: transit-institution-account
        0.8,
        context_confidence=0.9,
        locale="ca",
        profile="finance",
    ),
    Recognizer(
        "bank_routing",
        "[BANK_ROUTING]",
        re.compile(r"(?<![\d-])(?:\d{5}[- ]\d{3}|0\d{8})(?![\d-])"),  # transit-institution, or the EFT form 0IIITTTTT
        0.0,
        context_words=("transit", "institution", "succursale", "routing", "acheminement", "branch", "domiciliation"),
        context_confidence=0.9,
        requires_context=True,
        locale="ca",
        profile="finance",
    ),
    Recognizer(
        "bank_routing",
        "[BANK_ROUTING]",
        re.compile(r"(?<![\d-])(?:\d{5}|\d{3})(?![\d-])"),  # transit (5) or institution (3) alone, right after its name
        0.0,
        context_words=("transit", "succursale", "branch", "institution", "numéro d'institution", "institution number"),
        context_confidence=0.85,
        requires_context=True,
        adjacent_context=True,
        locale="ca",
        profile="finance",
    ),
    Recognizer(
        "bank_routing",
        "[BANK_ROUTING]",
        re.compile(r"(?<![\d-])\d{9}(?![\d-])"),
        0.0,
        context_words=("routing", "aba", "rtn"),
        context_confidence=0.9,
        validate=_aba,
        requires_context=True,
        locale="us",
        profile="finance",
    ),
    Recognizer(
        "card_security_code",
        "[CVV]",
        re.compile(r"(?<![\d-])\d{3,4}(?![\d-])"),
        0.0,
        context_words=(
            "cvv",
            "cvv2",
            "cvc",
            "cvc2",
            "cid",
            "csc",
            "security code",
            "code de sécurité",
            "code de securite",
            "cryptogramme",
            "card verification",
        ),
        context_confidence=0.9,
        requires_context=True,
        adjacent_context=True,
        business_sensitive=False,
        locale="any",
        profile="finance",
    ),
    # ---- Profile "address"
    Recognizer(
        "street_address",
        "[ADDRESS]",
        re.compile(
            rf"{_NUM},?\s+(?i:{_FR_STREET})(?!\w)(?:\s+(?:{_CONNECTOR}\s+)*(?:[ldLD]['’])?{_NAMEWORD}){{1,5}}{_DIRECTION}{_UNIT}"
        ),
        0.8,
        context_words=_ADDRESS_WORDS,
        context_confidence=0.95,
        business_sensitive=False,
        locale="any",
        profile="address",
    ),
    Recognizer(
        "street_address",
        "[ADDRESS]",
        re.compile(rf"{_NUM}(?:\s+{_NAMEWORD}){{1,4}}\s+(?i:{_EN_STREET})(?!\w){_DIRECTION}{_UNIT}"),
        0.8,
        context_words=_ADDRESS_WORDS,
        context_confidence=0.95,
        business_sensitive=False,
        locale="any",
        profile="address",
    ),
]


def _unique(values):
    return tuple(dict.fromkeys(values))


# Types on by default (Canada, core rules), and every type a detector can be asked for
TYPES = _unique(r.type for r in RECOGNIZERS if r.locale in ("ca", "any") and r.profile == "core")
ALL_TYPES = _unique(r.type for r in RECOGNIZERS)
LABELS = {r.type: r.label for r in RECOGNIZERS}


@lru_cache(maxsize=64)
def _words_pattern(words: tuple) -> re.Pattern:
    alternatives = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)")


def _has(words: Iterable[str], before: str) -> bool:
    words = tuple(words)
    return bool(words) and _words_pattern(words).search(before) is not None


@lru_cache(maxsize=64)
def _adjacent_pattern(words: tuple) -> re.Pattern:
    alternatives = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)(?:\s*{_FILLER})*\s*$")


def _adjacent(words: tuple, before: str) -> bool:
    """One of `words` ends `before`, possibly followed by a label ("no", "n°", "#", "numéro", ":")"""
    return bool(words) and _adjacent_pattern(tuple(words)).search(before) is not None


class PiiDetector:
    """Detect personal information in text with validated rules

    Parameters
    ----------
    types : sequence of str, optional
        Types to look for (default: every type enabled by `locales` and `profiles`; see `TYPES` and `ALL_TYPES`). A
        type named here is always enabled, whatever the locales and profiles.
    min_confidence : float, optional
        Findings below are dropped (default 0.5)
    locales : sequence of str, optional
        Country formats to recognize, among `LOCALES` (default ("ca",)): "us" (SSN), "fr" (phone, NIR), "uk"
        (phone, National Insurance number), "intl" (E.164 phone numbers, passports and driver's licences after a
        context word). E-mails, cards and dates of birth are recognized whatever the locale.
    profiles : sequence of str, optional
        Extra groups of rules, among `PROFILES` (default none): "finance" (IBAN, SWIFT/BIC, bank account, transit,
        institution and routing numbers, card security code) and "address" (street addresses).
    """

    def __init__(
        self,
        types: Optional[Iterable[str]] = None,
        min_confidence: float = 0.5,
        locales: Iterable[str] = ("ca",),
        profiles: Iterable[str] = (),
    ):
        self.locales = tuple(locales)
        self.profiles = tuple(profiles)
        for name, given, known in (("locales", self.locales, LOCALES), ("profiles", self.profiles, PROFILES)):
            unknown = set(given) - set(known)
            if unknown:
                raise ValueError(f"unknown {name} {sorted(unknown)}; known: {known}")
        enabled = [
            r
            for r in RECOGNIZERS
            if (r.locale == "any" or r.locale in self.locales) and (r.profile == "core" or r.profile in self.profiles)
        ]
        if types is not None:
            wanted = set(types)
            unknown = wanted - set(ALL_TYPES)
            if unknown:
                raise ValueError(f"unknown personal information types {sorted(unknown)}; known: {ALL_TYPES}")
            kept = {id(r) for r in enabled if r.type in wanted}
            missing = wanted - {r.type for r in enabled}
            kept |= {id(r) for r in RECOGNIZERS if r.type in missing}  # named explicitly: on whatever the locale
            enabled = [r for r in RECOGNIZERS if id(r) in kept]
        self.recognizers = enabled
        self.min_confidence = min_confidence

    def scan(self, text: str, business_scope: str = "window") -> list:
        """Findings in `text`, without overlaps (the most confident, then the longest, wins), in reading order

        `business_scope`: where the business words and the words proper to a type lower the confidence: anywhere in
        the 40 characters before the value ("window", default) or only immediately before it ("adjacent", used for
        the outputs of agents so that a leak is not hidden by an earlier mention of a file).
        """
        if business_scope not in BUSINESS_SCOPES:
            raise ValueError(f"business_scope must be one of {BUSINESS_SCOPES}")
        if not text:
            return []
        candidates = []
        lowered = text.lower()
        for r in self.recognizers:
            for match in r.pattern.finditer(text):
                if r.validate is not None and not r.validate(match):
                    continue
                before = lowered[max(0, match.start() - CONTEXT_CHARS) : match.start()]
                base = r.base_confidence(match) if r.base_confidence is not None else r.confidence
                if r.adjacent_context:
                    context = _adjacent(r.context_words, before)
                else:
                    context = bool(r.context_words) and _has(r.context_words, before)
                overridden = False
                if context and r.weak_context_words and _adjacent(r.negative_words, before):
                    # "Numéro de membre 514-555-0123": the word proper to the type, right before, beats a generic one
                    strong = tuple(w for w in r.context_words if w not in r.weak_context_words)
                    context = _has(strong, before)
                    overridden = not context
                if overridden:
                    confidence = min(base, 0.3)
                elif context:
                    confidence = max(base, r.context_confidence)
                elif r.requires_context:
                    continue
                else:
                    confidence = base
                    business = BUSINESS_WORDS if r.business_sensitive else ()
                    if business_scope == "adjacent":
                        lowered_by_word = _adjacent(business, before)
                    else:
                        lowered_by_word = bool(business) and _has(business, before)
                    # Words proper to a type count only right before the value, whatever the scope: "No client
                    # 130 692 544" is a customer number, "Customer Jean Tremblay 130 692 544" may be a SIN
                    lowered_by_word = lowered_by_word or _adjacent(r.negative_words, before)
                    if lowered_by_word or (r.negative_prefix is not None and r.negative_prefix.search(before)):
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
            out.append(LABELS.get(f.type, f"[{f.type.upper()}]"))
            last = f.end
        out.append(text[last:])
        return "".join(out)


_TEXT_TYPES = ("email", "postal_code", "ramq", "drivers_licence_qc", "passport", "date_of_birth")
_CODE_TYPES = ("drivers_licence", "nir_fr", "nino_uk", "iban", "bic")
_TRUNK = {"33": 9, "44": 10}  # country code -> national digits after the leading 0 (France, United Kingdom)


def normalize(finding: Finding, text: str) -> str:
    """Comparable form of a finding's value (digits only, or lower case for e-mails and postal codes), used in memory"""
    value = text[finding.start : finding.end]
    if finding.type in _TEXT_TYPES:
        return f"{finding.type}:{re.sub(r'[ -]', '', value).lower()}"
    if finding.type in _CODE_TYPES:
        return finding.type + ":" + re.sub(r"[\s.-]", "", value).lower()
    if finding.type == "street_address":
        return finding.type + ":" + re.sub(r"\W", "", value).lower()
    digits = _digits(value)
    if finding.type == "phone":
        if len(digits) == 11 and digits[0] == "1":
            digits = digits[1:]
        for code, national in _TRUNK.items():  # "+33 1 23 45 67 89" and "01 23 45 67 89" are the same number
            if digits.startswith(code) and len(digits) in (len(code) + national, len(code) + national + 1):
                digits = "0" + digits[-national:]
    return f"{finding.type}:{digits}"
