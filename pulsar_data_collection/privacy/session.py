"""Personal information in the conversations of LLM applications and agents, and its leaks.

A leak is personal information in an output (an answer, a tool call) that the user did not provide in the same
conversation, e.g. the social insurance number of someone else read in a database and repeated to the user. The
values given by the user are kept in memory, for the session only, to tell them apart; only counts and types reach
the spans.
"""

import inspect
from dataclasses import dataclass, field
from typing import Optional

from .attributes import PII_COUNT, PII_TYPES
from .detectors import BUSINESS_SCOPES, PiiDetector, normalize

ROLES = ("input", "context", "output")


@dataclass
class PiiScan:
    findings: list
    leaks: list = field(default_factory=list)

    @property
    def types(self) -> list:
        return sorted({f.type for f in self.findings})


def _add(span, count_key: str, types_key: str, findings: list) -> None:
    current = dict(getattr(span, "attributes", None) or {})
    count = int(current.get(count_key, 0)) + len(findings)
    types = set(filter(None, str(current.get(types_key, "")).split(","))) | {f.type for f in findings}
    span.set_attribute(count_key, count)
    span.set_attribute(types_key, ",".join(sorted(types)))


class PiiSession:
    """One conversation (one agent run): detect personal information by role and count the leaks

    Roles: "input" (from the user: its values are allowed back), "context" (retrieved documents, tool results: counted,
    never treated as provided by the user) and "output" (answers and tool calls: personal information not provided by
    the user is a leak). Use as a context manager so that the values held in memory are dropped at the end.

    `output_business_scope`: in outputs, a business word ("dossier", "order"...) or a word proper to a type
    ("client", "confirmation"...) lowers a finding only when it is right before the value ("adjacent", default:
    "Votre dossier 130 692 544 est en traitement" is not a leak, "Le dossier voisin indique aussi 130 692 544" is), or
    anywhere in the 40 characters before ("window", the previous behaviour, which hid leaks from agents that
    mention files). Inputs and contexts are scanned as by the detector; the values the user provided are taken from
    both scans, so that a value given by the user is never counted as a leak when repeated.
    """

    def __init__(self, detector: Optional[PiiDetector] = None, output_business_scope: str = "adjacent"):
        if output_business_scope not in BUSINESS_SCOPES:
            raise ValueError(f"output_business_scope must be one of {BUSINESS_SCOPES}")
        self.detector = detector or PiiDetector()
        self.output_business_scope = output_business_scope
        try:
            self._scoped = "business_scope" in inspect.signature(self.detector.scan).parameters
        except (TypeError, ValueError):  # pragma: no cover - builtins without a signature
            self._scoped = False
        self._provided = set()

    def _scan(self, text: str, scope: str) -> list:
        if scope == "window" or not self._scoped:  # a detector without scopes (e.g. a model) scans one way only
            return self.detector.scan(text)
        return self.detector.scan(text, business_scope=scope)

    def scan(self, role: str, text: str) -> PiiScan:
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        text = text or ""
        if role == "output":
            findings = self._scan(text, self.output_business_scope)
            return PiiScan(findings, [f for f in findings if normalize(f, text) not in self._provided])
        findings = self._scan(text, "window")
        if role == "input":
            self._provided |= {normalize(f, text) for f in findings}
            if self.output_business_scope != "window" and self._scoped:
                self._provided |= {normalize(f, text) for f in self._scan(text, self.output_business_scope)}
        return PiiScan(findings)

    def record(self, span, role: str, text: str) -> PiiScan:
        """Scan `text` and add the counts and types to `span` (pulsar.pii.<role>.count / .types, pulsar.pii.leak.*)"""
        result = self.scan(role, text)
        _add(span, PII_COUNT.format(role=role), PII_TYPES.format(role=role), result.findings)
        if role == "output":
            _add(span, PII_COUNT.format(role="leak"), PII_TYPES.format(role="leak"), result.leaks)
        return result

    def clear(self) -> None:
        self._provided.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.clear()
