"""Personal information in the conversations of LLM applications and agents, and its leaks.

A leak is personal information in an output (an answer, a tool call) that the user did not provide in the same
conversation, e.g. the social insurance number of someone else read in a database and repeated to the user. The
values given by the user are kept in memory, for the session only, to tell them apart; only counts and types reach
the spans.
"""

from dataclasses import dataclass, field
from typing import Optional

from .attributes import PII_COUNT, PII_TYPES
from .detectors import PiiDetector, normalize

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
    """

    def __init__(self, detector: Optional[PiiDetector] = None):
        self.detector = detector or PiiDetector()
        self._provided = set()

    def scan(self, role: str, text: str) -> PiiScan:
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        findings = self.detector.scan(text or "")
        values = {normalize(f, text) for f in findings}
        if role == "input":
            self._provided |= values
            return PiiScan(findings)
        if role == "output":
            return PiiScan(findings, [f for f in findings if normalize(f, text) not in self._provided])
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
