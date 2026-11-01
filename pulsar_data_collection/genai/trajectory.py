"""Turn finished GenAI spans into Pulsar records: LLM calls, tool calls and agent runs.

An agent run is the subtree of an `invoke_agent` span that has no `invoke_agent` ancestor. It is assembled when its
root span ends, from the spans of the same trace already received (children end before their parent).
"""

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

from . import semconv as sc


@dataclass
class SpanView:
    """Minimal, SDK-independent view of a finished span"""

    trace_id: int
    span_id: int
    parent_id: Optional[int]
    name: str
    start_ns: int
    end_ns: int
    status_error: bool
    attributes: dict = field(default_factory=dict)

    @property
    def operation(self) -> Optional[str]:
        return self.attributes.get(sc.OPERATION)

    @property
    def duration_s(self) -> float:
        return (self.end_ns - self.start_ns) / 1e9

    @property
    def failed(self) -> bool:
        return self.status_error or bool(self.attributes.get(sc.ERROR_TYPE))


def _finish_reason(attrs: dict) -> Optional[str]:
    reasons = attrs.get(sc.FINISH_REASONS)
    if isinstance(reasons, (list, tuple)):
        return reasons[0] if reasons else None
    return reasons


def _pii_fields(spans: list) -> dict:
    """pii_<role>_count / pii_<role>_types summed over the spans, only when a PiiSession recorded something"""
    out = {}
    for role in sc.PII_ROLES:
        count_key, types_key = sc.PII_COUNT.format(role=role), sc.PII_TYPES.format(role=role)
        recorded = [s for s in spans if count_key in s.attributes]
        if not recorded:
            continue
        types = set()
        for s in recorded:
            types |= set(filter(None, str(s.attributes.get(types_key, "")).split(",")))
        out[f"pii_{role}_count"] = sum(int(s.attributes[count_key]) for s in recorded)
        out[f"pii_{role}_types"] = ",".join(sorted(types)) or "none"
    return out


def llm_call_record(span: SpanView) -> dict:
    a = span.attributes
    return {
        "request_model": a.get(sc.REQUEST_MODEL),
        "response_model": a.get(sc.RESPONSE_MODEL),
        "provider": a.get(sc.PROVIDER),
        "temperature": a.get(sc.REQUEST_TEMPERATURE),
        "input_tokens": a.get(sc.INPUT_TOKENS),
        "output_tokens": a.get(sc.OUTPUT_TOKENS),
        "finish_reason": _finish_reason(a),
        "latency_s": span.duration_s,
        "error": span.failed,
        "agent_name": a.get(sc.AGENT_NAME),
        "trace_id": format(span.trace_id, "032x"),
        **_pii_fields([span]),
    }


def tool_call_record(span: SpanView, allowed_tools: Optional[set]) -> dict:
    a = span.attributes
    tool = a.get(sc.TOOL_NAME) or span.name
    return {
        "tool_name": tool,
        "error": span.failed,
        "latency_s": span.duration_s,
        "out_of_scope": allowed_tools is not None and tool not in allowed_tools,
        "agent_name": a.get(sc.AGENT_NAME),
        "trace_id": format(span.trace_id, "032x"),
        **_pii_fields([span]),
    }


def _call_signature(span: SpanView) -> str:
    """Tool and arguments of a call, hashed: identical consecutive signatures reveal a loop"""
    a = span.attributes
    args = a.get(sc.TOOL_CALL_ARGUMENTS, "")
    if not isinstance(args, str):
        args = json.dumps(args, sort_keys=True, default=str)
    raw = f"{a.get(sc.TOOL_NAME) or span.name}|{args}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def agent_run_record(root: SpanView, descendants: list[SpanView], allowed_tools: Optional[set]) -> dict:
    """Trajectory features of one agent run"""
    a = root.attributes
    llm_calls = [s for s in descendants if s.operation in sc.LLM_OPERATIONS]
    tool_calls = sorted((s for s in descendants if s.operation == sc.OP_EXECUTE_TOOL), key=lambda s: s.start_ns)
    tools = [s.attributes.get(sc.TOOL_NAME) or s.name for s in tool_calls]
    signatures = [_call_signature(s) for s in tool_calls]
    repeated = sum(1 for previous, current in zip(signatures, signatures[1:]) if previous == current)
    counts = Counter(tools)

    termination = a.get(sc.TERMINATION) or ("error" if root.failed else "completed")
    record = {
        "agent_name": a.get(sc.AGENT_NAME) or root.name,
        "steps": len(llm_calls),
        "tool_calls": len(tool_calls),
        "distinct_tools": len(counts),
        "tool_error_rate": (sum(s.failed for s in tool_calls) / len(tool_calls)) if tool_calls else 0.0,
        "repeated_calls": repeated,
        "loop_detected": repeated >= 2,
        "out_of_scope_calls": sum(1 for t in tools if allowed_tools is not None and t not in allowed_tools),
        "delegation_depth": _delegation_depth(root, descendants),
        "input_tokens": sum(int(s.attributes.get(sc.INPUT_TOKENS) or 0) for s in llm_calls),
        "output_tokens": sum(int(s.attributes.get(sc.OUTPUT_TOKENS) or 0) for s in llm_calls),
        "latency_s": root.duration_s,
        "termination": termination,
        "error": root.failed,
        "first_tool": tools[0] if tools else "none",
        "tool_sequence": ">".join(tools) if tools else "none",
        "request_model": next(
            (s.attributes.get(sc.REQUEST_MODEL) for s in llm_calls if s.attributes.get(sc.REQUEST_MODEL)), None
        ),
        "prompt_version": a.get(sc.PROMPT_VERSION),
        "trace_id": format(root.trace_id, "032x"),
    }
    if sc.TASK_SUCCESS in a:
        record["task_success"] = bool(a[sc.TASK_SUCCESS])
    for key, attribute in (
        ("cf_set_id", sc.CF_SET_ID),
        ("cf_variant", sc.CF_VARIANT),
        ("cf_attribute", sc.CF_ATTRIBUTE),
        ("cf_control", sc.CF_CONTROL),
    ):
        if attribute in a:
            record[key] = str(a[attribute])
    record.update(_pii_fields([root, *descendants]))
    return record


def _delegation_depth(root: SpanView, descendants: list[SpanView]) -> int:
    """Number of nested invoke_agent levels below the root (0 = no sub-agent)"""
    by_id = {s.span_id: s for s in descendants}
    depth = 0
    for span in descendants:
        if span.operation != sc.OP_INVOKE_AGENT:
            continue
        level, parent = 1, by_id.get(span.parent_id)
        while parent is not None:
            if parent.operation == sc.OP_INVOKE_AGENT:
                level += 1
            parent = by_id.get(parent.parent_id)
        depth = max(depth, level)
    return depth


class TrajectoryAssembler:
    """Accumulate finished spans and emit records as soon as they are complete

    LLM calls and tool calls are emitted as soon as their span ends. Agent runs are emitted when the whole trace is
    known: when its root span (no parent) ends, or, if the root belongs to another process, after `trace_timeout_s`
    without new span. Children end before their parent, so a sub-agent cannot be mistaken for a run.

    Parameters
    ----------
    allowed_tools : set of str, optional
        Tools the agent is allowed to call; calls outside the list are counted as out-of-scope
    trace_timeout_s : float, optional
        Inactivity after which an incomplete trace is assembled with the spans received (default 30 s)
    max_pending_traces : int, optional
        Traces kept in memory; beyond, the oldest is assembled immediately (default 10,000)
    """

    def __init__(self, allowed_tools: Optional[set] = None, trace_timeout_s: float = 30.0, max_pending_traces: int = 10_000):
        self.allowed_tools = set(allowed_tools) if allowed_tools is not None else None
        self.trace_timeout_ns = int(trace_timeout_s * 1e9)
        self.max_pending_traces = max_pending_traces
        self._pending: dict[int, list[SpanView]] = {}
        self._last_seen: dict[int, int] = {}

    def add(self, span: SpanView, now_ns: Optional[int] = None) -> list[tuple[str, dict, int]]:
        """Return the records completed by this span, as (record_type, record, end_time_ns)"""
        span.attributes = sc.normalize(span.attributes)
        records = []
        if span.operation in sc.LLM_OPERATIONS:
            records.append(("llm_call", llm_call_record(span), span.end_ns))
        elif span.operation == sc.OP_EXECUTE_TOOL:
            records.append(("tool_call", tool_call_record(span, self.allowed_tools), span.end_ns))

        self._pending.setdefault(span.trace_id, []).append(span)
        self._last_seen[span.trace_id] = now_ns if now_ns is not None else span.end_ns
        if span.parent_id is None:
            records += self._assemble(span.trace_id)
        elif len(self._pending) > self.max_pending_traces:
            records += self._assemble(next(iter(self._pending)))
        return records

    def flush_idle(self, now_ns: int) -> list[tuple[str, dict, int]]:
        """Assemble the traces without new span for `trace_timeout_s`"""
        idle = [t for t, seen in self._last_seen.items() if now_ns - seen >= self.trace_timeout_ns]
        return [record for trace_id in idle for record in self._assemble(trace_id)]

    def flush_all(self) -> list[tuple[str, dict, int]]:
        """Assemble every pending trace (shutdown)"""
        return [record for trace_id in list(self._pending) for record in self._assemble(trace_id)]

    def _assemble(self, trace_id: int) -> list[tuple[str, dict, int]]:
        trace = self._pending.pop(trace_id, [])
        self._last_seen.pop(trace_id, None)
        by_id = {s.span_id: s for s in trace}
        records = []
        for span in trace:
            if span.operation != sc.OP_INVOKE_AGENT or self._has_agent_ancestor(span, by_id):
                continue
            descendants = self._descendants(span, trace)
            records.append(("agent_run", agent_run_record(span, descendants, self.allowed_tools), span.end_ns))
        return records

    @staticmethod
    def _has_agent_ancestor(span: SpanView, by_id: dict) -> bool:
        parent = by_id.get(span.parent_id)
        while parent is not None:
            if parent.operation == sc.OP_INVOKE_AGENT:
                return True
            parent = by_id.get(parent.parent_id)
        return False

    @staticmethod
    def _descendants(root: SpanView, trace: list[SpanView]) -> list[SpanView]:
        children = defaultdict(list)
        for s in trace:
            children[s.parent_id].append(s)
        result, stack = [], [root.span_id]
        while stack:
            for child in children.get(stack.pop(), []):
                result.append(child)
                stack.append(child.span_id)
        return result
