"""Turn finished GenAI spans into Pulsar records: LLM calls, tool calls and agent runs.

An agent run is the subtree of an `invoke_agent` span that has no `invoke_agent` ancestor. It is assembled when its
local root span ends (no parent, or a parent in another process), from the spans of the same trace already received
(children end before their parent). A trace is never assembled before its root arrives, however long the run takes;
a run still open after `max_run_s` is reported once by an `agent_run` record with `termination="incomplete"`.
"""

import hashlib
import json
import logging
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from . import semconv as sc

log = logging.getLogger(__name__)


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
    parent_remote: bool = False  # the parent span lives in another process: this span is a local root

    @property
    def operation(self) -> Optional[str]:
        return self.attributes.get(sc.OPERATION)

    @property
    def duration_s(self) -> float:
        return (self.end_ns - self.start_ns) / 1e9

    @property
    def failed(self) -> bool:
        return self.status_error or bool(self.attributes.get(sc.ERROR_TYPE))


@dataclass
class OpenRun:
    """An agent span that has started and not ended yet (seen by OpenRunTracker.on_start)"""

    trace_id: int
    span_id: int
    parent_id: Optional[int]
    name: str
    start_ns: int
    attributes: dict = field(default_factory=dict)


def _finish_reason(attrs: dict) -> Optional[str]:
    reasons = attrs.get(sc.FINISH_REASONS)
    if isinstance(reasons, (list, tuple)):
        return reasons[0] if reasons else None
    return reasons


def _pii_fields(spans: list) -> dict:
    """pii_<role>_count / pii_<role>_types summed over the spans, only when a PiiSession recorded something"""
    out = {}
    if not any(sc.PII_COUNT.format(role=role) in s.attributes for s in spans for role in sc.PII_ROLES):
        return out
    # every role is written (0 when absent) so that a batch never mixes integers and missing values, which the
    # storage would receive as floats and reject against the integers already stored
    for role in sc.PII_ROLES:
        count_key, types_key = sc.PII_COUNT.format(role=role), sc.PII_TYPES.format(role=role)
        recorded = [s for s in spans if count_key in s.attributes]
        if not recorded:
            out[f"pii_{role}_count"] = 0
            out[f"pii_{role}_types"] = "none"
            continue
        types = set()
        for s in recorded:
            types |= set(filter(None, str(s.attributes.get(types_key, "")).split(",")))
        out[f"pii_{role}_count"] = sum(int(s.attributes[count_key]) for s in recorded)
        out[f"pii_{role}_types"] = ",".join(sorted(types)) or "none"
    return out


def _detector_fields(spans: list) -> dict:
    """det_<name>: highest score of each detector over the spans (pulsar.detector.<name>, see genai.detectors)"""
    out = {}
    for s in spans:
        for key, value in s.attributes.items():
            if key.startswith("pulsar.detector.") and isinstance(value, (int, float)):
                name = "det_" + key[len("pulsar.detector.") :]
                out[name] = max(out.get(name, float("-inf")), float(value))
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
        **_detector_fields([span]),
    }


_TIMEOUT_ERROR = re.compile(r"time[d_ ]?out", re.IGNORECASE)


def tool_status(span: SpanView) -> str:
    """Outcome of a tool call: `pulsar.tool.status` when the application set it, else derived from the span

    A failed span is a "timeout" when its error.type names a timeout (TimeoutError, ReadTimeout, asyncio.TimeoutError...),
    else an "error"; a span whose output failed the schema check (`pulsar.tool.output_valid` False) is "invalid".
    """
    a = span.attributes
    status = a.get(sc.TOOL_STATUS)
    if status:
        return str(status).strip().lower()
    if span.failed:
        return sc.TOOL_TIMEOUT if _TIMEOUT_ERROR.search(str(a.get(sc.ERROR_TYPE) or "")) else sc.TOOL_ERROR
    if a.get(sc.TOOL_OUTPUT_VALID) is False:
        return "invalid"
    return sc.TOOL_OK


def _is_error(span: SpanView, status: str) -> bool:
    return span.failed or status in (sc.TOOL_ERROR, sc.TOOL_TIMEOUT)


def _is_soft_failure(span: SpanView, status: str) -> bool:
    """The call did not fail but returned nothing usable (status "retry", "invalid"..., or output_valid False)"""
    return not span.failed and (status in sc.SOFT_FAILURE_STATUSES or span.attributes.get(sc.TOOL_OUTPUT_VALID) is False)


def tool_call_record(span: SpanView, allowed_tools: Optional[set]) -> dict:
    a = span.attributes
    tool = a.get(sc.TOOL_NAME) or span.name
    status = tool_status(span)
    return {
        "tool_name": tool,
        "error": span.failed,
        "status": status,
        "timeout": status == sc.TOOL_TIMEOUT,
        "soft_failure": _is_soft_failure(span, status),
        "latency_s": span.duration_s,
        "out_of_scope": allowed_tools is not None and tool not in allowed_tools,
        "agent_name": a.get(sc.AGENT_NAME),
        "trace_id": format(span.trace_id, "032x"),
        **_pii_fields([span]),
        **_detector_fields([span]),
    }


def _raw_arguments(span: SpanView) -> str:
    args = span.attributes.get(sc.TOOL_CALL_ARGUMENTS, "")
    if not isinstance(args, str):
        args = json.dumps(args, sort_keys=True, default=str)
    return args


def _call_signature(span: SpanView) -> str:
    """Tool and raw arguments of a call, hashed: identical consecutive signatures give `repeated_calls`"""
    raw = f"{span.attributes.get(sc.TOOL_NAME) or span.name}|{_raw_arguments(span)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


_NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def _normalize_value(value):
    """Canonical form of an argument value: 36000, 36000.0 and "36000" are equal; strings are trimmed and lower-cased"""
    if isinstance(value, dict):
        return {str(k).strip().lower(): _normalize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(v) for v in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if _NUMBER.match(text):
            return float(text)
        return " ".join(text.lower().split())
    return str(value)


def normalized_signature(span: SpanView) -> str:
    """Tool and normalized arguments (JSON parsed, keys sorted, numbers and strings canonical), hashed"""
    raw = _raw_arguments(span)
    try:
        args = _normalize_value(json.loads(raw)) if raw else ""
    except (ValueError, TypeError):
        args = _normalize_value(raw)
    tool = str(span.attributes.get(sc.TOOL_NAME) or span.name).strip().lower()
    text = f"{tool}|{json.dumps(args, sort_keys=True)}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _window_repeats(signatures: list, window: int) -> int:
    """Calls whose signature already appeared among the `window` previous calls (non-consecutive repetitions)"""
    return sum(1 for i, sig in enumerate(signatures) if sig in signatures[max(0, i - window) : i])


def _cycle_repeats(names: list, max_period: int = 3) -> int:
    """Longest repetition in a row of a block of 2 to `max_period` tool names (A>B>A>B>A>B: 3), 0 without cycle"""
    best = 0
    for period in range(2, max_period + 1):
        for start in range(len(names) - 2 * period + 1):
            block = names[start : start + period]
            if len(set(block)) < 2:
                continue  # a single tool repeated is counted by max_same_tool_calls
            repeats = 1
            while names[start + repeats * period : start + (repeats + 1) * period] == block:
                repeats += 1
            if repeats >= 2:
                best = max(best, repeats)
    return best


def _field_name(tool: str) -> str:
    return re.sub(r"[^0-9a-z_]", "_", str(tool).lower())


def _catalog(spans: list) -> Optional[set]:
    """Tools offered to the model (pulsar.agent.available_tools on the agent spans), None when not recorded"""
    catalog = None
    for s in spans:
        if s.operation != sc.OP_INVOKE_AGENT or sc.AVAILABLE_TOOLS not in s.attributes:
            continue
        value = s.attributes[sc.AVAILABLE_TOOLS]
        names = value.split(",") if isinstance(value, str) else list(value or [])
        catalog = (catalog or set()) | {str(n).strip() for n in names if str(n).strip()}
    return catalog


def _tool_fields(tool_calls: list, tools: list, allowed_tools: Optional[set], catalog: Optional[set]) -> dict:
    """Per-tool and per-status counts, soft failures, tool catalogue and calls to tools that are not available"""
    statuses = [tool_status(s) for s in tool_calls]
    errors = [_is_error(s, st) for s, st in zip(tool_calls, statuses)]
    timeouts = [st == sc.TOOL_TIMEOUT for st in statuses]
    soft = [_is_soft_failure(s, st) for s, st in zip(tool_calls, statuses)]
    unavailable = [
        st in sc.UNAVAILABLE_STATUSES or (catalog is not None and tool not in catalog) for tool, st in zip(tools, statuses)
    ]
    out = {
        "tool_errors": sum(errors),
        "tool_timeouts": sum(timeouts),
        "tool_soft_failures": sum(soft),
        "soft_failure_rate": (sum(soft) / len(tool_calls)) if tool_calls else 0.0,
        "unavailable_tool_calls": sum(unavailable),
        "tool_status_counts": ",".join(f"{k}={v}" for k, v in sorted(Counter(statuses).items())) or "none",
    }
    # every tool of allowed_tools and of the catalogue is written (0 when not called), so that a per-tool test always
    # has a value for each run
    max_rate = 0.0
    for name in sorted(set(tools) | set(allowed_tools or ()) | set(catalog or ())):
        idx = [i for i, t in enumerate(tools) if t == name]
        key = _field_name(name)
        n_errors = sum(errors[i] for i in idx)
        out[f"tool_{key}_calls"] = len(idx)
        out[f"tool_{key}_errors"] = n_errors
        out[f"tool_{key}_timeouts"] = sum(timeouts[i] for i in idx)
        out[f"tool_{key}_soft_failures"] = sum(soft[i] for i in idx)
        if idx:
            max_rate = max(max_rate, n_errors / len(idx))
    out["max_tool_error_rate"] = max_rate
    if catalog is not None:
        ordered = sorted(catalog)
        out["tool_catalog_size"] = len(ordered)
        out["tool_catalog_hash"] = hashlib.sha256(",".join(ordered).encode("utf-8")).hexdigest()[:12]
        if allowed_tools is not None:
            missing = sorted(set(allowed_tools) - catalog)
            out["catalog_missing_tools"] = len(missing)
            out["missing_tools"] = ",".join(missing) or "none"
    return out


def _rule_id(rule) -> str:
    return str(getattr(rule, "id", rule))


def compliance_fields(compliance, tools: list, rules_version: Optional[str] = None) -> dict:
    """violated_rules / violations / critical_violations / rules_version of one run

    `compliance` is a pulsar_metrics ComplianceMonitor (any object with `violations(trace)`, and optionally `rules`
    carrying `id` / `severity` and `version`) or a callable trace -> violated rule ids. The trace is the list of the
    tool names called, as `pulsar_metrics.genai.compliance.trace_from(tool_sequence)` rebuilds it.
    """
    check = getattr(compliance, "violations", compliance)
    violated = sorted({_rule_id(r) for r in (check(list(tools)) or [])})
    severity = {_rule_id(r): str(getattr(r, "severity", "")).lower() for r in getattr(compliance, "rules", None) or []}
    return {
        "violated_rules": ",".join(violated) or "none",
        "violations": len(violated),
        "critical_violations": sum(1 for r in violated if severity.get(r) == "critical"),
        "rules_version": str(rules_version or getattr(compliance, "version", None) or "unversioned"),
    }


def agent_run_record(
    root: SpanView,
    descendants: list[SpanView],
    allowed_tools: Optional[set],
    *,
    loop_window: int = 4,
    loop_min_repeats: int = 2,
    incomplete: bool = False,
) -> dict:
    """Trajectory features of one agent run

    `repeated_calls` keeps its historical meaning (consecutive calls identical byte for byte). `repeated_calls_window`
    counts the calls whose normalized signature (tool + arguments, see `normalized_signature`) appeared among the
    `loop_window` previous calls; `loop_detected` is True from `loop_min_repeats` such calls.
    """
    a = root.attributes
    llm_calls = [s for s in descendants if s.operation in sc.LLM_OPERATIONS]
    tool_calls = sorted((s for s in descendants if s.operation == sc.OP_EXECUTE_TOOL), key=lambda s: s.start_ns)
    tools = [s.attributes.get(sc.TOOL_NAME) or s.name for s in tool_calls]
    signatures = [_call_signature(s) for s in tool_calls]
    repeated = sum(1 for previous, current in zip(signatures, signatures[1:]) if previous == current)
    repeated_window = _window_repeats([normalized_signature(s) for s in tool_calls], loop_window)
    counts = Counter(tools)

    termination = a.get(sc.TERMINATION) or ("error" if root.failed else "completed")
    record = {
        "agent_name": a.get(sc.AGENT_NAME) or root.name,
        "steps": len(llm_calls),
        "tool_calls": len(tool_calls),
        "distinct_tools": len(counts),
        "tool_error_rate": (sum(s.failed for s in tool_calls) / len(tool_calls)) if tool_calls else 0.0,
        "repeated_calls": repeated,
        "repeated_calls_window": repeated_window,
        "max_same_tool_calls": max(counts.values()) if counts else 0,
        "tool_cycle_repeats": _cycle_repeats(tools),
        "loop_detected": repeated_window >= loop_min_repeats,
        "out_of_scope_calls": sum(1 for t in tools if allowed_tools is not None and t not in allowed_tools),
        "delegation_depth": _delegation_depth(root, descendants),
        "input_tokens": sum(int(s.attributes.get(sc.INPUT_TOKENS) or 0) for s in llm_calls),
        "output_tokens": sum(int(s.attributes.get(sc.OUTPUT_TOKENS) or 0) for s in llm_calls),
        "latency_s": root.duration_s,
        "termination": termination,
        "incomplete": incomplete,
        "error": root.failed,
        "first_tool": tools[0] if tools else "none",
        "tool_sequence": ">".join(tools) if tools else "none",
        "request_model": next(
            (s.attributes.get(sc.REQUEST_MODEL) for s in llm_calls if s.attributes.get(sc.REQUEST_MODEL)), None
        ),
        "prompt_version": a.get(sc.PROMPT_VERSION),
        "trace_id": format(root.trace_id, "032x"),
    }
    record.update(_tool_fields(tool_calls, tools, allowed_tools, _catalog([root, *descendants])))
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
    record.update(_detector_fields([root, *descendants]))
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


def _seconds_to_ns(seconds: Optional[float]) -> Optional[int]:
    return None if seconds is None else int(seconds * 1e9)


class TrajectoryAssembler:
    """Accumulate finished spans and emit records as soon as they are complete

    LLM calls and tool calls are emitted as soon as their span ends. Agent runs are emitted when their local root ends:
    the root span of the trace (no parent), or a span whose parent lives in another process (`parent_remote`).
    Children end before their parent, so a sub-agent cannot be mistaken for a run.

    By default (`wait_for_root=True`) a trace is never assembled before its root arrives, however long the run: a run
    longer than `trace_timeout_s` used to be assembled without its root, its LLM and tool calls were then lost. A trace
    whose root never arrives is assembled after `orphan_timeout_s` without new span, or when more than
    `max_pending_traces` traces are waiting; it is then counted in `stats()["partial_traces_dropped"]` and its agent
    runs, if any, are marked `incomplete`.

    Parameters
    ----------
    allowed_tools : set of str, optional
        Tools the agent is allowed to call; calls outside the list are counted as out-of-scope
    trace_timeout_s : float, optional
        Only with `wait_for_root=False` (former behaviour): inactivity after which a trace whose root is not received is
        assembled with the spans received (default 30 s)
    max_pending_traces : int, optional
        Traces kept in memory; beyond, the oldest is assembled immediately and counted as partial (default 10,000)
    wait_for_root : bool, optional
        Never assemble a trace before its root on inactivity alone (default True)
    orphan_timeout_s : float, optional
        With `wait_for_root`: inactivity after which a trace without root is given up (default 3600 s; None: never,
        memory is then bounded by `max_pending_traces` only)
    max_run_s : float, optional
        A run open for longer is reported once by an `agent_run` with `termination="incomplete"` and
        `incomplete=True`, built from the spans received so far; the complete record follows if the run ends
        (default 600 s; None: off). Runs are known from their start with `OpenRunTracker`, otherwise from the first
        span of their trace that ended.
    loop_window, loop_min_repeats : int, optional
        `loop_detected` is True when at least `loop_min_repeats` calls repeat (normalized tool + arguments) a call among
        the `loop_window` previous ones (defaults 4 and 2)
    compliance : ComplianceMonitor or callable, optional
        Rules checked on each run (pulsar_metrics.genai.ComplianceMonitor, or any callable trace -> violated rule ids):
        adds `violated_rules`, `violations`, `critical_violations` and `rules_version` to the agent runs
    rules_version : str, optional
        Version of the rules written with the violations (default: `compliance.version`, else "unversioned")
    """

    def __init__(
        self,
        allowed_tools: Optional[set] = None,
        trace_timeout_s: float = 30.0,
        max_pending_traces: int = 10_000,
        *,
        wait_for_root: bool = True,
        orphan_timeout_s: Optional[float] = 3600.0,
        max_run_s: Optional[float] = 600.0,
        loop_window: int = 4,
        loop_min_repeats: int = 2,
        compliance: Any = None,
        rules_version: Optional[str] = None,
    ):
        self.allowed_tools = set(allowed_tools) if allowed_tools is not None else None
        self.trace_timeout_ns = int(trace_timeout_s * 1e9)
        self.max_pending_traces = max_pending_traces
        self.wait_for_root = wait_for_root
        self.orphan_timeout_ns = _seconds_to_ns(orphan_timeout_s)
        self.max_run_ns = _seconds_to_ns(max_run_s)
        self.loop_window = loop_window
        self.loop_min_repeats = loop_min_repeats
        self.compliance = compliance
        self.rules_version = rules_version
        self._pending: dict[int, list[SpanView]] = {}
        self._last_seen: dict[int, int] = {}
        self._first_start: dict[int, int] = {}
        self._open: dict[int, OpenRun] = {}  # span id -> open run (top-level agent span started, not ended)
        self._open_agents: dict[int, int] = {}  # span id -> trace id of every open agent span, sub-agents included
        # span id -> (trace id, parent id) of the open agent spans and of the open spans below them, so that a sub-agent
        # under an intermediate span (e.g. a tool that delegates) is recognised as part of the run
        self._open_tree: dict[int, tuple] = {}
        self._reported: dict[int, set] = defaultdict(set)  # trace id -> runs already reported incomplete (span id/None)
        self._counters = Counter(partial_traces_dropped=0, partial_spans_dropped=0, incomplete_runs=0, compliance_errors=0)

    def stats(self) -> dict:
        """partial_traces_dropped, partial_spans_dropped, incomplete_runs, compliance_errors, pending_traces, open_runs"""
        return {**self._counters, "pending_traces": len(self._pending), "open_runs": len(self._open)}

    def run_started(self, run: OpenRun) -> None:
        """Register a span that has just started (called by OpenRunTracker), to report its run if it never ends

        Agent spans whose ancestors include no open agent span are top-level runs; other spans are kept only when they
        are below an open agent span (to recognise the sub-agents started under them).
        """
        attributes = sc.normalize(run.attributes)
        node = self._open_tree.get(run.parent_id) if run.parent_id is not None else None
        under_agent = node is not None and node[0] == run.trace_id  # the tree only holds agent spans and their subtrees
        if attributes.get(sc.OPERATION) == sc.OP_INVOKE_AGENT:
            self._open_tree[run.span_id] = (run.trace_id, run.parent_id)
            self._open_agents[run.span_id] = run.trace_id
            if not under_agent:
                self._open[run.span_id] = OpenRun(run.trace_id, run.span_id, run.parent_id, run.name, run.start_ns, attributes)
        elif under_agent:
            self._open_tree[run.span_id] = (run.trace_id, run.parent_id)
        for registry, limit in ((self._open, self.max_pending_traces), (self._open_agents, self.max_pending_traces)):
            while len(registry) > limit:
                registry.pop(next(iter(registry)))
        while len(self._open_tree) > 10 * self.max_pending_traces:
            self._open_tree.pop(next(iter(self._open_tree)))

    def add(self, span: SpanView, now_ns: Optional[int] = None) -> list[tuple[str, dict, int]]:
        """Return the records completed by this span, as (record_type, record, end_time_ns)"""
        span.attributes = sc.normalize(span.attributes)
        records = []
        if span.operation in sc.LLM_OPERATIONS:
            records.append(("llm_call", llm_call_record(span), span.end_ns))
        elif span.operation == sc.OP_EXECUTE_TOOL:
            records.append(("tool_call", tool_call_record(span, self.allowed_tools), span.end_ns))
        if span.operation == sc.OP_INVOKE_AGENT:
            self._open.pop(span.span_id, None)
            self._open_agents.pop(span.span_id, None)
        self._open_tree.pop(span.span_id, None)

        trace_id = span.trace_id
        self._pending.setdefault(trace_id, []).append(span)
        self._last_seen[trace_id] = now_ns if now_ns is not None else span.end_ns
        self._first_start[trace_id] = min(self._first_start.get(trace_id, span.start_ns), span.start_ns)
        if span.parent_id is None:
            records += self._assemble(trace_id)
        elif span.parent_remote:
            records += self._assemble(trace_id, root=span)
        elif len(self._pending) > self.max_pending_traces:
            records += self._assemble(next(iter(self._pending)), partial=True)
        return records

    def flush_idle(self, now_ns: int) -> list[tuple[str, dict, int]]:
        """Report the runs open for longer than `max_run_s`, and give up the traces without root that are idle

        With `wait_for_root` (default), a trace is given up after `orphan_timeout_s` without new span; otherwise after
        `trace_timeout_s` (former behaviour). Given-up traces are counted as partial.
        """
        records = self.overdue_runs(now_ns)
        timeout = self.orphan_timeout_ns if self.wait_for_root else self.trace_timeout_ns
        if timeout is not None:
            idle = [t for t, seen in self._last_seen.items() if now_ns - seen >= timeout]
            records += [record for trace_id in idle for record in self._assemble(trace_id, partial=True)]
        self._prune(now_ns)
        return records

    def _prune(self, now_ns: int) -> None:
        """Forget the hung runs already reported once `orphan_timeout_s` has passed, and the reports of forgotten traces"""
        if self.orphan_timeout_ns is not None:
            for run in list(self._open.values()):
                hung = now_ns - run.start_ns >= self.orphan_timeout_ns
                if hung and run.span_id in self._reported.get(run.trace_id, ()):
                    self._drop_open(run.trace_id)
        live = set(self._pending) | {run.trace_id for run in self._open.values()}
        for trace_id in [t for t in self._reported if t not in live]:
            del self._reported[trace_id]

    def _drop_open(self, trace_id: int) -> None:
        """Stop tracking the open spans of a trace that is given up"""
        for registry in (self._open, self._open_agents):
            for span_id in [k for k, v in registry.items() if (v.trace_id if isinstance(v, OpenRun) else v) == trace_id]:
                del registry[span_id]
        for span_id in [k for k, (t, _) in self._open_tree.items() if t == trace_id]:
            del self._open_tree[span_id]

    def overdue_runs(self, now_ns: int, max_run_ns: Optional[int] = None, infer: bool = True) -> list[tuple[str, dict, int]]:
        """One `incomplete` agent_run for each run open for longer than `max_run_s` and not reported yet

        Runs registered by `run_started` are known from their start. With `infer`, a trace whose root has not arrived
        `max_run_s` after the start of its first ended span is reported too, from the ended spans that carry an agent
        name (gen_ai.agent.name) and are not below an agent span already received; never from LLM or tool calls alone.
        """
        limit = self.max_run_ns if max_run_ns is None else max_run_ns
        if limit is None:
            return []
        records = []
        for run in list(self._open.values()):
            if now_ns - run.start_ns < limit or run.span_id in self._reported.get(run.trace_id, ()):
                continue
            root = SpanView(run.trace_id, run.span_id, run.parent_id, run.name, run.start_ns, now_ns, False, dict(run.attributes))
            descendants = self._descendants(root, self._pending.get(run.trace_id, []))
            records.append(self._incomplete_record(root, descendants, key=run.span_id, now_ns=now_ns))
        if not infer:
            return records
        # without OpenRunTracker, an open run is inferred only from ended spans that carry an agent name and do not
        # belong to an agent span already received (whose run is complete, e.g. under a long-lived session root)
        open_traces = {run.trace_id for run in self._open.values()}
        for trace_id, spans in list(self._pending.items()):
            if trace_id in open_traces or None in self._reported.get(trace_id, ()):
                continue
            start = self._first_start.get(trace_id, now_ns)
            if now_ns - start < limit:
                continue
            by_id = {s.span_id: s for s in spans}
            covered = set()
            for s in spans:
                if s.operation == sc.OP_INVOKE_AGENT and not self._has_agent_ancestor(s, by_id):
                    covered |= {s.span_id, *(d.span_id for d in self._descendants(s, spans))}
            loose = [s for s in spans if s.span_id not in covered]
            name = next((s.attributes.get(sc.AGENT_NAME) for s in loose if s.attributes.get(sc.AGENT_NAME)), None)
            if not name:
                continue  # no agent identity: never synthesize a run
            attributes = {sc.OPERATION: sc.OP_INVOKE_AGENT, sc.AGENT_NAME: name}
            root = SpanView(trace_id, 0, None, name, start, now_ns, False, attributes)
            records.append(self._incomplete_record(root, loose, key=None, now_ns=now_ns))
        return records

    def flush_all(self, now_ns: Optional[int] = None) -> list[tuple[str, dict, int]]:
        """Report every open run as incomplete and assemble every pending trace (shutdown)"""
        now_ns = time.time_ns() if now_ns is None else now_ns
        records = self.overdue_runs(now_ns, max_run_ns=0, infer=False)
        records += [record for trace_id in list(self._pending) for record in self._assemble(trace_id, partial=True)]
        self._open.clear()
        self._open_agents.clear()
        self._open_tree.clear()
        self._reported.clear()
        return records

    def _incomplete_record(self, root: SpanView, descendants: list, key, now_ns: int) -> tuple[str, dict, int]:
        root.attributes[sc.TERMINATION] = sc.TERMINATION_INCOMPLETE
        root.attributes.pop(sc.TASK_SUCCESS, None)
        self._reported[root.trace_id].add(key)
        self._counters["incomplete_runs"] += 1
        return "agent_run", self._run_record(root, descendants, incomplete=True), now_ns

    def _run_record(self, root: SpanView, descendants: list, incomplete: bool = False) -> dict:
        record = agent_run_record(
            root,
            descendants,
            self.allowed_tools,
            loop_window=self.loop_window,
            loop_min_repeats=self.loop_min_repeats,
            incomplete=incomplete,
        )
        if self.compliance is not None:
            tools = [] if record["tool_sequence"] == "none" else record["tool_sequence"].split(">")
            try:
                record.update(compliance_fields(self.compliance, tools, self.rules_version))
            except Exception:  # noqa: BLE001 - a faulty rule set must never stop the collection
                self._counters["compliance_errors"] += 1
                log.warning("compliance check failed for trace %s", record["trace_id"], exc_info=True)
        return record

    def _assemble(self, trace_id: int, root: Optional[SpanView] = None, partial: bool = False) -> list:
        trace = self._pending.get(trace_id, [])
        if root is not None:  # a local root under a remote parent: only its subtree is complete
            subtree = [root, *self._descendants(root, trace)]
            ids = {s.span_id for s in subtree}
            rest = [s for s in trace if s.span_id not in ids]
            if rest:
                self._pending[trace_id] = rest
                self._reported.get(trace_id, set()).discard(root.span_id)
            else:
                self._forget(trace_id)
            trace = subtree
        else:
            self._forget(trace_id)
            if partial:
                self._drop_open(trace_id)
        by_id = {s.span_id: s for s in trace}
        records, covered = [], set()
        for span in trace:
            if span.operation != sc.OP_INVOKE_AGENT or self._has_agent_ancestor(span, by_id):
                continue
            descendants = self._descendants(span, trace)
            covered |= {span.span_id, *(s.span_id for s in descendants)}
            records.append(("agent_run", self._run_record(span, descendants, incomplete=partial), span.end_ns))
        if partial and trace:
            self._counters["partial_traces_dropped"] += 1
            self._counters["partial_spans_dropped"] += len(trace) - len(covered)
        return records

    def _forget(self, trace_id: int) -> None:
        self._pending.pop(trace_id, None)
        self._last_seen.pop(trace_id, None)
        self._first_start.pop(trace_id, None)
        self._reported.pop(trace_id, None)

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
