"""OpenTelemetry SpanExporter turning GenAI spans into Pulsar records written with Pulse."""

import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Optional, Protocol, Sequence

import pandas as pd
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import StatusCode

from ..models import CaptureOptions, DataWithPrediction, PulseParameters
from ..pulse import Pulse
from . import semconv as sc
from .trajectory import OpenRun, SpanView, TrajectoryAssembler

log = logging.getLogger(__name__)

RECORD_TYPES = ("agent_run", "tool_call", "llm_call")


class RecordSink(Protocol):
    def write(self, record_type: str, record: dict, end_ns: int) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


class PulseSink:
    """Write each record type with its own Pulse, tagged `record_type` (agent_run, tool_call, llm_call)

    All records share the measurement `{model_id}_{model_version}_input_data` and the tags of `params`; queries select a
    record type with `r["record_type"] == "agent_run"`. Writes are non-blocking (see CaptureOptions).
    """

    def __init__(self, params: PulseParameters, options: Optional[CaptureOptions] = None):
        self.params = params
        self.options = options
        self._pulses: dict[str, Pulse] = {}

    def _pulse(self, record_type: str) -> Pulse:
        if record_type not in self._pulses:
            tags = {**(self.params.additional_tags or {}), "record_type": record_type, "schema": sc.SCHEMA_VERSION}
            self._pulses[record_type] = Pulse(self.params.model_copy(update={"additional_tags": tags}), self.options)
        return self._pulses[record_type]

    def write(self, record_type: str, record: dict, end_ns: int) -> None:
        row = pd.DataFrame([{k: v for k, v in record.items() if v is not None}])
        self._pulse(record_type).capture_data(
            DataWithPrediction(
                data_points=row,
                predictions=None,
                timestamp=datetime.fromtimestamp(end_ns / 1e9, tz=timezone.utc),
                features_names=list(row.columns),
            )
        )

    def flush(self) -> None:
        for pulse in self._pulses.values():
            pulse.flush()

    def stats(self) -> dict:
        return {record_type: pulse.stats() for record_type, pulse in self._pulses.items()}

    def close(self) -> None:
        for pulse in self._pulses.values():
            pulse.close()


class ListSink:
    """Keep records in memory (tests, notebooks)"""

    def __init__(self):
        self.records: list[tuple[str, dict, int]] = []

    def write(self, record_type: str, record: dict, end_ns: int) -> None:
        self.records.append((record_type, record, end_ns))

    def of(self, record_type: str) -> list[dict]:
        return [r for t, r, _ in self.records if t == record_type]

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


def span_view(span: ReadableSpan) -> SpanView:
    return SpanView(
        trace_id=span.context.trace_id,
        span_id=span.context.span_id,
        parent_id=span.parent.span_id if span.parent is not None else None,
        name=span.name,
        start_ns=span.start_time,
        end_ns=span.end_time,
        status_error=span.status.status_code == StatusCode.ERROR,
        attributes=dict(span.attributes or {}),
        parent_remote=bool(span.parent is not None and span.parent.is_remote),
    )


class PulsarSpanExporter(SpanExporter):
    """OpenTelemetry exporter: GenAI spans -> agent runs, tool calls and LLM calls -> sink (InfluxDB via Pulse)

    Only features are kept: prompts, responses and tool arguments are not written (arguments are hashed to detect
    loops). Use with a BatchSpanProcessor so that the application thread never waits. Register `run_tracker()` as a
    second span processor to report the runs that never end (`termination="incomplete"` after `max_run_s`).

    Parameters
    ----------
    sink : RecordSink
        Usually PulseSink(params, options)
    allowed_tools : set of str, optional
        Tools the agent may call; others are counted as out-of-scope calls
    trace_timeout_s : float, optional
        Only with `wait_for_root=False`: inactivity after which a trace whose root span is not received is assembled
        anyway (default 30 s; this former behaviour lost the LLM and tool calls of the runs longer than 30 s)
    wait_for_root, orphan_timeout_s, max_run_s, max_pending_traces : optional
        Assembly of the traces, see TrajectoryAssembler
    loop_window, loop_min_repeats, compliance, rules_version : optional
        Agent run features, see TrajectoryAssembler
    """

    def __init__(
        self,
        sink: RecordSink,
        allowed_tools: Optional[set] = None,
        trace_timeout_s: float = 30.0,
        *,
        wait_for_root: bool = True,
        orphan_timeout_s: Optional[float] = 3600.0,
        max_run_s: Optional[float] = 600.0,
        max_pending_traces: int = 10_000,
        loop_window: int = 4,
        loop_min_repeats: int = 2,
        compliance: Any = None,
        rules_version: Optional[str] = None,
    ):
        self.sink = sink
        self.assembler = TrajectoryAssembler(
            allowed_tools=allowed_tools,
            trace_timeout_s=trace_timeout_s,
            max_pending_traces=max_pending_traces,
            wait_for_root=wait_for_root,
            orphan_timeout_s=orphan_timeout_s,
            max_run_s=max_run_s,
            loop_window=loop_window,
            loop_min_repeats=loop_min_repeats,
            compliance=compliance,
            rules_version=rules_version,
        )
        self._lock = threading.Lock()
        self._started: deque = deque()  # filled by OpenRunTracker.on_start without taking the lock

    def _drain_started(self) -> None:
        while self._started:
            self.assembler.run_started(self._started.popleft())

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        now = time.time_ns()
        with self._lock:
            self._drain_started()
            for span in spans:
                for record in self.assembler.add(span_view(span), now_ns=now):
                    self.sink.write(*record)
            for record in self.assembler.flush_idle(now):
                self.sink.write(*record)
        return SpanExportResult.SUCCESS

    def check_open_runs(self, now_ns: Optional[int] = None) -> int:
        """Write the `incomplete` records of the runs open for longer than `max_run_s`; return how many were written

        Called periodically by `run_tracker()`, and on every export.
        """
        now = time.time_ns() if now_ns is None else now_ns
        with self._lock:
            self._drain_started()
            records = self.assembler.flush_idle(now)
            for record in records:
                self.sink.write(*record)
        return sum(1 for record in records if record[1].get("incomplete"))

    def run_tracker(self, check_interval_s: Optional[float] = 10.0) -> "OpenRunTracker":
        """Span processor that sees the agent spans start, so that a run that never ends is still reported"""
        return OpenRunTracker(self, check_interval_s=check_interval_s)

    def stats(self) -> dict:
        """Assembly counters: partial_traces_dropped, partial_spans_dropped, incomplete_runs, compliance_errors,
        pending_traces, open_runs"""
        with self._lock:
            self._drain_started()
            return self.assembler.stats()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        self.sink.flush()
        return True

    def shutdown(self) -> None:
        with self._lock:
            self._drain_started()
            for record in self.assembler.flush_all():
                self.sink.write(*record)
        self.sink.close()


class OpenRunTracker(SpanProcessor):
    """Span processor registering the start of the agent spans with a PulsarSpanExporter

    The exporter only receives finished spans: a run blocked in a model call that never returns sends nothing. With
    this processor, such a run is reported by an `agent_run` record (`termination="incomplete"`, `incomplete=True`,
    `latency_s` = time open so far) once it has been open for `max_run_s`. `on_start` only appends to a queue: the
    application thread never waits for the exporter. A background thread checks every `check_interval_s` seconds
    (None: no thread; checks then happen on each export and on `exporter.check_open_runs()`).

    Usage: `provider.add_span_processor(BatchSpanProcessor(exporter))`, then
    `provider.add_span_processor(exporter.run_tracker())`.
    """

    def __init__(self, exporter: PulsarSpanExporter, check_interval_s: Optional[float] = 10.0):
        self._exporter = exporter
        self._stop = threading.Event()
        self._thread = None
        if check_interval_s:
            self._thread = threading.Thread(target=self._loop, args=(check_interval_s,), name="pulsar-open-runs", daemon=True)
            self._thread.start()

    def _loop(self, interval_s: float) -> None:
        while not self._stop.wait(interval_s):
            try:
                self._exporter.check_open_runs()
            except Exception:  # noqa: BLE001 - the watchdog must survive a faulty sink
                log.warning("checking the open agent runs failed", exc_info=True)

    def on_start(self, span, parent_context=None) -> None:
        attributes = span.attributes or {}
        if attributes.get(sc.OPERATION) != sc.OP_INVOKE_AGENT:
            return
        parent = span.parent
        self._exporter._started.append(
            OpenRun(
                trace_id=span.context.trace_id,
                span_id=span.context.span_id,
                parent_id=parent.span_id if parent is not None else None,
                name=span.name,
                start_ns=span.start_time,
                attributes=dict(attributes),
            )
        )

    def on_end(self, span) -> None:
        pass

    def shutdown(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True
