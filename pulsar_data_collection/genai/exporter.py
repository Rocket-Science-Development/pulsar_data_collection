"""OpenTelemetry SpanExporter turning GenAI spans into Pulsar records written with Pulse."""

import time
from datetime import datetime, timezone
from typing import Optional, Protocol, Sequence

import pandas as pd
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import StatusCode

from ..models import CaptureOptions, DataWithPrediction, PulseParameters
from ..pulse import Pulse
from . import semconv as sc
from .trajectory import SpanView, TrajectoryAssembler

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
    )


class PulsarSpanExporter(SpanExporter):
    """OpenTelemetry exporter: GenAI spans -> agent runs, tool calls and LLM calls -> sink (InfluxDB via Pulse)

    Only features are kept: prompts, responses and tool arguments are not written (arguments are hashed to detect
    loops). Use with a BatchSpanProcessor so that the application thread never waits.

    Parameters
    ----------
    sink : RecordSink
        Usually PulseSink(params, options)
    allowed_tools : set of str, optional
        Tools the agent may call; others are counted as out-of-scope calls
    trace_timeout_s : float, optional
        Inactivity after which a trace whose root span is not received is assembled anyway (default 30 s)
    """

    def __init__(self, sink: RecordSink, allowed_tools: Optional[set] = None, trace_timeout_s: float = 30.0):
        self.sink = sink
        self.assembler = TrajectoryAssembler(allowed_tools=allowed_tools, trace_timeout_s=trace_timeout_s)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        now = time.time_ns()
        for span in spans:
            for record in self.assembler.add(span_view(span), now_ns=now):
                self.sink.write(*record)
        for record in self.assembler.flush_idle(now):
            self.sink.write(*record)
        return SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        self.sink.flush()
        return True

    def shutdown(self) -> None:
        for record in self.assembler.flush_all():
            self.sink.write(*record)
        self.sink.close()
