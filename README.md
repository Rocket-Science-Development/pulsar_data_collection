# pulsar-data-collection

[![Continuous Integration](https://github.com/Rocket-Science-Development/pulsar_data_collection/actions/workflows/continuous-integration.yaml/badge.svg)](https://github.com/Rocket-Science-Development/pulsar_data_collection/actions/workflows/continuous-integration.yaml)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14%20%7C%203.15-blue)
![pandas](https://img.shields.io/badge/pandas-2.2%2B%20%7C%203.x-blue)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**pulsar-data-collection** is a lightweight Python SDK that collects the data used by a machine learning model in
production (features, predictions, ground truth when available, and metadata) from an inference service (FastAPI,
Flask, batch job) or a notebook, and writes it to a time series database. The stored data is then analyzed with
[pulsar-metrics](https://github.com/Rocket-Science-Development/pulsar_metrics) to monitor data drift, concept drift and
model performance.

It also observes **LLM applications and agents** through OpenTelemetry GenAI traces: each agent run, tool call and
LLM call becomes a record of features (steps, tools, errors, loops, out-of-scope calls, tokens, latency), without
storing prompts or answers. Capture never blocks the service: rows are queued and written in the background, and kept
on disk while the database is unavailable.

![Pulsar.ml architecture](https://raw.githubusercontent.com/Rocket-Science-Development/pulsar_demo/main/images/pulsarml-architecture.png)

Supported storage: **InfluxDB 2.x** (tested with 2.6.1). The storage layer is a factory, so other engines can be added.

- [Installation](#installation)
- [Quick start](#quick-start)
- [Non-blocking capture and resilience](#non-blocking-capture-and-resilience)
- [How data is stored](#how-data-is-stored)
- [LLM applications and agents (OpenTelemetry GenAI)](#llm-applications-and-agents-opentelemetry-genai)
- [Personal information](#personal-information)
- [Analysis results for dashboards](#analysis-results-and-capture-statistics-for-dashboards)
- [API reference](#api-reference)
- [Analyzing the collected data with pulsar-metrics](#analyzing-the-collected-data-with-pulsar-metrics)
- [Development](#development)
- [Changelog](CHANGELOG.md)

## Installation

Requires **Python 3.12 or later** (tested on 3.12, 3.13, 3.14 and 3.15) and pandas 2.2+ or 3.x (Python 3.15 requires
pandas 3).

```bash
pip install "pulsar-data-collection @ git+https://github.com/Rocket-Science-Development/pulsar_data_collection.git"
```

Pin a commit or a tag for reproducible installs (`...pulsar_data_collection.git@<sha>`). The PyPI release (0.1.1)
predates the current API and only supports Python 3.9.

## Quick start

### 1. Start InfluxDB

```bash
docker run -d --name influxdb -p 8086:8086 \
  -e DOCKER_INFLUXDB_INIT_MODE=setup \
  -e DOCKER_INFLUXDB_INIT_USERNAME=admin \
  -e DOCKER_INFLUXDB_INIT_PASSWORD=change-me-please \
  -e DOCKER_INFLUXDB_INIT_ORG=pulsarml \
  -e DOCKER_INFLUXDB_INIT_BUCKET=monitoring \
  -e DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=change-me-token \
  influxdb:2.6.1
```

The bucket must exist before collecting: `Pulse(...)` checks it and raises `StorageConfigurationError` if the bucket
or the organization does not exist or if the token is rejected. If InfluxDB is merely unreachable, `Pulse` starts
anyway (see [resilience](#non-blocking-capture-and-resilience)). Keep the token out of the code: the example reads it
from environment variables.

### 2. Collect predictions

Create one `Pulse` when the service starts and call `capture_data()` for every prediction or batch of predictions.
`capture_data()` does not wait for the database: it queues the rows (under a millisecond) and a background thread
writes them.

```python
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from pulsar_data_collection.models import DataWithPrediction, PulseParameters
from pulsar_data_collection.pulse import Pulse

# A small model, standing in for the model served in production
rng = np.random.default_rng(0)
train = pd.DataFrame({"income": rng.normal(50, 15, 500), "age": rng.integers(18, 80, 500).astype(float)})
model = LogisticRegression().fit(train, (train["income"] > 50).astype(int))

params = PulseParameters(
    model_id="credit_model",
    model_version="1.0",
    data_id="production",
    reference_data_storage="s3://bucket/credit_model/reference.parquet",
    target_name="approved",
    storage_engine="influxdb",
    timestamp_column_name="_time",
    login={
        "url": os.environ.get("INFLUX_URL", "http://localhost:8086"),
        "token": os.environ["INFLUX_TOKEN"],
        "org": os.environ.get("INFLUX_ORG", "pulsarml"),
        "bucket_name": os.environ.get("INFLUX_BUCKET", "monitoring"),
    },
    additional_tags={"environment": "production", "region": "ca-central"},
)

with Pulse(data=params) as pulse:  # the connection is closed when leaving the block
    for _ in range(3):  # e.g. three incoming requests
        features = pd.DataFrame({"income": rng.normal(58, 15, 200), "age": rng.integers(18, 80, 200).astype(float)})
        predictions = pd.DataFrame(
            {
                "prediction": model.predict(features),
                "proba": model.predict_proba(features)[:, 1].round(4),
            }
        )
        pulse.capture_data(
            DataWithPrediction(
                data_points=features,
                predictions=predictions,
                timestamp=datetime.now(timezone.utc),
                features_names=features.columns.tolist(),
            )
        )
```

In a long-running service, keep the `Pulse` object for the lifetime of the process and call `pulse.close()` at
shutdown instead of using `with`: `close()` writes the rows still queued.

### 3. Read the data back

```python
from influxdb_client import InfluxDBClient

# stop: 1m keeps points written "now" even if the client clock is slightly ahead of the server's
query = f"""
from(bucket: "{params.login['bucket_name']}")
  |> range(start: -1h, stop: 1m)
  |> filter(fn: (r) => r["_measurement"] == "credit_model_1.0_input_data" and r["data_id"] == "production")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
"""
with InfluxDBClient(url=params.login["url"], token=params.login["token"], org=params.login["org"]) as client:
    collected = client.query_api().query_data_frame(query=query)

print(len(collected), "rows")
print(collected[["_time", "model_id", "model_version", "environment", "income", "age", "prediction", "proba"]].head(3))
```

## Non-blocking capture and resilience

Writing to the database must not slow down predictions. By default (`mode="background"`):

- `capture_data()` only prepares the rows and puts them in an in-memory queue, then returns (about 1 ms). It never
  raises because of the database.
- A background thread writes the queue in batches, every `flush_interval_s` seconds or as soon as `max_batch_rows`
  rows are waiting, with a request timeout of `write_timeout_s`.
- When a write fails, the database is considered unavailable for an exponential delay (1 s, 2 s, 4 s... up to
  `max_retry_delay_s`). Meanwhile batches go to the **spool directory** (`spool_dir`) instead of waiting, and are
  replayed, oldest first, as soon as the database is available again. Without `spool_dir`, these rows are dropped and
  counted. During the delay InfluxDB is probed (`/ping`) every `probe_interval_s` (1 s): writing resumes about a second
  after the end of an outage, not up to 60 s later.
- A write that **timed out** may have been stored by InfluxDB: with `spool_dir` it is spooled and written again
  (writes are idempotent: same measurement, tags and timestamp overwrite the same point, no duplicate); without it the
  rows are counted in `uncertain_rows`, not in `dropped_rows`.
- A batch InfluxDB **refuses** (HTTP 400 / 422, e.g. a field type conflict) is never retried nor spooled. Frames are
  written together only when they share their columns *and* dtypes, object columns mixing numbers and strings are split
  by kind, and the counters are exact: on a partial write (422 `dropped=N`) the accepted rows are counted in
  `written_rows` and the refused ones in `rejected_rows`; when the response gives no count, the request is written
  again batch by batch, then in halves down to single rows.
- The queue is bounded (`max_queue_rows`): if it is full, the oldest batches are dropped and counted rather than
  blocking the caller or exhausting memory. A single capture larger than `max_queue_rows` keeps only its newest rows.
- If InfluxDB is unreachable when `Pulse` is created, the service starts anyway (warning in the logs) and spools until
  the database is back. Set `strict_startup=True` to fail instead.

```python
from pathlib import Path

from pulsar_data_collection.models import CaptureOptions

options = CaptureOptions(
    spool_dir=Path("/var/lib/my-service/pulsar-spool"),  # keep rows on disk while InfluxDB is down
    flush_interval_s=1.0,
    max_queue_rows=100_000,
)
pulse = Pulse(data=params, options=options)

# ... in the request handler
pulse.capture_data(DataWithPrediction(data_points=features, predictions=predictions,
                                      timestamp=datetime.now(timezone.utc), features_names=["income", "age"]))

# completeness counters, e.g. exposed on a health or metrics endpoint
print(pulse.stats())

pulse.close()  # at shutdown: writes (or spools) what is still queued
```

`pulse.stats()` returns `captured_rows`, `written_rows`, `spooled_rows`, `replayed_rows`, `dropped_rows`,
`rejected_rows`, `uncertain_rows`, `write_errors`, `write_timeouts`, `last_error`, `queued_rows`, `writer_lag_s` (age
of the oldest queued batch), `last_write_ms` / `max_write_ms` (request latency), `spool_files`, `spool_bytes`,
`spool_evicted_rows`, `memory_only_rows` (`durability="wal"` batches kept in memory only because the spool was
full) and `storage_available`. Once everything is flushed, `written_rows + spooled_rows + dropped_rows +
rejected_rows + uncertain_rows = captured_rows`, and spooled rows end up in `replayed_rows` (or `rejected_rows`); any
`dropped_rows` (which includes `spool_evicted_rows`) is a measured gap in the monitoring data. Errors and drops are also
logged (logger `pulsar_data_collection`).

`pulse.flush(timeout_s)` waits until the queue is empty, for instance in a notebook before reading the data back.

Measured with InfluxDB 2.6.1 stopped for 15 s in the middle of the service: capture latency stayed at 0.8 ms (1.3 ms
maximum), the rows went to the spool and were replayed when InfluxDB restarted, with **no row lost** (3,000 / 3,000).

`mode="sync"` writes in `capture_data()` and raises on failure: use it for batch jobs, notebooks or tests that need
the data stored before going on, not in a latency-sensitive service.

### Many models in one process: the shared writer

By default each `Pulse` has its own writer thread and HTTP client. With tens or hundreds of models in one process, use
the **shared writer**: one thread and one HTTP client for every Pulse of the process, the rows of several models sent in
the same requests, the bucket checked once, up to `concurrency` requests in flight.

```python
from pulsar_data_collection.writer import SharedWriter

options = CaptureOptions(shared_writer=True)  # SharedWriter.default(): 5,000 rows per request, 2 in flight
pulses = {model: Pulse(params_for(model), options) for model in models}

# or an explicit writer, e.g. more requests in flight
writer = SharedWriter(max_batch_rows=5_000, flush_interval_s=1.0, concurrency=4)
pulse = Pulse(params, CaptureOptions(), writer=writer)
```

Each Pulse keeps its own queue bound, spool and counters; `max_batch_rows`, `flush_interval_s`, `max_retry_delay_s`
and `write_concurrency` of its options are replaced by those of the writer. `pulse.close()` detaches the Pulse;
`writer.close()` stops the thread.

### Durability modes

| `durability` | Rows lost by a crash of the process (`kill -9`) | Cost |
|---|---|---|
| `"memory"` (default) | the in-memory queue and the request in flight: about `flush_interval_s` of traffic at normal load, up to `max_queue_rows` + one batch when the writer is saturated (lab: 80,000 to 100,000 rows in a 50k rows/s burst) | none |
| `"wal"` (needs `spool_dir`) | none: each batch is written to `spool_dir` before `capture_data()` returns, deleted once stored, replayed at the next start | one file write per capture (no fsync: an OS crash or a power loss can still lose the last batches) |

At startup the spool also recovers what a crash left: complete `.tmp` files become spooled batches, incomplete ones are
deleted, write-ahead copies are replayed. A spool directory belongs to one Pulse of one process at a time. A spooled batch that can no longer be read or serialized is renamed `.bad`
and counted in `rejected_rows`.

### Throughput (measured)

Against InfluxDB 2.6.1 in Docker on the same machine, rows of 11 numeric fields, 500-row captures, 15 s per run:

| Setup | Written rows/s | Threads | Capture p99 |
|---|---|---|---|
| one Pulse, one request at a time (before, and `write_concurrency=1`) | 21,000 to 23,000 | 2 | 17 ms |
| one Pulse, `write_concurrency=4` | 37,000 | 2 | 50 ms |
| 100 Pulse, private writers (before) | 27,500 | 101 | 358 ms |
| 100 Pulse, one `SharedWriter(concurrency=4)` | 28,600 (40,000 rows/s offered, no loss) | 2 | 90 ms |

Profile of one 5,000-row request: about 60 ms to build the line protocol (influxdb-client serializer, pure Python,
under the GIL) and 125 ms of HTTP and ingestion, which `concurrency` overlaps. The ceiling of one process is therefore
the serialization, in the order of 40,000 to 80,000 rows/s depending on the columns; beyond, run several processes.

### `CaptureOptions`

| Field | Default | Description |
|---|---|---|
| `mode` | `"background"` | `"background"` (non-blocking) or `"sync"` |
| `max_queue_rows` | 100,000 | Rows kept in memory; beyond, the oldest batches are dropped and counted |
| `max_batch_rows` | 5,000 | Rows per write; a write starts as soon as this many rows are waiting |
| `flush_interval_s` | 1.0 | Maximum time a row waits before being written |
| `write_timeout_s` | 5.0 | Timeout of every request to InfluxDB |
| `max_retry_delay_s` | 60.0 | Upper bound of the delay between attempts while InfluxDB is unavailable |
| `spool_dir` | `None` | Directory for the batches that could not be written (JSON files, replayed then deleted) |
| `strict_startup` | `False` | Raise at creation if InfluxDB cannot be reached |
| `shared_writer` | `False` | Use the process-wide `SharedWriter` (see above) |
| `write_concurrency` | 1 | Requests in flight at once for a private writer |
| `probe_interval_s` | 1.0 | Probe of an unavailable InfluxDB (`/ping`); `None`: back-off only |
| `durability` | `"memory"` | `"wal"`: write-ahead copy of each batch in `spool_dir` (see above) |
| `spool_max_bytes` | `None` | Size limit of `spool_dir`; beyond, rows are dropped and counted (`spool_evicted_rows`) |
| `spool_eviction` | `"drop_oldest"` | Which rows go when the spool is full: `"drop_oldest"` or `"drop_newest"` |
| `spool_compression` | `None` | `"gzip"` (level 1): spooled batches 2.2 times smaller on random floats, more on repetitive values (the lab measured about 330 bytes per row of JSON: 1,880 MB for a 30 min outage at 200k rows/min) |
| `replay_max_rows_per_s` | `None` | Throttle of the spool replay. After a 30 min outage, an unthrottled replay raised the capture p99 from 1 ms to 144 ms |
| `row_spacing_ns` | 1,000 | Spacing of the timestamps of the rows of one capture (1 µs) |

## How data is stored

| InfluxDB concept | Content |
|---|---|
| Bucket | `login["bucket_name"]` (must exist) |
| Measurement | `{model_id}_{model_version}_input_data` |
| Tags | `model_id`, `model_version`, `data_id`, plus every key of `additional_tags` |
| Fields | every column of `data_points` and of `predictions` (numbers, booleans or strings) |
| Timestamp | row *i* of a capture is stored at `timestamp + i` microseconds (`row_spacing_ns`) |

Consequences:

- InfluxDB keeps a single point per measurement, tags and timestamp. Each `Pulse` therefore keeps its timestamps
  strictly increasing: a capture starting at or before the end of the previous one is shifted just after it. System
  clocks are coarse (15.6 ms on Windows) and concurrent requests share timestamps; without this, rows would silently
  overwrite each other. Several processes writing the same series should add a distinguishing tag (e.g. `instance`).
  Rows are 1 µs apart. They were 1 ms apart, so that a model writing more than 1,000 rows/s ran ahead of the clock
  (+2,760 s after 3 min at 1M rows/min) and was missed by queries stopping at `now()`; a model now stays on time up to
  1 million rows/s.
- To store the ground truth (when it is known at prediction time, or for a replay), add it as a column of
  `predictions`.
- A Flux `range()` stops at the server's `now()` by default. Points timestamped with the client clock can be a few
  milliseconds "in the future" for the server (clock skew, frequent with Docker on Windows or macOS) and are then missing
  from a query run right after the write. Use an explicit future stop, e.g. `range(start: -1h, stop: 1m)`.

## LLM applications and agents (OpenTelemetry GenAI)

`pulsar_data_collection.genai` (extra `genai`: `pip install "pulsar-data-collection[genai] @ git+..."`) collects
LLM applications and agents through **OpenTelemetry**, using the GenAI semantic conventions (`gen_ai.*`). Any span
carrying `gen_ai.operation.name` is read, whether it comes from an existing instrumentation or from the helpers below.
`PulsarSpanExporter` turns the spans into three record types written with Pulse (non-blocking):

| `record_type` | One row per | Main fields |
|---|---|---|
| `agent_run` | agent run (`invoke_agent` span and its subtree) | `steps` (LLM calls), `tool_calls`, `distinct_tools`, `tool_error_rate`, `tool_errors`, `tool_timeouts`, `tool_soft_failures`, `tool_<tool>_calls` / `_errors` / `_timeouts` / `_soft_failures`, `tool_status_counts`, `repeated_calls`, `repeated_calls_window`, `max_same_tool_calls`, `tool_cycle_repeats`, `loop_detected`, `unavailable_tool_calls`, `tool_catalog_hash`, `incomplete`, `out_of_scope_calls`, `delegation_depth`, `input_tokens`, `output_tokens`, `latency_s`, `termination`, `task_success`, `first_tool`, `tool_sequence`, `request_model`, `prompt_version`; with `compliance`: `violated_rules`, `critical_violations`, `rules_version` |
| `tool_call` | `execute_tool` span | `tool_name`, `error`, `status`, `timeout`, `soft_failure`, `latency_s`, `out_of_scope` |
| `llm_call` | `chat` span | `request_model`, `response_model`, `provider`, `temperature`, `input_tokens`, `output_tokens`, `finish_reason`, `latency_s`, `error` |

Only features are stored: prompts, responses and tool arguments are **not** written (arguments are hashed in memory to
detect loops). Records carry the tags of `PulseParameters` plus `record_type` and `schema` (`pulsar-genai/1`).

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from pulsar_data_collection.genai import PulsarSpanExporter, PulseSink, agent_span, llm_span, record_llm_response, tool_span

exporter = PulsarSpanExporter(
    PulseSink(params, CaptureOptions(spool_dir=Path("/var/lib/my-agent/spool"))),
    allowed_tools={"search_programs", "check_eligibility"},  # other tools are counted as out-of-scope calls
)
provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("permit-assistant")

with agent_span(tracer, "permit_assistant", prompt_version="v3") as run:
    with llm_span(tracer, model="llama3.2:3b", provider="ollama", temperature=0.0) as call:
        ...  # call the model
        record_llm_response(call, input_tokens=812, output_tokens=64, finish_reason="tool_calls")
    with tool_span(tracer, "search_programs", arguments={"topic": "renovation"}):
        ...  # run the tool; an exception marks the call as failed
    run.set_attribute("pulsar.agent.termination", "completed")
    run.set_attribute("pulsar.agent.task_success", True)

provider.shutdown()  # at exit: assembles pending traces and closes the Pulse instances
```

Notes:
- The GenAI conventions are at the "Development" stage: the attribute names read by Pulsar are pinned in
  `genai/semconv.py`, and aliases used by other SDKs (`llm.model_name`, `llm.token_count.prompt`...) are normalized.
- An agent run is assembled when its local root span ends: the trace's root, or a span whose parent lives in another
  process (HTTP server span). A trace is **never** assembled before its root arrives, however long the run takes.
  Sub-agents are part of the run (`delegation_depth`).
- Read a record type back with `|> filter(fn: (r) => r["record_type"] == "agent_run")`.

### Long, stuck and failing runs

```python
exporter = PulsarSpanExporter(
    PulseSink(params, options),
    allowed_tools={"search_programs", "check_eligibility", "compute_amount"},
    max_run_s=600,  # a run still open after 10 min is reported as incomplete (default)
)
provider.add_span_processor(BatchSpanProcessor(exporter))
provider.add_span_processor(exporter.run_tracker())  # sees agent spans start: a run that never ends is still reported

with agent_span(tracer, "permit_assistant", available_tools=[t["function"]["name"] for t in TOOL_SCHEMAS]):
    with tool_span(tracer, "check_eligibility", arguments=args) as span:
        output = check_eligibility(**args)
        record_tool_result(span, output, validator=lambda out: isinstance(out.get("eligible"), bool))

exporter.stats()  # {"partial_traces_dropped": 0, "partial_spans_dropped": 0, "incomplete_runs": 0, ...}
```

| Situation | What is written |
|---|---|
| Run longer than 30 s | The complete run, when its root ends. Former behaviour (`wait_for_root=False`): the trace was assembled after `trace_timeout_s` of inactivity, without its root, and the run lost its LLM and tool calls (`tool_calls=0`, `tool_sequence="none"`). |
| Run that never ends (model generating for minutes, agent stuck in a loop) | One `agent_run` with `termination="incomplete"`, `incomplete=True`, `latency_s` = time open so far and the calls received so far, after `max_run_s` (default 600 s); the complete record follows if the run ends. With `run_tracker()` the run is known from its start, even when no span ever ends; without it, only from ended spans that carry `gen_ai.agent.name` and are not below an agent span already received (never from LLM or tool calls alone). A hung run is reported once, then forgotten after `orphan_timeout_s`. |
| Trace whose root never arrives | Given up after `orphan_timeout_s` without new span (default 3600 s), or when more than `max_pending_traces` (10,000) traces wait; counted in `stats()["partial_traces_dropped"]` and `["partial_spans_dropped"]`; agent runs found inside are marked `incomplete=True`. Runs under a long-lived local root (a chat session span) wait for it: lower `orphan_timeout_s` (e.g. 300 s) for chat-style roots, or end the session span per turn. |
| Tool timeouts and errors | Per tool: `tool_<tool>_calls`, `tool_<tool>_errors`, `tool_<tool>_timeouts`, `tool_<tool>_soft_failures` (every tool of `allowed_tools` and of the catalogue is written, 0 when not called). Per run: `tool_errors`, `tool_timeouts`, `max_tool_error_rate`, `tool_status_counts` (`"ok=3,timeout=2"`). A failed call is a `timeout` when its `error.type` names one (`TimeoutError`, `ReadTimeout`...). `tool_call` records carry `status`, `timeout` and `soft_failure`. |
| Tool that answers but returns nothing usable | `record_tool_result(span, output, status=None, valid=None, validator=None)` sets `pulsar.tool.status` (by default the `status` key of a dict output) and `pulsar.tool.output_valid`. The statuses `retry`, `invalid`, `empty`, `partial`, `garbage` and invalid outputs are soft failures: `tool_soft_failures`, `soft_failure_rate`, `tool_<tool>_soft_failures`. The output itself is never written. |
| Loops | `repeated_calls` keeps its meaning (consecutive calls identical byte for byte). `repeated_calls_window`: calls whose **normalized** tool and arguments (JSON parsed, keys sorted, `36000` = `36000.0` = `"36000"`, strings trimmed and lower-cased) appeared among the `loop_window` (4) previous calls; `loop_detected` is True from `loop_min_repeats` (2) such calls, so ping-pong A>B>A>B and type variations are loops. `max_same_tool_calls` and `tool_cycle_repeats` (longest repetition of a block of 2 or 3 tools) are counts, not verdicts: the same tool called with other identifiers can be legitimate, test them against a reference. |
| Removed or added tool | With `agent_span(..., available_tools=[...])` (`pulsar.agent.available_tools`): `tool_catalog_size` and `tool_catalog_hash`, which changes with the catalogue from the first run; with `allowed_tools`, `catalog_missing_tools` and `missing_tools`. `unavailable_tool_calls`: calls to a tool outside the catalogue, or with status `unavailable`, `unknown_tool` or `not_found`. |

### Rule violations in `agent_run`

`PulsarSpanExporter(..., compliance=monitor, rules_version="2026-11")` checks every run when it is assembled and adds
`violated_rules` (rule ids, comma-separated, `"none"`), `violations`, `critical_violations` (rules with
`severity="critical"`) and `rules_version` (default: `monitor.version`). `compliance` is a
`pulsar_metrics.genai.ComplianceMonitor` (pulsar-metrics is not a dependency: any object with `violations(trace)`
fits) or a callable `trace -> violated rule ids`; the trace is the list of the tools called, as
`trace_from(tool_sequence)` rebuilds it. A rule set that raises never stops the collection
(`stats()["compliance_errors"]`). Without `compliance`, the fields are absent.

### Known limits (night lab, 2026-11-02)

- `loop_detected` changed meaning (non-consecutive window, normalized arguments): it now flags the ping-pong and type
  variations that the lab found invisible (ping-pong x 6 and `36000` / `36000.0` / `"36000"` x 4 gave
  `loop_detected=False`). Its rate on unchanged runs has not been re-measured (0 loops on 1196 A/A runs with the former
  definition).
- A tool returning junk with status `retry` is visible only if the application records it (`record_tool_result` or
  `pulsar.tool.status`): Pulsar never reads tool outputs. In the lab, 100 % of the 56 eligibility runs of the "garbage"
  mode had `tool_error_rate` 0.
- Per-tool counters are features, not alerts: the per-tool test (Fisher or binomial against a reference; p = 0.0013
  from 10 % of check_eligibility calls in timeout, 2/8 against 0/199) belongs to pulsar-metrics. The run-level
  `tool_error_rate` (0.10 to 0.27 against 0.074) was never flagged in windows of 30 runs.
- A removed tool is a configuration change only when the catalogue is recorded (`available_tools`); otherwise it is
  seen through `first_tool` and `task_success`, from about half of the runs affected.
- The exporter only receives finished spans: without `run_tracker()`, a run blocked in its first model call (about
  37,000 tokens over 9 minutes in the lab) stays invisible until it ends. Also cap `max_tokens` and set a client
  timeout without retry in the agent.

### Counterfactual probes

`CounterfactualProbe` is an **active** collector: it builds requests that differ only by one attribute (e.g. a first
name typical of each group), sends them to the application and records what it does. An audit of a language model
can produce its own verdict (order effects, wording, randomness), so the controls are built in:

- **placebo sets** vary an attribute that must not matter; their flip rate is the noise floor of the instrument;
- `ComparativeProbe` asks to choose between two profiles **in both orders**, with or without a tie option, to measure
  and cancel the position effect;
- requests are sent in a random order and look like ordinary requests.

`probe.span_attributes()` gives the `pulsar.cf.set_id`, `pulsar.cf.variant`, `pulsar.cf.attribute` and
`pulsar.cf.control` attributes: set them on the agent span and they reach the `agent_run` records (`cf_set_id`,
`cf_variant`...). The rows returned by `run` are analysed with `pulsar_metrics.fairness.counterfactual_analysis` and
`comparative_analysis`.

The value lists are the user's responsibility: lists of names standing for groups are sensitive and must be reviewed
(e.g. by a GBA Plus expert) before an audit.

```python
from pulsar_data_collection.genai import CounterfactualProbe

probe = CounterfactualProbe(
    templates=["Bonjour, je m'appelle {name} (dossier {reference}). Je gagne {income} $. Suis-je admissible ?"],
    attribute="name",
    values={"fr_qc": ["Gabrielle Tremblay", "Mathieu Gagnon"], "en_ca": ["Emily Thompson", "Ryan MacDonald"]},
    placebo=("reference", {"ref_a": ["A-1042"], "ref_b": ["B-7731"]}),  # must not change anything
    contexts=[{"income": 24000}, {"income": 47000}],
)
probes = probe.generate(n_sets=2)
print(probes[0].text)
print(probes[0].span_attributes())


def my_agent(text: str, span_attributes: dict) -> list:  # stands for the instrumented application
    return ["check_eligibility"]


rows = probe.run(probes, target=lambda p: my_agent(p.text, p.span_attributes()), outcome=lambda tools: tools[0])
print(rows[["set_id", "variant", "control", "outcome"]].head(4).to_string(index=False))
```

```
Bonjour, je m'appelle Mathieu Gagnon (dossier A-1042). Je gagne 47000 $. Suis-je admissible ?
{'pulsar.cf.set_id': 'cf-00000', 'pulsar.cf.variant': 'fr_qc', 'pulsar.cf.attribute': 'name', 'pulsar.cf.control': 'none'}
       set_id variant control           outcome
     cf-00001   en_ca    none check_eligibility
placebo-00003   ref_a placebo check_eligibility
     cf-00000   en_ca    none check_eligibility
placebo-00002   ref_b placebo check_eligibility
```

### Detectors and canary set

`pulsar_data_collection.genai.detectors` scores each run in-process; only the scores reach the spans
(`pulsar.detector.<name>`) and the records (`det_<name>`, maximum over the run). Local open models, installed with the
`detectors` extra (`pip install "pulsar-data-collection[detectors]"`):

| Detector | Score | Default model |
|---|---|---|
| `InjectionDetector` | score that a text is a prompt injection or a jailbreak | protectai/deberta-v3-base-prompt-injection-v2 (Apache 2.0) |
| `ModelPiiDetector` | rule-based findings plus names and street addresses | urchade/gliner_multi_pii-v1 (Apache 2.0) |
| `FaithfulnessScorer` | share of the answer entailed by the retrieved passages (multilingual NLI) | MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 (MIT) |
| `ContentSafetyDetector` | harmful content, from a classifier of your choice | none: pass `model` and its unsafe `positive_labels` |

Any classifier with a `score()` method fits (e.g. Prompt Guard 2, under the Llama licence); `record_score(span, name,
value)` writes it, and `TextClassifierDetector(model, name=..., positive_labels=[...])` wraps any Hugging Face text
classifier. The scores are **ranks, not calibrated probabilities**: alert on a threshold fitted on a benign reference
of the deployment (A/A), not on 0.5.

```python
from pulsar_data_collection.genai.detectors import (
    FaithfulnessScorer, InjectionDetector, ModelPiiDetector, ScoreCalibrator, calibrate_threshold,
    french_benign_calibration_set,
)

injection = InjectionDetector(from_logits=True, device="auto")  # double-precision scores, see below
fit = injection.calibrate(french_benign_calibration_set(), language="fr", target_fpr=0.01, positives=known_fr_attacks)
print(fit.threshold, fit.fit_fpr, fit.recall)  # the French threshold, and the French recall it costs
injection.record(span, text)                   # pulsar.detector.injection and pulsar.detector.injection_flag

details = injection.score_details(long_text)  # score, n_tokens, tokens_seen, windows, truncated
sliding = InjectionDetector(long_text="sliding")  # every 510-token window of long texts, maximum score

pii = ModelPiiDetector(device="auto", thresholds={"street address": 0.3})  # windows of 384 GLiNER tokens, overlap 96
found = pii.scan(text)                        # a list of findings, with found.windows and found.truncated_windows
many = pii.scan_many(texts)                   # short texts batched across requests, long ones one by one
scorer = FaithfulnessScorer(max_sentences=None)
result = scorer.record(span, answer, passages)  # faithfulness, _min, _coverage and _windows on the span
threshold = calibrate_threshold(reference_scores, target_fpr=0.05, direction="low").threshold
calibrated = ScoreCalibrator("isotonic").fit(scores, labels)  # with labelled data of the deployment
```

Parameters:

- `InjectionDetector` and `TextClassifierDetector`: `model`, `device` (None = CPU, `"auto"`, `"cuda:0"`...),
  `threshold` (default 0.5), `thresholds` (per language, e.g. `{"fr": ...}`), `language` (callable, default
  `guess_language`, a French/English heuristic), `calibrator` (applied to the raw score), `from_logits` (default False).
  `calibrate(benign, language, target_fpr, positives)` fits a threshold so that at most `target_fpr` of the benign
  texts raise the alert, and returns a `ThresholdCalibration` (`threshold`, `fit_fpr`, `recall` on the given
  positives, `sufficient`: False when fewer than 1/target_fpr - 1 benign texts are given). `flag(text, language=None)`
  and `record(span, text)` apply the threshold of the text's language; `score_many(texts)` scores in batches.
  `french_benign_calibration_set()` gives 154 synthetic benign French requests (5 kinds, imperatives such as
  "Ouvrez-moi un dossier" or "Ignorez les détails..."): a starting point, to be replaced by the deployment's traffic.
- `from_logits=True` computes the score from the logits in double precision. The pipeline's single-precision scores
  saturate at 0.99999976: benign French imperatives and real attacks then tie, and no threshold separates them
  (`calibrate` warns). On the default model, a French threshold fitted on half of the starter set kept 1.3 % false
  alarms on the other half and caught 3 of 6 hand-written French attacks with `from_logits=True`, none without.
- **The 512-token horizon.** `InjectionDetector` reads `max_length=512` tokens, 510 of text. With
  `long_text="truncate"` (default) the rest of a longer text is not scored: an injection appended to a 2000-token
  text was flagged in 10 % of the cases (SC-09), as often as the benign texts. The detector warns once when it cuts a
  text, and `score_details(text)` returns a `ClassifierScore` (`score`, `raw`, `n_tokens`, `tokens_seen`, `windows`,
  `truncated`, `window_scores`); `score(text)` still returns a float. `long_text="sliding"` scores windows of 510
  tokens overlapping by `overlap_tokens` (default 128) and keeps the maximum: the payload is seen wherever it is, but
  a long benign text has one chance to score high per window, so **fit the thresholds again in sliding mode**
  (`calibrate` on long benign texts of the deployment). On 40 synthetic benign administrative texts (English and
  French) per length, the injection appended at the end was flagged 0/40 in truncate mode at 500, 1000 and 2000
  tokens and 40/40 in sliding mode, with no benign text above 0.5 in either mode (maximum 0.011); these texts are
  cleaner than real traffic, where the lab saw 5 to 20 % benign flags at 0.5 on the first window alone. Sliding costs
  one model call per window: 5 windows, about 3.9x the time, at 2000 tokens on CPU. The default stays "truncate".
- **dtype.** Every detector takes a `dtype` (`InjectionDetector` and `TextClassifierDetector` pass any keyword to
  `transformers.pipeline`, e.g. `InjectionDetector(device="cuda", dtype=torch.float16)`: 2.48x faster on a T4,
  scores within 0.0043, 1 decision in 64 flipped at 0.5, SC-09). Without it, models run in float32 on the CPU and in
  the dtype of their checkpoint on a GPU. The default NLI checkpoint is stored in float16, and ran so on the CPU,
  6 to 14x slower than in float32 (SC-09: 4.8 s against 0.67 s per 50-token request): it now runs in float32 there
  (here, one answer against a 500-token context: 17.1 s in float16, 1.5 s in float32, same score to 1e-4).
- `ModelPiiDetector`: `device`, `thresholds` (per GLiNER label: `"person"`, `"street address"`, `"date of birth"`),
  `window_tokens` and `overlap_tokens` (default 384, the model's `max_len`, and 96): GLiNER reads at most 384 of its
  own tokens (words and punctuation marks apart, about 1.9 per word) and drops the rest, so the windows are counted
  with the model's own words splitter. Up to 0.10 they were 250 *words* (`window_words=250, overlap_words=50`, still
  accepted): a window often held more than 384 tokens and its end was never read (a name appended to a 1000-token
  text was found in 65 % of the cases, SC-09). `scan(text)` returns a `PiiScan`, a list of findings with `windows`
  and `truncated_windows` (windows longer than `max_len`, 0 with token windows; the detector warns once). `merge`
  (`"enclosing"`, the default: an address that holds a rule-found postal code or phone number replaces them, and is
  merged with overlapping rule-found addresses; `"rules"`: the model entity is dropped, as before 0.10), `locales`
  and `profiles` (passed to the default `PiiDetector`), `batch_size`, `dtype`. `scan(text, business_scope=...)`
  forwards `business_scope` to the rules detector when it accepts it. `scan_many(texts, short_tokens=200)` sends the
  texts of at most `short_tokens` GLiNER tokens through the model together and the longer ones one by one: batching
  across requests was 3.3x faster at 50 tokens, 1.1x at 200 and slower from 500 (SC-09, T4); on CPU, 64 requests of
  about 30 tokens went 2.3x faster through `scan_many` than through `scan`, with the same findings. On 20 synthetic
  invoice listings of 330 to 700 words (punctuation-heavy) ending with a name, the name was found in 13/20 with token
  windows and 4/20 with 250-word windows, 40 of whose 45 windows were truncated.
- `FaithfulnessScorer`: `window_tokens` and `stride_tokens` (default 400 and 300: the joined passages are cut into
  overlapping windows and each sentence keeps its best window; `None` restores the single 512-token premise, silently
  truncated), `max_sentences` (default 6, `None` = all), `lexical_threshold` (default 1.0: a sentence whose word
  trigrams are all in the passages, i.e. a verbatim copy, is supported without the model; `None` disables),
  `max_claim_words` (splits long sentences at commas and semicolons, default off), `aggregate` (`"mean"` or `"min"`),
  `dtype` and other `transformers.pipeline` keywords.
  `details(answer, passages)` returns a `FaithfulnessResult`: `score`, `min`, `sentences` (support per sentence),
  `lexical`, `sentences_total`, `sentences_scored`, `coverage`, `premise_tokens`, `windows`, `truncated`.
- **Premises cut at sentence boundaries.** In a token window, the support of a supported sentence falls with the
  unrelated text around its evidence: median 0.32 when the evidence opens a 500-token context (a full 400-token
  window) against 0.62 when it closes it (a short last window), 0.40 against 0.93 at 1000 tokens (SC-09).
  `premise_split="sentences"` cuts the passages into groups of `premise_sentences` consecutive sentences (default 2,
  one group per starting sentence) and judges each answer sentence against the `top_k_premises` groups (default 4;
  `None` = all) that share the most words with it, so the support no longer depends on where the evidence falls
  (here, 10 synthetic contexts per length, a supported claim: median support start / end 0.04 / 0.87 at 500 tokens,
  0.11 / 0.99 at 1000 and 0.30 / 0.82 at 2000 with token windows, 0.997 / 0.998 at every length with sentence
  groups; the claim without its evidence stayed at a median of 0.02-0.03, and sentence groups took 0.4-0.5 s per
  answer against 1.5-6.5 s on CPU). The default stays `"tokens"`; fit the alert threshold again before switching.

`CanaryProbe` asks a fixed set of prompts several times and saves the answers (they are not user data) for
`pulsar_metrics.genai.model_equality`. Since 0.10 the default is 48 prompts x 10 answers (480 calls per run;
`DEFAULT_PROMPTS`, whose first 12 are the former set). Presets: `CanaryProbe.preset("quick")` (12 x 5), `"standard"`
(48 x 10) and `"thorough"` (48 x 20). A canary smaller than `required_answers("quantisation")` warns.

| Change of the served model | Canary answers needed (lab) |
|---|---|
| other weights, other model family | about 60 (`required_answers("weights")`) |
| same weights, other quantisation | about 480 (`required_answers("quantisation")`) |

#### Known limits of the detectors (night lab, 2026-11-02)

- **Injection recall (LI-01).** At 0.5, recall 0.414 [0.357-0.475] on deepset/prompt-injections (n = 662, 263
  positives; English subset 0.345); 52 % of its positives score under 0.01. A threshold cannot recover attacks that
  the model scores near 0: change `model`, or combine detectors.
- **Benign prompts flagged (LI-01).** False-alarm rate 0.0935 [0.089-0.098] at 0.5 on 16 989 "regular" in-the-wild
  prompts (TrustAIRLab; English 0.077, other languages 0.144), whose labels are noisy. Fit the threshold on the
  deployment's benign traffic, and report the benign false-alarm rate with the alert.
- **French imperatives (LI-02, LS-01).** Benign French agent requests flagged 151/1000 at 0.5, all of the "Ouvrez-moi
  un dossier" kind (151/253, 0.60), against 0/256 in English; an absolute `injection > 0.5` alert fired in 10 A/A
  windows out of 10. A French threshold fitted on a benign A/A reference (0.999994) brings the French false-alarm rate
  down to 0.0105, but **French recall falls from 0.759 to 0.567**: a per-language threshold trades recall for quiet.
- **Not calibrated (LI-01, LR-01).** Expected calibration error 0.237 (injection, deepset), 0.100 (injection,
  TrustAIRLab), 0.209 (faithfulness, RAGTruth). On TrustAIRLab, injection scores in 0.9-1.0 held 45 % positives; on
  RAGTruth, answers with a faithfulness score in 0.0-0.1 were faithful in 65 % of the cases. Use `ScoreCalibrator` with
  labelled data, or `calibrate_threshold`.
- **No content safety (LI-01).** Harmful requests without an injection wrapper (JailbreakBench) were flagged 0/100
  [0-0.037] by `InjectionDetector`: `det_injection` does not cover harmful content. `ContentSafetyDetector` has no
  default model, since none has been evaluated here for French and English.
- **Addresses (LP-01).** Recall of rules + GLiNER at 0.5: 0.496 (English) and 0.639 (French addresses, France format).
  Part of it was lost to the former merge rule (an address holding a postal code was dropped; it is kept with
  `merge="enclosing"`). No threshold sweep was run: lower the `"street address"` threshold after measuring on your
  own documents.
- **Long documents (LP-01).** Before windows, names and addresses beyond GLiNER's 384-token window were never found
  (recall 0.0, against 0.775 and 0.833 inside it); the windows are sized in GLiNER tokens since SC-09. Windows without any person can produce generic false names ("je",
  "le client"); a higher `"person"` threshold reduces them. GLiNER on CPU: p50 2.6 s per document, 0.14 s on a T4.
- **Faithfulness ranking (LR-01).** Answer-level AUROC 0.616 on RAGTruth (n = 2700, 34.9 % hallucinated; QA 0.526,
  Data2txt 0.552, Summary 0.599), below the number of sentences alone (0.645). Premise windows raised it to 0.689, and
  the mean score of faithful answers from 0.485 to 0.669, on 400 answers (0.19 s per answer). The weakest-sentence
  aggregate reached 0.675 over all sentences, still to be checked against a length-matched reference. Alert on window
  means relative to an A/A reference rather than on an absolute level, and plug a RAG-specific checker through the
  same `score(answer, context)` interface if needed.
- **Sentence cap (LR-01).** 41.6 % of the RAGTruth answers have more than 6 sentences, and 15.6 % of the hallucinated
  spans come after the 6th: use `max_sentences=None` (the cost grows with sentences x windows) and watch `coverage`.
- **Length (LR-02).** Reference answers copied from the passage scored 0.292 on average (0.237 above 25 words): long
  hypotheses are rarely entailed. The lexical shortcut fixes verbatim copies only. Paraphrases keep a level that falls
  with length, so an alert on a window mean drifts when the length of the answers changes.
- **Canary sample size (LM-01).** 12 x 5 detected a q4_K_M to q8_0 change in 20 % of the runs (llama3.2 3B; 40 % for
  llama3.1 8B), 48 x 5 in 60 %, 48 x 10 in 5 runs out of 5; q4_K_M to fp16 at 12 x 10: 0 %. The repetitions came from
  few, correlated batches (the lower 95 % bound of 5/5 is near 0.48), and 48 x 10 was never run against fp16: check
  on your own model.

## Personal information

`pulsar_data_collection.privacy` detects personal information with **validated rules** (checksums, valid dates,
issuer prefixes, context words) for Canada, in French and English, and **masks it before anything is written**.
Other countries and financial identifiers are recognized on request. Findings carry the type, the position and a
confidence, never the value.

| Type | Label | Validation |
|---|---|---|
| `sin` (social insurance number) | `[SIN]` | 9 digits, Luhn checksum, first digit issued |
| `ramq` (Québec health insurance number) | `[RAMQ]` | 4 letters + 8 digits (spaces, dashes or none; any case), valid month (women: +50) and day |
| `credit_card` | `[CREDIT_CARD]` | 13–19 digits, Luhn, issuer prefix |
| `email`, `phone` (North American), `postal_code` (Canada) | `[EMAIL]`, `[PHONE]`, `[POSTAL_CODE]` | format |
| `drivers_licence_qc`, `passport` (Canada) | `[DRIVERS_LICENCE]`, `[PASSPORT]` | format; a passport needs a context word |
| `date_of_birth` | `[DATE_OF_BIRTH]` | valid date **and** a context word ("né le", "date of birth"...) |

Opt-in, with `PiiDetector(locales=..., profiles=...)` or `PiiPolicy(locales=[...], profiles=[...])`:

| Option | Types | Validation |
|---|---|---|
| `locales=("ca", "us")` | `ssn_us` (`[SSN]`) | 3-2-4 digits, never-issued ranges rejected; 9 plain digits only after "SSN" |
| `locales=("ca", "fr")` | `phone` (French), `nir_fr` (`[NIR]`, sécurité sociale) | format; NIR key (97 − number mod 97), Corsica 2A/2B |
| `locales=("ca", "uk")` | `phone` (UK), `nino_uk` (`[NINO]`, National Insurance) | format, excluded prefixes |
| `locales=("ca", "intl")` | `phone` (E.164, "+33 1 23 45 67 89"), `passport`, `drivers_licence` (`[DRIVERS_LICENCE]`) | 8–15 digits; passports and licences of any country only after a context word |
| `profiles=("finance",)` | `iban`, `bic`, `bank_account`, `bank_routing`, `card_security_code` (`[IBAN]`, `[BIC]`, `[BANK_ACCOUNT]`, `[BANK_ROUTING]`, `[CVV]`) | IBAN mod 97; BIC country code and a context word ("SWIFT", "BIC"); account, transit, institution and routing numbers after a context word ("numéro de compte", "transit", "routing"...), ABA checksum (with `us`), Canadian cheque order transit-institution-account; a CVV only right after its name |
| `profiles=("address",)` | `street_address` (`[ADDRESS]`) | civic number and street type: "1234, boul. Saint-Laurent Ouest, app. 3", "12 rue de la Paix", "55 Maple Street", "100 King St W, Suite 200" |

`TYPES` lists the types on by default, `ALL_TYPES` every type; a type named in `types=[...]` is on whatever the
locales and profiles.

- **Context**: a word such as "NAS", "téléphone" or "passport" just before raises the confidence; a word announcing a
  business number ("dossier", "commande", "order", "invoice"...) lowers it below the threshold, and so does a currency
  right after a number ("514 555 0199 $" is an amount). Some types also have words of their own, which count
  only right before the value (possibly followed by "no", "n°", "#" or ":"): "client", "customer", "membre",
  "confirmation", "réservation"... before a number shaped like a SIN ("No client 130 692 544"), and "membre",
  "confirmation"... before a phone number ("Numéro de membre 514-555-0123"); a SIN is never written after "#".
  "Customer Jean Tremblay 130 692 544" is still a SIN.
  Findings below `min_confidence` (0.5) are ignored. A RAMQ number in lower case gets 0.55 instead of 0.9.
- **Tabular models**: `CaptureOptions(pii=PiiPolicy())` masks the text columns in `capture_data`, in the caller's
  thread, so raw values never reach the queue, the spool directory or the database; a `pii_count` column counts the
  findings per row. `PiiPolicy(action="flag")` only counts.
- **LLM applications and agents**: `PiiSession` scans the text of each role and adds counts and types to the spans:
  `input` (from the user), `context` (retrieved documents, tool results) and `output` (answers, tool calls). A
  **leak** is personal information in an output that the user did not provide in the same session. The
  `agent_run`, `llm_call` and `tool_call` records gain `pii_<role>_count` and `pii_<role>_types` (`input`, `context`,
  `output`, `leak`), only when a session recorded something. The values given by the user stay in memory for the
  session only. In outputs, a business word lowers a finding only when it is **right before the value**
  (`output_business_scope="adjacent"`, default): "Votre dossier 130 692 544 est en traitement" is not a leak, "Le
  dossier voisin indique aussi 130 692 544" is. `output_business_scope="window"` restores the 40-character window,
  which hid every leak of an agent that mentions files. `PiiDetector.scan(text, business_scope="adjacent")` gives the
  same rule outside a session.
- Measured on a synthetic French and English corpus (fictitious values with valid checksums, plus business numbers
  that must not be detected): precision and recall of 1.00 for every type. The corpus is ours: real text will have
  misses and false positives, and names are not covered by rules.

```python
import pandas as pd

from pulsar_data_collection.privacy import PiiDetector, PiiPolicy, PiiSession

detector = PiiDetector()
text = "Mon NAS est 130 692 544, écrivez à lea@exemple.ca. Dossier 450123456."
print([(f.type, f.start, f.end, f.confidence) for f in detector.scan(text)])  # positions, never the values
print(detector.mask(text))

# Tabular models: masked before the rows are queued, spooled or written
policy = PiiPolicy()  # CaptureOptions(pii=PiiPolicy()) applies it in Pulse.capture_data
print(policy.apply(pd.DataFrame({"comment": ["Joignez-moi au 514-555-0199", "RAS"], "income": [32000, 41000]})))

# Agents: counts and leaks by role, on the spans
with PiiSession() as session:
    session.scan("input", "Je suis lea@exemple.ca")
    print(session.scan("output", "Je réponds à lea@exemple.ca. Votre voisin : 418 555 0142.").leaks)
```

```
[('sin', 12, 23, 0.99), ('email', 35, 49, 0.95)]
Mon NAS est [SIN], écrivez à [EMAIL]. Dossier 450123456.
                  comment  income  pii_count
0  Joignez-moi au [PHONE]   32000          1
1                     RAS   41000          0
[Finding(type='phone', start=44, end=56, confidence=0.6)]
```

### Known limits

Measured by the night lab of 2026-11-02 (LP-01: 800 synthetic financial documents per language, from
gretelai/synthetic_pii_finance_multilingual; LP-02: 100,000 generated sentences with 30 % hard negatives, 5,000
scripted conversations).

- **LP-02, after the fixes** (same generator and seed, rerun on the new rules): precision and recall 1.00 for `sin`,
  `ramq`, `postal_code` and `phone` (were 0.811, 0.803, 1.00 and 0.896); leak counts exact in 5,000 of 5,000
  conversations (5,006 of 5,006 leaks, were 2,381; with "dossier" in the output, 2,493 of 2,493, were 0). About
  38,000 sentences per second on one core of a workstation (14,000 with the previous rules on the same machine).
  The generator's hard negatives are all Luhn-valid (about 10 % of random 9-digit identifiers are), so real precision
  differs; a Luhn-valid number with no context word at all is still reported as a SIN at 0.6.
- **LP-01, rules only, before the fixes**: phone recall 0.325 (EN) and 0.029 (FR), passport, US SSN and driver's
  licence 0.0, credit card 0.667 (EN) and 0.318 (FR), financial identifiers (IBAN, BBAN, SWIFT/BIC, routing numbers)
  0 to 0.24, street addresses 0.0 (0.496 EN and 0.639 FR with `ModelPiiDetector`). The new locales, the "finance"
  profile and the "address" profile target these gaps but are off by default and were not re-measured on that corpus;
  some of its synthetic phone and card numbers are invalid, and rejecting them is correct. Customer and employee
  identifiers, IP addresses, PINs and passwords have no type.
- **Addresses**: the rules need a civic number and a street type ("rue", "avenue", "boul.", "Street", "Ave"...) and a
  capitalized street name; an address without a number, in lower case, or with an unusual street type is left to the
  model (`ModelPiiDetector`). Postal codes and cities are separate findings.
- **Customer numbers**: "Customer 130-692-544" (the word right before, no name in between) is read as a customer
  number and not reported; a SIN written that way is missed.

## Analysis results and capture statistics for dashboards

The analyses of pulsar-metrics return `MetricResults` in memory. `ResultsSink` writes them to InfluxDB, in one
measurement (`pulsar_results`, one point per result), so that dashboards can show verdicts with their intervals;
`StatsReporter` publishes the counters of Pulse instances (`pulsar_capture_stats`) every few seconds.

| Tags | Fields |
|---|---|
| `metric_type`, `metric_name`, `feature_name`, `group`, `reference_group`, `test_type`, plus the caller's (`model_id`, `campaign`...) | `metric_value`, `ci_low`, `ci_high`, `threshold_low` / `threshold_high`, `n`, `e_value`, `status` (1 = drift or disparity established, 0 = none, -1 = inconclusive), `suppressed`, `method`, extra numeric fields such as `p_adjusted` or `confirmed` |

```python
from pulsar_data_collection.results import ResultsSink, StatsReporter

with ResultsSink(url, token, org, bucket) as sink:
    sink.write_results(analyzer.get_result(), tags={"model_id": "credit_model", "model_version": "1.0"})
    with StatsReporter(pulse, sink, interval_s=10):  # a failed report never disturbs the capture
        ...  # serve predictions
```

One reporter can serve the whole process (one thread, one request per interval for every Pulse), with the optional
process health in its own measurement, `pulsar_process_health` (tags `host`, `pid`): `threads`, and with
[psutil](https://pypi.org/project/psutil/) installed `rss_bytes`, `os_threads`, `open_fds` (`open_handles` on
Windows) and `cpu_percent`, plus the writer gauges of the registered Pulse instances: `queued_rows`, `writer_lag_s`,
`max_write_ms`.

```python
reporter = StatsReporter(None, sink, interval_s=10, tags={"environment": "prod"}, process_health=True)
for pulse in pulses.values():
    reporter.add(pulse)
reporter.start()
```

The optional fields of `MetricResults` are written only when present: `n_labelled`, `label_coverage`,
`missing_share_current` and `missing_share_reference` always as floats (stable field types from one write to the
next), `reliable` as a boolean, and the string `status` (e.g. `constant_reference`) as the field `result_status`, so
that it never collides with the numeric verdict `status`.

The Grafana dashboards built on these measurements are in
[pulsar_demo/observability](https://github.com/Rocket-Science-Development/pulsar_demo/tree/main/observability).

## API reference

### `PulseParameters`

| Field | Type | Description |
|---|---|---|
| `model_id` | `str` | Model identifier (tag) |
| `model_version` | `str` | Model version (tag) |
| `data_id` | `str` | Identifier of the data stream, e.g. `production`, `shadow`, `replay` (tag) |
| `reference_data_storage` | any | Location of the reference data used to train or validate the model (metadata) |
| `target_name` | `str` | Name of the target variable |
| `storage_engine` | `str` | `"influxdb"` |
| `login` | `dict` | For InfluxDB: `url`, `token`, `org`, `bucket_name` |
| `additional_tags` | `dict`, optional | Extra tags written with every row (environment, region...) |
| `features_metadata` | `dict`, optional | Feature name to type |
| `timestamp_column_name` | `str` | Name of the timestamp column; use `"_time"` for InfluxDB |

Unknown fields are rejected with a pydantic `ValidationError`, so that a misspelt field is not silently lost (for
example `other_labels` instead of `additional_tags`). The same applies to `DataWithPrediction`.

### `DataWithPrediction`

| Field | Type | Description |
|---|---|---|
| `data_points` | `pandas.DataFrame` | Features used for the predictions, one row per prediction |
| `predictions` | `pandas.DataFrame` | Predictions (labels, scores, probabilities...), same number of rows |
| `timestamp` | `datetime` or ISO string | Time of the first prediction |
| `features_names` | `list[str]` | Names of the features |
| `prediction_id` | `str`, optional | Identifier of the prediction request |

### `Pulse`

| Method | Description |
|---|---|
| `Pulse(data: PulseParameters, options: CaptureOptions = None)` | Checks the configuration and opens the connection. Raises `StorageConfigurationError` for an unknown bucket or organization, or a rejected token; `StorageUnavailableError` if InfluxDB is unreachable, only in sync mode or with `strict_startup=True` (both in `pulsar_data_collection.exceptions`) |
| `capture_data(data: DataWithPrediction)` | Background mode: queues the rows and returns; sync mode: writes and raises on failure. The caller's DataFrames are not modified |
| `flush(timeout_s=30)` | Waits until the queued rows are written, spooled or dropped; `True` if done in time |
| `stats()` | Completeness counters (see above); still available after `close()` |
| `close(timeout_s=30)` | Writes or spools the queued rows, then closes the connection |
| `with Pulse(...) as pulse:` | Calls `close()` when leaving the block |

## Analyzing the collected data with pulsar-metrics

The collected data has the columns expected by the `Analyzer` of
[pulsar-metrics](https://github.com/Rocket-Science-Development/pulsar_metrics) once `_time` is renamed:

```python
from pulsar_metrics.analyzers.base import Analyzer

# Drop the technical columns added by InfluxDB, then use the column names expected by the Analyzer
current = collected.drop(columns=["result", "table", "_start", "_stop", "_measurement"]).rename(
    columns={"_time": "pred_timestamp"}
)
reference = train.assign(model_id="credit_model", model_version="1.0", pred_timestamp=pd.Timestamp("2026-01-01"))

analyzer = Analyzer(name="hourly check", model_id="credit_model", model_version="1.0")
analyzer.add_drift_metrics(metrics_list=["ks_2samp", "psi"], features_list=["income", "age"])
analyzer.run(current=current, reference=reference, options={"psi": {"threshold": 0.2}})
print(analyzer.results_to_pandas().query("metric_type == 'drift'")[["metric_name", "feature_name", "drift_status"]])
```

## Development

The project uses [uv](https://docs.astral.sh/uv/) and a standard PEP 621 `pyproject.toml`.

```bash
cp tests/db/influxdb/.env.example tests/db/influxdb/.env   # test credentials for the InfluxDB container
uv sync                                                    # .venv with runtime and dev dependencies
uv run pytest                                              # starts InfluxDB 2.6.1 with Docker (pytest-docker)
uv run ruff check . && uv run black --check .
uv run bandit -c pyproject.toml -r pulsar_data_collection
```

The integration tests need a running Docker engine. The tests of the shared writer against a real InfluxDB
(`tests/test_writer_influxdb.py`) run only when `PULSAR_TEST_INFLUX_URL`, `PULSAR_TEST_INFLUX_TOKEN`,
`PULSAR_TEST_INFLUX_ORG` and `PULSAR_TEST_INFLUX_BUCKET` point at an instance you own; they are skipped otherwise.
Continuous integration runs lint, bandit and the tests on
Python 3.12 to 3.15 with pandas 3, plus Python 3.12 with pandas 2, for every pull request; tests are required to pass. Pushing a tag builds and publishes the
package.

## About [PulsarML](https://pulsar.ml/)

PulsarML helps monitor machine learning models in production and gain insights into their performance. It is made of
two open-source packages:

- [pulsar-data-collection](https://github.com/Rocket-Science-Development/pulsar_data_collection): lightweight SDK that
  collects features, predictions and metadata from a model serving code.
- [pulsar-metrics](https://github.com/Rocket-Science-Development/pulsar_metrics): data drift, concept drift and
  performance metrics, with custom metrics.

[pulsar-demo](https://github.com/Rocket-Science-Development/pulsar_demo) shows an example use case combining both
packages.

Powered by [Rocket Science Development](https://rocketscience.one/).

## Contributing

1. Open an issue describing the change.
2. Fork the repository, develop on a branch and add tests.
3. Check `uv run pytest` and the linters, then submit a pull request referencing the issue.

Licensed under the [Apache License 2.0](LICENSE).
