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
| `agent_run` | agent run (`invoke_agent` span and its subtree) | `steps` (LLM calls), `tool_calls`, `distinct_tools`, `tool_error_rate`, `repeated_calls`, `loop_detected`, `out_of_scope_calls`, `delegation_depth`, `input_tokens`, `output_tokens`, `latency_s`, `termination`, `task_success`, `first_tool`, `tool_sequence`, `request_model`, `prompt_version` |
| `tool_call` | `execute_tool` span | `tool_name`, `error`, `latency_s`, `out_of_scope` |
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
- An agent run is assembled when its trace's root span ends; if the root lives in another process (HTTP server span),
  after `trace_timeout_s` without new span. Sub-agents are part of the run (`delegation_depth`).
- Read a record type back with `|> filter(fn: (r) => r["record_type"] == "agent_run")`.

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
| `InjectionDetector` | probability that a text is a prompt injection or a jailbreak | protectai/deberta-v3-base-prompt-injection-v2 (Apache 2.0) |
| `ModelPiiDetector` | rule-based findings plus names and street addresses | urchade/gliner_multi_pii-v1 (Apache 2.0) |
| `FaithfulnessScorer` | share of the answer entailed by the retrieved passages (multilingual NLI) | MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 (MIT) |

Any classifier with a `score()` method fits (e.g. Prompt Guard 2, under the Llama licence); `record_score(span, name,
value)` writes it. `CanaryProbe` asks a fixed set of prompts several times and saves the answers (they are not user
data) for `pulsar_metrics.genai.model_equality`.

## Personal information

`pulsar_data_collection.privacy` detects personal information with **validated rules** (checksums, valid dates,
issuer prefixes, context words) for Canada, in French and English, and **masks it before anything is written**.
Findings carry the type, the position and a confidence, never the value.

| Type | Label | Validation |
|---|---|---|
| `sin` (social insurance number) | `[SIN]` | 9 digits, Luhn checksum, first digit issued |
| `ramq` (Québec health insurance number) | `[RAMQ]` | 4 letters + 8 digits, valid month (women: +50) and day |
| `credit_card` | `[CREDIT_CARD]` | 13–19 digits, Luhn, issuer prefix |
| `email`, `phone` (North American), `postal_code` (Canada) | `[EMAIL]`, `[PHONE]`, `[POSTAL_CODE]` | format |
| `drivers_licence_qc`, `passport` (Canada) | `[DRIVERS_LICENCE]`, `[PASSPORT]` | format; a passport needs a context word |
| `date_of_birth` | `[DATE_OF_BIRTH]` | valid date **and** a context word ("né le", "date of birth"...) |

- **Context**: a word such as "NAS", "téléphone" or "passport" just before raises the confidence; a word announcing a
  business number ("dossier", "commande", "order", "invoice"...) lowers it below the threshold, and so does a currency
  right after a number ("514 555 0199 $" is an amount). Findings below `min_confidence` (0.5) are ignored.
- **Tabular models**: `CaptureOptions(pii=PiiPolicy())` masks the text columns in `capture_data`, in the caller's
  thread, so raw values never reach the queue, the spool directory or the database; a `pii_count` column counts the
  findings per row. `PiiPolicy(action="flag")` only counts.
- **LLM applications and agents**: `PiiSession` scans the text of each role and adds counts and types to the spans:
  `input` (from the user), `context` (retrieved documents, tool results) and `output` (answers, tool calls). A
  **leak** is personal information in an output that the user did not provide in the same session. The
  `agent_run`, `llm_call` and `tool_call` records gain `pii_<role>_count` and `pii_<role>_types` (`input`, `context`,
  `output`, `leak`), only when a session recorded something. The values given by the user stay in memory for the
  session only.
- Measured on a synthetic French and English corpus (fictitious values with valid checksums, plus business numbers
  that must not be detected): precision and recall of 1.00 for every type. The corpus is ours: real text will have
  misses and false positives, and names and street addresses are not covered by rules.

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

The integration tests need a running Docker engine. Continuous integration runs lint, bandit and the tests on
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
