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
- [Sending evidence records to a central service](#sending-evidence-records-to-a-central-service)
- [Data control: residence, retention, deletion, export](#data-control-residence-retention-deletion-export)
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
  replayed, oldest first, as soon as the database is available again. Without `spool_dir`, they stay in the in-memory
  queue, in order, and are written once the database is back; the queue bound applies (`max_queue_rows`, below), so a
  longer outage drops the oldest batches and counts them in `dropped_rows` (before 0.12.1 every batch taken during the
  delay was dropped, even with room in the queue). During the delay InfluxDB is probed (`/ping`) every
  `probe_interval_s` (1 s): writing resumes about a second after the end of an outage, not up to 60 s later.
  `flush()` waits for the database during the delay (and returns `False` at its timeout); `close()` does not wait:
  during the delay it drops (or spools) and counts what is still queued. Requests in flight at once
  (`write_concurrency`, `SharedWriter(concurrency=...)`) can reach InfluxDB out of order after a retry, as before;
  the points keep their own timestamps.
- A write that **timed out**, or whose **connection was cut** (reset, closed) once the request was sent and before
  the answer, may have been stored by InfluxDB: with `spool_dir` it is spooled and written again (writes are
  idempotent: same measurement, tags and timestamp overwrite the same point, no duplicate); without it the rows go
  back to the queue, in their place, and are written **once more** when the back-off ends and InfluxDB answers its
  `/ping` (`retry_uncertain=True`, the
  default since 0.13.1; counted in `retried_uncertain_rows`). They are counted in `uncertain_rows` (not in
  `dropped_rows`) only if that second write is uncertain too, if the queue bound drops them (they are the oldest
  rows, so they go first), or if `close()` finds them still waiting; never written a third time.
  `retry_uncertain=False` counts them in `uncertain_rows` at once, as before. A failure that certainly stored
  nothing (a refused connection, an error answer other than 400 / 422) puts the rows of the failed request back in the
  queue with the batches that follow (or spools them).
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
`rejected_rows`, `uncertain_rows`, `retried_uncertain_rows` (rows of uncertain writes written once more; their
outcome is in the other counters), `write_errors`, `write_timeouts`, `last_error`, `queued_rows`, `writer_lag_s` (age
of the oldest queued batch), `last_write_ms` / `max_write_ms` (request latency), `spool_files`, `spool_bytes`,
`spool_evicted_rows`, `memory_only_rows` (`durability="wal"` batches kept in memory only because the spool was
full) and `storage_available`. Once everything is flushed, `written_rows + spooled_rows + dropped_rows +
rejected_rows + uncertain_rows = captured_rows`, and spooled rows end up in `replayed_rows` (or `rejected_rows`); any
`dropped_rows` (which includes `spool_evicted_rows`) is a measured gap in the monitoring data. Errors and drops are also
logged (logger `pulsar_data_collection`).

`pulse.flush(timeout_s)` waits until the queue is empty, for instance in a notebook before reading the data back.
During an outage without `spool_dir` it waits for the database, so it returns `False` after `timeout_s`: give it a
short timeout in a request path. The GenAI exporter's `force_flush(timeout_millis)` gives all its record types one
deadline together.

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

Failures are handled as for a private writer (see [Non-blocking capture and resilience](#non-blocking-capture-and-resilience)):
the Pulses writing to the same InfluxDB client and bucket share one back-off. While it lasts, each Pulse's batches go
to its spool or, without `spool_dir`, wait in its own queue up to its `max_queue_rows`, including batches already
taken for a request when another request in flight failed (0.11 dropped those: 10,000 rows at 300 models and 5,000 at
500 in the lab, with InfluxDB timing out under load and room left in the queues). The rows of requests that timed
out or were cut are written once more, then counted in `uncertain_rows` if still uncertain. A request holding rows of
several Pulse is retried per Pulse and batch: rows already retried are counted, the others get their own second write.
Rows waiting for one destination neither hold back the requests
of the others nor the replay of their spools.

**Memory during an outage.** Without `spool_dir`, each Pulse can hold its full `max_queue_rows` until the end of the
outage, so the bound of the process is *number of Pulse × `max_queue_rows`*: with the default 100,000 rows, 300
models can hold 30 million rows, about 3 GB at the roughly 100 bytes per queued row seen in the lab (SC-01, 500 Pulse:
973 MB of RSS with 6.2 million rows queued). There is no writer-wide cap: size `max_queue_rows` to the memory budget
(e.g. 10,000 rows for 300 models, about 300 MB), and for a multi-model server set `spool_dir`, which keeps the rows on
disk instead (bounded by `spool_max_bytes`) and survives a crash.

**Private writers or shared writer?** The shared writer now drains as fast as private writers (the line protocol
encoding used to limit it to 13,000 to 19,000 rows/s in the lab, see [Throughput](#throughput-measured)), so the
choice is about the caller side:

- a few models (up to about 10) per process: private writers (the default) are fine;
- tens to hundreds of models per process: `SharedWriter(concurrency=4)`. One thread instead of one per Pulse (the lab
  counted 39 threads instead of 534 at 500 models, and one TCP connection instead of 500), the capture p99 stays low
  while private writers fight for the GIL (lab, 100 models: 0.94 ms instead of 13.8 ms), and InfluxDB receives
  5,000-row requests instead of one small request per model and per second;
- above the per-process ceiling (next section), split the models over several processes: one Python process encodes
  on one core whatever the number of writers (four `SharedWriter` of 25 models drain 28,000 rows/s, one of 100 models
  25,000 rows/s, with the same CPU time per row).

### Durability modes

| `durability` | Rows lost by a crash of the process (`kill -9`) | Cost |
|---|---|---|
| `"memory"` (default) | the in-memory queue and the request in flight: about `flush_interval_s` of traffic at normal load, up to `max_queue_rows` + one batch (`max_batch_rows`, or the writer's) when the writer is saturated or the storage down (lab: up to 100,500 rows with the defaults in a 50k rows/s burst; with the storage down, one 5,000-row batch in 0.11, up to `max_queue_rows` since 0.12.1, which keeps the rows in the queue during the back-off) | none |
| `"wal"` (needs `spool_dir`) | none: each batch is written to `spool_dir` before `capture_data()` returns, deleted once stored, replayed at the next start (lab: 0 row lost and 0 duplicate over 9 `kill -9`, 3.4 million rows) | one file write per capture (no fsync: an OS crash or a power loss can still lose the last batches) |

The default loss window on `kill -9` is therefore **`max_queue_rows` + one batch** per Pulse. For a deployment that must
not lose rows on a crash (rolling restarts with `SIGKILL`, OOM killer), use `durability="wal"` with a `spool_dir` on a
local disk, and a `spool_max_bytes` sized for the longest outage to absorb (once the spool is full, new batches are
kept in memory only and counted in `memory_only_rows`).

At startup the spool also recovers what a crash left: complete `.tmp` files become spooled batches, incomplete ones are
deleted, write-ahead copies are replayed. A spool directory belongs to one Pulse of one process at a time. A spooled batch that can no longer be read or serialized is renamed `.bad`
and counted in `rejected_rows`.

### Throughput (measured)

The line protocol is built column by column (one conversion per frame, then one `str.format` per row), with the same
lines as the influxdb-client serializer, which costs about 5 ms per frame whatever its size. The shared writer encodes
one frame per model in each request (50 rows per frame with 100 models), so this was its ceiling. Encoding of 20 float
fields and an integer prediction, one core:

| Frame | influxdb-client serializer | column-wise encoder |
|---|---|---|
| 50 rows | 5.3 ms (9,500 rows/s) | 0.9 ms (55,000 rows/s) |
| 5,000 rows | 165 ms (30,000 rows/s) | 75 ms (66,000 rows/s) |

100 models in one process, 50-row captures of 20 float fields and an integer prediction, against InfluxDB 2.6.1 in
Docker on the same workstation (16 logical CPUs, about 3 times slower per core than the lab machine: capture p50
0.9 ms instead of 0.3 ms), before and after the new encoder:

| Setup | Writer alone (full queues) | Offered 1M rows/min (16,700 rows/s): written, lag at the end, capture p99 | Threads |
|---|---|---|---|
| `SharedWriter(concurrency=4)`, before | 5,900 rows/s | 4,500 rows/s, 16 s, 103 ms | 6 |
| 100 private writers, before | 22,500 rows/s | 9,500 rows/s, 2.0 s, 87 ms | 101 |
| `SharedWriter(concurrency=4)`, after | **24,800 rows/s** | **16,400 rows/s, 0.1 to 0.3 s, 21 ms** | 6 |
| 100 private writers, after | 27,700 rows/s | 14,700 rows/s, 0.6 to 1.1 s, 45 to 50 ms | 101 |

At 500k rows/min offered, the shared writer keeps up with a capture p99 of 2.0 ms (before: 5,800 rows/s written, 7.4 s
behind, p99 118 ms); private writers keep up too, with a p99 of 25 ms. At 1M rows/min this workstation reaches the
ceiling of one process: the caller threads and the writer share one core (the GIL), hence the higher p99.

**Per-process ceiling.** Encoding, HTTP client and `capture_data()` all run under the GIL, on one core. The lab
measured about **2,300 captures/s per process** on the caller side (`SharedWriter`, 500 models, p99 0.88 ms, one core
at 99 %), and the writer drains about 25,000 rows/s per process on this workstation (more on a faster core, less with
text columns). Beyond either, run several processes (e.g. several workers of the model server, each with its own
`SharedWriter`); more `SharedWriter` instances in one process do not add throughput.

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
| `spool_compression` | `None` | `"gzip"` (level 1), **recommended** with a spool: spooled batches 2.2 times smaller on random floats, more on repetitive values (lab, 30 min outage at 200k rows/min: 819 MB instead of 1,734 MB, 143 bytes per row instead of 303, capture p99 after the outage 1.2 ms), for a replay 9 to 21 % longer |
| `replay_max_rows_per_s` | `None` | Throttle of the spool replay, per Pulse (see below) |
| `row_spacing_ns` | 1,000 | Spacing of the timestamps of the rows of one capture (1 µs) |

**Replay throttle.** After an outage, the replay of the spool competes with the capture for the GIL: in the lab, with
10 private writers, the capture p99 rose to about 30 ms during the replay of a 10 or 30 min outage (1.2 ms with one
`SharedWriter` and gzip, unthrottled). `replay_max_rows_per_s` stays unthrottled by default, because the recovery
time becomes `backlog / replay_max_rows_per_s`, and the backlog depends on the traffic of each Pulse: any fixed value
low enough to matter would turn the 30 min outage of a 25,000 rows/s Pulse (45 million rows) into hours of replay.
Set it from the normal rate of the Pulse instead: **2 to 6 times its live rate** replays an outage in a half to a sixth
of its duration. Measured on this workstation with 10 private writers (333 rows/s each, 20,000 rows spooled each, the real encoder and
a storage answering in 2 ms):

| `replay_max_rows_per_s` | Recovery | Capture p99 during the replay |
|---|---|---|
| `None` | 6 s | 61 to 72 ms (about 2 ms without replay) |
| 2,000 (6 times the live rate) | 11 s | 21 to 36 ms |
| 666 (2 times the live rate) | 31 s | 16 to 20 ms |

With tens of models, the shared writer does more for the p99 than the throttle, since one thread replays instead of
one per Pulse.

## Capacity and degraded networks

**Capture ceiling (measured, 0.11 write path).** On the lab machines (load generator c7i.8xlarge, Intel Xeon
Platinum 8488C, 32 vCPU; InfluxDB 2.6.1 on an r7i.2xlarge), 50-row captures of 20 float features:

| | Ceiling per process |
|---|---|
| `SharedWriter(concurrency=4)`, 100 models | 33,000 to 34,000 rows/s written |
| Private writer (one Pulse) | about 25,000 rows/s written |
| Caller side, `SharedWriter` (300 to 500 models) | about 2,100 captures/s, one core at 99 % |
| Caller side, capture cost alone (p50 0.34 ms) | about 2,900 captures/s per core |

`capture_data()`, encoding and the HTTP client share the GIL, so one process uses one core whatever the number of
writers. Above these numbers, run several processes (each with its own `SharedWriter`), and size or partition InfluxDB
(the lab saw server-side timeouts at 35,000 to 38,000 rows/s per process). A slower core lowers every line (about
3 times slower per core on a laptop).

**Degraded network (lab test SC-04).** A Toxiproxy between Pulse and InfluxDB, 10 models at 2,000 rows/s,
`max_queue_rows=5,000`, `write_timeout_s=5`, 45 s of fault, both write profiles (run locally on a shared laptop, so
latencies are higher than on the lab machines):

- In every case the caller was never blocked (0 captures of 200 ms or more, p99 2 to 17 ms while the laptop was also
  running another test) and the counters were exact: `captured = written + dropped + uncertain + rejected + spooled`,
  and the rows read back from InfluxDB matched them (0 unexplained, 0 duplicate).
- **Latency** (+200 ms, +1 s per request): no effect on the data. Writes take longer, the queues hold a few
  thousand rows more, nothing is dropped.
- **Lossy link and flapping link** (stalls and connection resets; proxy down 5 s out of 10): a request whose
  connection is cut after it was sent has an uncertain outcome. Up to 0.13.0, without `spool_dir`, those rows were
  never written again, even with room in the queue: 5 to 32 % of the traffic was lost this way (all counted in
  `uncertain_rows`; InfluxDB had stored 0 to 10 % of them). A backend that is down behind a port forwarder or a load
  balancer (Docker, Kubernetes) often accepts the connection and then closes it, which also counts as uncertain, not
  refused. Since 0.13.1 these rows are written once more, once InfluxDB answers `/ping` (see above). Rerun without
  spool, rows lost (private writers / `SharedWriter`): lossy link 5 %, 200 / 0 (11,700 / 6,050 in 0.13.0); lossy link
  20 %, 600 / 5,000 (37,100 / 35,650; second write cut too); flapping link, 0 / 0 (41,450 / 39,800). Still 0
  duplicate, exact counters, no blocked capture, queues and RSS within their bounds. With `spool_dir` these rows are
  spooled and replayed: 0 row lost, 0 duplicate. **Set `spool_dir` on any link that can drop connections**: the
  single in-memory retry does not cover a second cut, a long outage or a restart.
- **Bandwidth cap** (1 Mbit/s): a 5,000-row request (about 560 bytes per row, 2.8 MB) needs about 22 s, so with
  `write_timeout_s=5` every request times out, the `/ping` probe answers, the writer tries again, and almost nothing is
  written while the cap lasts (rows uncertain, or spooled and replayed afterwards). Keep
  `max_batch_rows × bytes per row` under `bandwidth × write_timeout_s` (here 1,000 rows, or `write_timeout_s=30`).
  Even then a link slower than the traffic fills the queues, and the oldest batches are dropped and counted, **with
  or without `spool_dir`**: the storage keeps answering, so it is never considered unavailable and the spool does not
  take the overflow.

**Memory bound.** The queues hold at most *number of Pulse × `max_queue_rows`* rows, plus the requests in flight:
up to `max_batch_rows` × concurrency per writer (private writers: one per Pulse), each row in flight also holding
its line protocol and the request body (roughly 1.2 KB per row at 20 features, against about 200 bytes for a queued
row). In SC-04 the RSS grew by at most 126 MB (10 private writers, 50,000 rows queued and 50,000 in flight) and 55 MB
(`SharedWriter`) for bounds of 178 and 99 MB. The spool is on disk and bounded by `spool_max_bytes`.

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
| `agent_run` | agent run (`invoke_agent` span and its subtree) | `steps` (LLM calls), `tool_calls`, `distinct_tools`, `tool_error_rate`, `tool_errors`, `tool_timeouts`, `tool_soft_failures`, `tool_<tool>_calls` / `_errors` / `_timeouts` / `_soft_failures` (tools called in the run), `tool_status_counts`, `repeated_calls`, `repeated_calls_window`, `max_same_tool_calls`, `tool_cycle_repeats`, `loop_detected`, `first_step_no_tool`, `unavailable_tool_calls`, with tool definitions: `tool_args_checked`, `tool_args_invalid`, `tool_args_invalid_rate`, `tool_args_missing_required`, `tool_args_type_errors`, `tool_args_string_numbers`, `tool_args_unparsable`; `tool_catalog_hash`, `incomplete`, `out_of_scope_calls`, `delegation_depth`, `input_tokens`, `output_tokens`, `latency_s`, `termination`, `task_success`, `first_tool`, `tool_sequence`, `request_model`, `prompt_version`; with `compliance`: `violated_rules`, `critical_violations`, `rules_version` |
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

with agent_span(tracer, "permit_assistant", available_tools=TOOL_SCHEMAS):  # names, or the definitions themselves
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
| Tool timeouts and errors | Per tool: `tool_<tool>_calls`, `tool_<tool>_errors`, `tool_<tool>_timeouts`, `tool_<tool>_soft_failures`, for the tools called in the run only (a missing counter means 0; see `tool_fields` below). Per run: `tool_errors`, `tool_timeouts`, `max_tool_error_rate`, `tool_status_counts` (`"ok=3,timeout=2"`). A failed call is a `timeout` when its `error.type` names one (`TimeoutError`, `ReadTimeout`...). `tool_call` records carry `status`, `timeout` and `soft_failure`. |
| Tool that answers but returns nothing usable | `record_tool_result(span, output, status=None, valid=None, validator=None)` sets `pulsar.tool.status` (by default the `status` key of a dict output) and `pulsar.tool.output_valid`. The statuses `retry`, `invalid`, `empty`, `partial`, `garbage` and invalid outputs are soft failures: `tool_soft_failures`, `soft_failure_rate`, `tool_<tool>_soft_failures`. The output itself is never written. |
| Loops | `repeated_calls` keeps its meaning (consecutive calls identical byte for byte). `repeated_calls_window`: calls whose **normalized** tool and arguments (JSON parsed, keys sorted, `36000` = `36000.0` = `"36000"`, strings trimmed and lower-cased) appeared among the `loop_window` (4) previous calls; `loop_detected` is True from `loop_min_repeats` (2) such calls, so ping-pong A>B>A>B and type variations are loops. `max_same_tool_calls` and `tool_cycle_repeats` (longest repetition of a block of 2 or 3 tools) are counts, not verdicts: the same tool called with other identifiers can be legitimate, test them against a reference. |
| Removed or added tool | With `agent_span(..., available_tools=[...])` (`pulsar.agent.available_tools`): `tool_catalog_size` and `tool_catalog_hash`, which changes with the catalogue from the first run; with `allowed_tools`, `catalog_missing_tools` and `missing_tools`. `unavailable_tool_calls`: calls to a tool outside the catalogue, or with status `unavailable`, `unknown_tool` or `not_found`. |
| Wrong arguments (no label needed) | When the run records the tool definitions (`agent_span(..., available_tools=TOOL_SCHEMAS)` with the definitions themselves, `gen_ai.tool.definitions`, or OpenLLMetry's `llm.request.functions.<i>.name` / `.parameters`), every call is checked against its tool's JSON schema (`type`, `required`, `enum`, `additionalProperties: false`, nested properties and items): `tool_args_checked`, `tool_args_invalid` (calls with at least one problem), `tool_args_invalid_rate`, `tool_args_missing_required`, `tool_args_type_errors`, `tool_args_string_numbers` (numbers sent as strings, e.g. `{"income": "36000"}`), `tool_args_unparsable` (not a JSON object). Only counts are written. Without definitions, the fields are absent. |
| Model that answers instead of acting | `first_step_no_tool`: the first model call was followed by no tool call before the next model call (or the end of the run). |

Per-tool counters are bounded with `tool_fields` (`PulsarSpanExporter` and `TrajectoryAssembler`):

| `tool_fields` | Tools with `tool_<tool>_*` counters |
|---|---|
| `"called"` (default) | the tools called in the run (at most 4 fields per tool called) |
| `"allowed"` | every tool of `allowed_tools`, 0 when not called (needs `allowed_tools`) |
| `"top_k"` | the `tool_fields_k` (10) tools called most in the run |
| `"all"` | former behaviour: tools called, allowed and of the catalogue, 0 when not called |
| `"none"` | none: the per-call detail stays in the `tool_call` records (`tool_name`, `status`...) |

Formerly every tool of the catalogue got four fields, 0 when not called: in the lab, 1200 runs on BFCL catalogues
carried 2308 per-tool fields. With `"called"`, a run has at most 8 of them there; over the series the field names still
follow the tools actually called (1568 for the 392 BFCL tools called): with catalogues that change per request, prefer
`"allowed"` or `"none"`.

### Rule violations in `agent_run`

`PulsarSpanExporter(..., compliance=monitor, rules_version="2026-11")` checks every run when it is assembled and adds
`violated_rules` (rule ids, comma-separated, `"none"`), `violations`, `critical_violations` (rules with
`severity="critical"`) and `rules_version` (default: `monitor.version`). `compliance` is a
`pulsar_metrics.genai.ComplianceMonitor` (pulsar-metrics is not a dependency: any object with `violations(trace)`
fits) or a callable `trace -> violated rule ids`; the trace is the list of the tools called, as
`trace_from(tool_sequence)` rebuilds it. A rule set that raises never stops the collection
(`stats()["compliance_errors"]`). Without `compliance`, the fields are absent. A monitor with `not_evaluable(trace)`
adds `not_evaluable_rules` (ids of the rules the trace cannot decide, comma-separated), absent when there is none.

A monitor declaring `accepts_rich_trace = True` receives the **rich trace** instead: the tool calls and the
conversation events of the run, in time order, each a dict:

```python
{"name": "get_order_details", "kind": "tool", "status": "ok", "args": {"order_id": "W123"}}
{"name": "user_confirmation", "kind": "event"}
{"name": "cancel_pending_order", "kind": "tool", "status": "timeout", "args": {"order_id": "W123"}}
```

`status` is `"ok"`, `"error"` (a call to an unavailable tool included), `"timeout"` or `"soft_failure"`; `args` holds
the parsed arguments when they are a JSON object. The values are used in memory for the check only and are never
written (the stored records keep the hashed signatures). Conversation events are OpenTelemetry span events named
`pulsar.event.<name>` on any span of the run; record them with `record_event`:

```python
from pulsar_data_collection.genai import record_event

with agent_span(tracer, "retail_assistant"):
    ...
    if user_said_yes:
        record_event("user_confirmation")  # on the current span; record_event(name, span=run) for another one
    with tool_span(tracer, "cancel_pending_order", arguments={"order_id": order_id}):
        ...
```

Policies that need the rich trace: confirmations ("a write only after the user said yes"), and every rule on
arguments or outcomes ("cancel the order that was looked up", "no retry after a timeout"). In the lab (tau-bench
retail, 40 episodes), the confirmation policy failed on 36 of 36 runs with a write on the tool names; through the
exporter with `user_yes` events in the rich trace it gives the 23 real violations. The tool spans and events recorded
by the helpers carry a sequence number (`pulsar.sequence`), so that their order holds when the clock is coarse
(15.6 ms on Windows); other instrumentations are ordered by timestamp only.

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

`ComparativeProbe` rows also say whether each answer is usable. `answer_status` is `"ok"` (a choice was read),
`"refusal"` (French or English refusal phrasings: "Je ne peux pas", "Je suis désolé", "I can't", "I'm sorry"...; or a
`refusal=` callable), `"unparsed"` or `"error"`; `unusable` is True for refusals and unparsed answers. `choice` is
`"refusal"` for a refusal (it was None, like an unparsed answer, which stays None); `chosen_group` is None for both. In the lab, 23.3 % of the answers of llama3.2:3b were refusals that
looked like unparsed answers. Two instruments help when the choice question is driven by the position (the 6 local
models of the lab answered "first" from 2.8 % to 99.1 % of the time):

- `labels="neutral"` rotates the option labels across sets (A/B, 1/2, X/Y and reversed; or a list of pairs); the
  template then needs `{first_label}` and `{second_label}` slots, and `run(probes, target)` without `choice` reads the
  label each probe offered (`label_choice`). The `labels` column says which pair was shown;
- `run_loglik(probes, loglik)` compares the log-likelihood of the two profiles (`loglik(text) -> float`) when the target
  exposes logprobs: nothing is asked, so there is no position effect (`instrument="loglik"`, `loglik_first`,
  `loglik_second`). Without logprobs (`loglik=None`, or a first call that raises NotImplementedError or returns None)
  the instrument is skipped: an empty frame, `rows.attrs["skipped"] == "no_logprobs"`.

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

A detector can be shared by several threads (e.g. an agent served with `--workers 4`): each one serializes its
tokenizer and model calls with a lock of its own, since the Hugging Face fast tokenizers are not thread-safe (before
0.12.1, 37 of 40 concurrent `FaithfulnessScorer.score` calls from 4 threads raised `RuntimeError: Already borrowed`).
The calls of one detector then run one at a time; to score in parallel, give each worker its own detector (one model
in memory each).

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

### Consistency of repeated answers

`ConsistencyProbe` asks each question several times (`repeats`, default 3) and, optionally, in other words supplied
by the user (`paraphrases`, asked `paraphrase_repeats` times, default 1), and scores how much the answers agree. The
calls go to any `ask(text) -> answer`; `openai_chat_target` builds one for an OpenAI-compatible endpoint (Ollama,
vLLM, a provider), with a cap on the answer length (`max_tokens`, default 128) and a client timeout (default 120 s).
Requests are sent round after round, `workers` at a time (default 1). Prefer `127.0.0.1` to `localhost` for a local
endpoint on Windows: there `localhost` is tried in IPv6 first, and each call waited about 20 s before the fallback.

| `answer_type` | Normalized final answer | Two answers agree when |
|---|---|---|
| `yes_no` | `"yes"` / `"no"` (first oui / yes / non / no / vrai / true / faux / false) | the normalized answers are equal |
| `number` | first number, `"6370"` for "6 370 $" or "6,370", `"3.5"` for "3,5 %" | idem |
| `label` | the label of `labels` named first (accent- and case-insensitive) | idem |
| `short` | folded text without punctuation nor leading article | idem |
| `free` | none | `similarity(a, b)` reaches `agreement_threshold` |

The text after a final "Answer:" or "Réponse :" marker is read when there is one. For free text, `similarity` is
`EmbeddingSimilarity` (cosine of the embeddings of an OpenAI-compatible `/embeddings` endpoint, e.g. Ollama's
`nomic-embed-text` or the multilingual `granite-embedding:278m`; threshold 0.9 by default, to fit on your answers),
`EntailmentSimilarity(FaithfulnessScorer())` (bidirectional NLI entailment, the weaker direction by default; threshold
0.5) or any callable; the default `LexicalSimilarity` (word overlap) is a weak proxy.

```python
from pulsar_data_collection.genai import ConsistencyProbe, ConsistencyQuestion, openai_chat_target

ask = openai_chat_target("http://127.0.0.1:11434/v1", "llama3.2:3b", temperature=0.7, max_tokens=48, timeout=180)
probe = ConsistencyProbe([
    ConsistencyQuestion("capital-mb", "Quelle est la capitale du Manitoba ? Le nom de la ville seulement.",
                        answer_type="label", labels=("Winnipeg", "Brandon", "Regina"), expected="Winnipeg"),
    ConsistencyQuestion("pi-yn", "Le système implique-t-il des renseignements personnels ? Oui ou non.",
                        paraphrases=("Des renseignements personnels sont-ils utilisés ? Oui ou non.",), answer_type="yes_no"),
], repeats=3)
result = probe.run(ask, workers=2)
result.questions  # one row per question, for pulsar_metrics.genai.consistency / consistency_drift
```

`result.answers` has one row per call: `question_id`, `variant` (`"original"`, `"paraphrase-1"`...), `repeat`,
`answer`, `normalized`, `answer_status` (`ok`, `refusal`, `unparsed`, `error`), `error`, `latency_s`; the question
texts are not stored. `result.questions` has one row per question: `n_answers`, `n_errors`, `n_refusals`,
`n_unparsed`, `n_pairs`, `agreement` (mean pair score), `agreement_repeats` (pairs of the original wording),
`agreement_paraphrases` (original x paraphrase), `min_pair`, `full_agreement` (every pair agrees), `majority_answer`
and `majority_share`, `correct_share` (when `expected` is given), `agreement_method` and `pair_scores`.

- Two refusals of the same question agree, a refusal and an answer do not; an answer whose final answer cannot be
  read agrees with nothing; failed calls are counted (`n_errors`) and left out.
- `keep_answers=False` drops the answer texts (the normalized answers stay); `pii_policy=PiiPolicy()` masks (or only
  counts) personal information in the answers, the normalized answers and the majority answer before they are
  returned. The scores are computed on the texts as received, in memory.

### Abstentions and answers without support (RAG agents)

`pulsar_data_collection.genai.abstention` judges each run of a retrieval agent on two points, without any model by
default:

- **abstention**: the answer says the information is not found, not in the sources, or that it cannot answer
  (`AbstentionDetector`, French and English rules: `NOT_FOUND_PATTERNS`, `CANNOT_ANSWER_PATTERNS` and
  `FILLER_PATTERNS`, documented in the module). An answer that says the sources lack the information and then answers
  anyway is *hedged*: it counts as an answer. An optional fallback scores the answers the rules leave undecided, e.g.
  `NliAbstentionFallback.from_scorer(FaithfulnessScorer())`;
- **answer without support**: the agent answered although the retrieval was empty, its best score was under a
  threshold, the application's own relevance check failed (`retrieval_relevant=False`, e.g. the entity named in the
  question is absent from the passages), or the faithfulness was under a threshold. The thresholds are the lower
  5 % of a reference of the deployment (`calibrate_support_thresholds`).

`record_answer_support` writes both verdicts (1.0 / 0.0) on the agent span: the `agent_run` record gains
`det_abstention` and `det_unsupported_answer`. Their rates per window are the hallucination proxy; test them in
pulsar-metrics with `DriftTestMetric("rate_increase", "det_unsupported_answer")` (one-sided exact test against the
reference). Only the verdicts reach the spans, never the text.

```python
from pulsar_data_collection.genai import calibrate_support_thresholds, record_answer_support

# once, on a reference of the deployment: best retrieval scores, faithfulness of the answered runs
thresholds = calibrate_support_thresholds(reference_top_scores, reference_faithfulness)

with agent_span(tracer, "register-assistant") as run:
    ...  # the agent searches and answers
    support = record_answer_support(run, answer, retrieval_empty=not passages, retrieval_top=best_score,
                                    faithfulness=faithfulness_score, thresholds=thresholds)
print(support.abstained, support.unsupported, support.reasons)  # e.g. False True ('low_retrieval_score',)
```

Measured offline on recorded runs (pulsar_demo `lab/replay/signal_abstention.py`, report
`lab/rapports/rapport_signal_abstention.md`): on 102 hand-labelled answers (33 abstentions) of the LR-03 lab test and
of the GC AI Register trial, the rules had a precision of 0.97 [0.85-0.99] and a recall of 1.00 [0.90-1.00]
(Wilson 95 %); the one error is a guess justified by the missing information ("..., car le passage n'a pas fourni
d'informations"). The patterns were written on the other answers of the same two runs, both from llama3.2:3b and
granite3.3:2b: other models and other languages need their own check.

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

## Sending evidence records to a central service

When the raw data must stay where it is collected and only aggregated results may leave, `EvidenceOutbox` sends
evidence records (e.g. the `"pulsar.evidence/1"` records of `pulsar_metrics.evidence`) to an HTTPS endpoint. It takes
plain dicts and only checks that each carries a `schema_version` and an `evidence_id`: it does not depend on
pulsar-metrics.

```python
from pulsar_data_collection.evidence import EvidenceOutbox

with EvidenceOutbox(
    "https://governance.example/api/evidence",
    directory="/var/lib/pulsar/outbox",
    token_env="PULSAR_EVIDENCE_TOKEN",  # or token=..., or headers=callable returning the Authorization header
    ca_bundle="/etc/pki/internal-ca.pem",  # optional: private CA; certificates are always verified by default
) as outbox:
    outbox.put_many([r.to_dict() for r in records])  # durable on return
    print(outbox.stats())
```

- **Durable first.** `put` / `put_many` write to a SQLite file in `directory` (WAL journal, `synchronous=FULL`) before
  returning; a background thread then POSTs batches (`batch_size`, `max_batch_bytes`) as
  `{"records": [...], "sent_at": "..."}`. Records left by a crash or a kill are sent by the next outbox opened on the
  same directory.
- **At least once.** A record leaves the store only on a 2xx answer, so a lost answer means a second send: the
  receiver deduplicates on `evidence_id`; a 409 answer (the receiver already holds the id) counts as delivered. A
  record put again while pending replaces the pending copy and goes to the end of the queue: the answer to a send of
  the old version never deletes nor quarantines the new one.
- **Retries.** Network errors, timeouts, 408, 429 and 5xx are retried with exponential backoff and jitter
  (`retry_initial_s` > 0, `retry_max_s` >= `retry_initial_s`, 0 <= `jitter` < 1), honouring `Retry-After`; for ever,
  or `max_attempts`. 401, 403, 404, 405, 410, 421 and redirects (refused, so that the token never follows them) are
  endpoint or configuration errors: retried and logged as errors, never discarded. Any other 4xx may mean that a
  record is refused: the batch is split and its records sent one by one. A record refused on its own goes to the
  quarantine table (`quarantined()`, never retried) when another record of the same batch was accepted, when it was
  alone in its batch, or when it is too large (413). When every record of a batch is refused one by one, the request
  is taken to be refused (schema or endpoint error), not the records: they are retried.
- **Bounded.** At most `max_records` pending (default 100 000); beyond, `overflow="drop_oldest"` (default) or
  `"drop_newest"`, and every record lost is counted in `dropped` and logged.
- **Secrets.** The token is never logged, shown by `repr()` or kept in errors; credentials in the URL are refused and
  the query string is left out of logs. `https` is required (`allow_http=True` for loopback tests only), every request
  has a timeout (`timeout_s`, default 10 s), and system proxies (`HTTPS_PROXY`) apply.
- `flush(timeout_s)` sends what is pending now and waits; `close(flush=False)` leaves it on disk for the next start.
  A closed outbox cannot be started again (`RuntimeError`): open a new one on the same directory.

`stats()`: `queued`, `sent`, `retried` (record sends that failed and will be retried), `failed` (moved to the
quarantine), `dropped`, `replaced`, `requests`, `pending`, `quarantined`, `oldest_pending_age_s`,
`consecutive_failures`, `receiver_available`, `last_error`.

## Data control: residence, retention, deletion, export

The client controls where the governance data lives and how long it is kept, can take it or delete it at any time,
and keeps an audit trail of who did what, from a policy file and a command line, without the vendor
(`pulsar_data_collection.governance`).

### Where the data lives

| Data | Where | Kept until | Leaves the client? |
|---|---|---|---|
| Captured rows (features, predictions, tags) | client's InfluxDB, bucket of role `raw` | the bucket's retention | no |
| Analysis results, capture statistics, process health | client's InfluxDB, bucket of role `results` (may be the same bucket) | the bucket's retention | no |
| Rows waiting for InfluxDB (spool, write-ahead copies) | `CaptureOptions.spool_dir`, on the capture host | written to InfluxDB | no |
| Evidence records not yet delivered (and quarantined ones) | `EvidenceOutbox` SQLite store (`evidence_outbox.sqlite3`), on the host running the outbox | delivered (2xx) | **yes, these only** |
| Audit journal of retention changes, deletions and exports | policy `journal` file | never deleted by Pulsar | no |
| Exports | policy `export_dir`, or the directory given | managed by the client | no |

Only the evidence records (aggregated results, schema `pulsar.evidence/1`) leave, to the `central_service` of the
policy; raw rows never do. Once delivered, a record is held by the central service, out of reach of the deletion
below. `describe()` (or the `describe` command) returns these locations as configured, with the actual retention of
each bucket and the size of each local store; never a token.

### Policy and command line

```yaml
# governance.yaml (JSON works too; YAML needs PyYAML). Relative paths are relative to this file.
influxdb:
  url: http://influxdb.ministry.internal:8086
  org: ministry
  token_env: PULSAR_INFLUX_TOKEN   # the token stays in the environment: a token in the file is refused
buckets:                           # role -> bucket; retention: 90d, 12h, 2w, 1y (365 days), seconds or infinite
  raw: {name: captures, retention: 90d, content: raw captured rows}
  results: {name: results, retention: 2y, content: analysis results and capture statistics}
spool_dirs: [/var/lib/pulsar/spool]
outbox_dirs: [/var/lib/pulsar/outbox]
journal: /var/lib/pulsar/governance/journal.jsonl
journal_key_env: PULSAR_JOURNAL_KEY   # optional: HMAC-SHA256 chain instead of SHA-256
export_dir: /var/lib/pulsar/exports
central_service: https://governance.example/api/evidence
```

```bash
export PULSAR_INFLUX_TOKEN=...      # a token allowed to read, write (delete) and update the buckets
G="python -m pulsar_data_collection.governance --policy governance.yaml --actor jdoe"
$G describe                                       # where each kind of data lives (--offline: no connection)
$G retention check                                # declared against actual; exit code 1 on drift
$G retention apply --reason "CHG-1234"            # set the declared retention on the buckets in drift
$G delete --system-id SYS-042 --dry-run           # what would be deleted: buckets, measurements, spool, outbox
$G delete --system-id SYS-042 --reason "request 17"
$G delete --model-id credit_model --start 2026-01-01 --stop 2026-03-31T23:59:59Z --role raw
$G export --system-id SYS-042 --start 2026-01-01 --out ./sys-042   # CSV (or --format parquet) and manifest.json
$G export-verify ./sys-042                        # checksums of the files against the manifest
$G journal verify --expected-head <hash recorded elsewhere>
```

Output is JSON; the exit code is 1 when a check fails (retention drift, broken journal, incomplete deletion, altered
export). The same from Python:

```python
from pulsar_data_collection.governance import DataController, DataSelector

with DataController("governance.yaml", actor="jdoe") as control:
    print(control.check_retention())
    selector = DataSelector(system_id="SYS-042")              # and/or model_id, model_version, data_id, tags=...
    print(control.delete(selector, dry_run=True)["buckets"])  # counts only
    control.delete(selector, reason="request 17")
    control.export(DataSelector(model_id="credit_model"), "./credit", start="2026-01-01")
    print(control.journal.verify())
```

- **Retention.** `check_retention()` compares the declared retention of every bucket with its actual retention rule
  (`ok`, `drift`, `missing`, or `unmanaged` when the policy declares none); `apply_retention()` sets it through the
  buckets API, with the shard group duration InfluxDB would pick (1 h below 2 days, 1 day below 6 months, 7 days
  beyond; `shard_group_duration` to choose another), reads it back and journals the old and new values. A missing
  bucket is reported, never created. InfluxDB refuses retentions shorter than one hour.
- **Selection.** Rows are selected by the tags Pulse writes (`model_id`, `model_version`, `data_id`, and the
  `additional_tags`: by convention, give every Pulse of a system `additional_tags={"system_id": ...}`) and the tags
  given to `ResultsSink` (e.g. `tags={"model_id": ..., "system_id": ...}`), optionally one `measurement`, and a period
  inclusive at both ends (default: everything). Every criterion must match. A period without any tag criterion needs
  `all_series=True` (`--all-series`). Values with quotes, backslashes or control characters are refused: they cannot
  be expressed in a delete predicate.
- **Deletion.** For each bucket: count, delete through the InfluxDB delete API with the predicate
  `system_id="SYS-042" AND ...` over the period, count again (`remaining` must be 0, else `status: incomplete`). Then
  the spool directories: selected batches are deleted, partly selected ones rewritten with the kept rows (same time
  prefix and id, so the replay order is kept; the old file is deleted only once the new one is in place). Then the
  outbox stores: pending and quarantined evidence records whose `system_id` / `model_id` / `model_version` match (and
  whose window intersects the period). Selectors on `data_id`, `measurement` or other tags do not apply to evidence
  records: the outbox is then left untouched and the report says so. `dry_run=True` counts everything and changes
  nothing. Stop the capture of the system (close its Pulse) before deleting: a spool directory belongs to one running
  Pulse, and rows still in its memory would be written afterwards. A running outbox copes with the deletion of its
  records, but those it had already read for a send in progress are still delivered: close the outbox first.
- **Export.** One file per bucket role and measurement (e.g. `raw/credit_model_1.0_input_data.csv`): `_time` (UTC,
  RFC 3339 with nanoseconds, exact), `_measurement`, the tags, then one column per field; `manifest.json` lists the
  selector, the period and every file with its rows, columns and types (pandas `columns`, InfluxDB `influx_types`),
  tags, size and SHA-256. Values keep their InfluxDB type, missing values included: integer fields are nullable
  integers (`Int64` / `UInt64`, exact: 2**60 + 1 is not rounded to a float), booleans and strings nullable too. CSV
  convention (`csv` in the manifest): strings are quoted, a missing value is an empty unquoted cell; InfluxDB 2.6
  returns an empty string field like a missing one, so both are exported as missing. Parquet needs
  `pyarrow`. A measurement is held in memory while it is written: export long periods in parts (`start` / `stop`).
- **Audit journal.** Every retention change, deletion (whatever its outcome, with the error if any) and export
  appends one JSON line, fsynced: sequence number, UTC time, actor, action, details (bucket, predicate, period,
  counts, reason); never data values, never tokens (details with secret-looking keys are refused). A deletion writes
  a `delete.intent` entry (predicate and counts) before deleting anything, then its `delete` outcome entry
  (`intent_seq`). Before any change, the journal must be able to take the entry: a key variable not set, a
  secret-looking key in the selector (e.g. a tag named `token_bucket`) or a last line that is not a complete entry
  (a write interrupted by a crash) stops the operation with nothing changed. A torn last line is never skipped:
  keep a copy of the file, remove that line, and check the journal with `verify()`. Each entry holds
  the hash of the previous one and its own (SHA-256, or HMAC-SHA256 with `journal_key_env`). `verify()` finds the first
  modified, inserted, removed or reordered entry. Removing the last entries, or rewriting the whole file without the
  key, leaves a valid chain: record the head (`journal.head()`, also in `describe()`) elsewhere, e.g. in the change
  ticket, and check it with `verify(expected_head=...)`. Dry runs are not journaled (nothing changes).

Known limits (InfluxDB 2.6):

- the delete predicate accepts only `key="value"` terms joined by `AND` (no `OR`, no regular expression, no field):
  one system, model or tag combination per call;
- a deletion applies at once to queries, but the disk space is reclaimed later, by compaction. A retention is
  enforced by dropping whole shard groups, checked every 30 minutes: rows can stay up to the retention plus one shard
  group duration;
- deleting from a large bucket is expensive for InfluxDB (tombstones on every series hit): prefer narrow selectors
  and periods on a busy server;
- the token needs read and write permission on the buckets (deleting is a write) and, for `apply_retention`,
  permission to update them.

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
| `flush(timeout_s=30)` | Waits until the queued rows are written, spooled or dropped; `True` if done in time (during an outage without `spool_dir`, the rows wait for the database: `False` at the timeout) |
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
The data control tests against a real InfluxDB (`tests/test_governance_influxdb.py`) need the first three of these
variables and a token allowed to create buckets: each test creates its own two buckets and deletes them.
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
