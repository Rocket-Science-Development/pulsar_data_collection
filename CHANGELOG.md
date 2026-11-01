# Changelog

All notable changes to pulsar-data-collection. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.5.0] - 2026-11-01

### Added
- `pulsar_data_collection.genai` (extra `genai`): collection for LLM applications and agents from OpenTelemetry
  GenAI spans (`gen_ai.*`). `PulsarSpanExporter` assembles agent runs (steps, tool calls, distinct tools, tool error
  rate, repeated calls and loops, out-of-scope calls, delegation depth, tokens, latency, termination, task success,
  tool sequence, model, prompt version), tool calls and LLM calls, and writes them with `PulseSink` (non-blocking
  Pulse, tags `record_type` and `schema=pulsar-genai/1`). Only features are stored, never prompts, responses or tool
  arguments.
- Pinned GenAI attribute names with normalization of aliases used by other SDKs (`genai/semconv.py`).
- Instrumentation helpers `agent_span`, `llm_span`, `record_llm_response`, `tool_span`.

## [0.4.0] - 2026-11-01

### Changed
- **`capture_data()` no longer waits for the database** (new default `mode="background"`). It queues the rows and a
  background thread writes them in batches. Previously every capture waited for the write (about 110 ms), and with
  InfluxDB stopped a single capture blocked the calling thread for 195 s (5 hidden retries) and then returned without
  error while the data was lost (#21).
- Writes use the synchronous InfluxDB API with a request timeout (`write_timeout_s`) and no hidden retries;
  `write_data()` raises on failure instead of printing from a background callback.
- `Pulse` starts even if InfluxDB is unreachable (warning, rows spooled or counted until it is back); configuration
  errors still raise. New `StorageUnavailableError`.

### Added
- `CaptureOptions`: mode (`background` / `sync`), bounded queue, batch size, flush interval, request timeout,
  retry delay, spool directory, strict startup.
- Spool directory: batches that cannot be written are saved to disk (exact round trip, nanosecond timestamps) and
  replayed when InfluxDB is back.
- `Pulse.flush()` and `Pulse.stats()` (captured, written, spooled, replayed, dropped rows; write errors; queue size).
- Validated with InfluxDB stopped for 15 s during the service: capture latency 0.8 ms (1.3 ms max), no row lost
  (3,000 / 3,000); service start with InfluxDB down in 2 s, rows replayed when it came back.

## [0.3.0] - 2026-11-01

### Changed
- **Unknown fields are rejected** by `PulseParameters` and `DataWithPrediction` (pydantic `extra="forbid"`): a misspelt
  field such as `other_labels` (instead of `additional_tags`) raised no error and its content was lost (#20).
- **The storage configuration is checked when `Pulse` is created**: an unknown bucket, an unknown organization or a
  rejected token raises `StorageConfigurationError` (new, in `pulsar_data_collection.exceptions`, subclass of
  `ValueError`) with the reason given by InfluxDB. Previously an unknown bucket gave a `None` connection and the error
  only appeared at the first capture; an unknown organization was not detected; a rejected token raised a raw
  `ApiException` (#20).

### Fixed
- `test_make_connection_fails` passed even when no exception was raised (#20).

## [0.2.1] - 2026-11-01

### Fixed
- A `Pulse` instance could be used only once: `write_data()` closed the shared InfluxDB connection after the first
  write and every following `capture_data()` failed (`ValueError: Invalid value for api_client`) (#18).
- `write_data()` modified the caller's `data_points` DataFrame when no prediction was given (#18).
- `_check_if_bucket_exists()` left its temporary client open (#18).

### Added
- `Pulse.close()` and context manager support: `with Pulse(params) as pulse: ...` (#18).

### Documentation
- README rewritten for the current API: InfluxDB setup, quick start (collect, read back, analyze with pulsar-metrics),
  storage layout (measurement, tags, fields, timestamps), API reference, clock skew with Flux `range()`.
  The previous example used `other_labels`, a field that does not exist and was silently ignored: the field is
  `additional_tags`.

## [0.2.0] - 2026-11-01

### Changed
- Python 3.12 or later (tested on 3.12 and 3.13), pandas 2, pydantic 2, influxdb-client >= 1.45 (#15).
- Packaging: PEP 621 `pyproject.toml` with hatchling, uv and `uv.lock`; poetry removed (#15).
- `features_metadata` and `prediction_id` are optional with a `None` default (required in pydantic 2 otherwise) (#15).
- Continuous integration runs ruff, black, bandit and the tests (InfluxDB 2.6.1 with Docker) on Python 3.12 and 3.13;
  tests are required to pass (#16).
- The test model is trained by a fixture instead of being loaded from a pickle (#15).

### Fixed
- The installed package could not be imported (absolute imports of `config`, `models`, `database_connectors`) (#14).

### Added
- `tests/db/influxdb/.env.example` for the integration tests (#14).

## [0.1.1] - 2022-12-20

Last release published on PyPI (Python 3.9, previous `DataCapture` API).
