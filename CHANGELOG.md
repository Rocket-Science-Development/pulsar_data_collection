# Changelog

All notable changes to pulsar-data-collection. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] - privacy-rules

### Added
- `PiiDetector(locales=...)`: formats of other countries, opt-in (default `("ca",)`): `us` (`ssn_us`), `fr`
  (French phone numbers, `nir_fr` with its key), `uk` (UK phone numbers, `nino_uk`), `intl` (E.164 phone numbers,
  passports and driver's licences of any country after a context word, `drivers_licence`). Night lab LP-01 (M15).
- `PiiDetector(profiles=("finance",))`: `iban` (mod 97), `bic` (country code, after "SWIFT"/"BIC"), `bank_account`,
  `bank_routing` (transit, institution, EFT and ABA routing numbers) and `card_security_code`, opt-in. LP-01 (M16).
- `PiiDetector(profiles=("address",))`: `street_address` (civic number and street type, French and English), opt-in.
  LP-01 (M13, rules side only).
- `PiiPolicy(locales=[...], profiles=[...])`, passed to its detector; `ALL_TYPES`, `LOCALES`, `PROFILES` exported.
- `PiiDetector.scan(text, business_scope="window" | "adjacent")` and `PiiSession(output_business_scope=...)`.

### Changed
- `PiiSession` outputs: a business word ("dossier", "order"...) lowers a finding only when it is right before the
  value (`output_business_scope="adjacent"`, new default). The previous default (anywhere in the 40 characters
  before) was the defect: it hid every leak of an agent that mentions a file (0 of 2,493 counted in LP-02, C4).
  `output_business_scope="window"` restores it. The values the user gave are taken from both scans, so that the
  narrow rule creates no false leak.
- A RAMQ number with dashes or in lower case is recognized (lower case at confidence 0.55 instead of 0.9); the date
  validation is unchanged. LP-02 (M17: recall 0.80 -> 1.00).
- Words proper to a type: before a number shaped like a SIN, "client", "customer", "membre", "confirmation",
  "matricule", "réservation"... lower it, and so does "#" right before; "membre", "confirmation"... right before a
  phone number beat a generic "numéro". LP-02 (M18: SIN precision 0.81 -> 1.00 on the lab generator).
- Context-word matching uses one compiled pattern per word list (LP-02 throughput about 2.7 times higher).

## [0.9.0] - 2026-11-02

### Added
- `genai.detectors`: `InjectionDetector`, `ModelPiiDetector` (rules + GLiNER names and addresses),
  `FaithfulnessScorer` (multilingual NLI), `record_score`; scores carried to the records as `det_<name>`; optional
  `detectors` extra (transformers 4.x, torch, gliner).
- `genai.CanaryProbe`: fixed canary prompts answered several times, saved for model equality tests.

### Fixed
- A batch the storage refuses (HTTP 400 / 422, e.g. a field type conflict) is counted in `rejected_rows` and never
  retried nor spooled; it used to be retried forever while blocking the backoff.
- The background writer no longer concatenates frames whose columns differ (one write per set of columns): the
  gaps became NaN, turning integer or boolean fields (e.g. `task_success`, absent for some agent runs) into floats or
  strings that InfluxDB rejected. Found during the S3/S4 campaign.
- Agent records written with a `PiiSession` carry every role (`pii_<role>_count` = 0 when absent), so that a batch never
  mixes integers and missing values, which InfluxDB received as floats and rejected against the stored integers.

## [0.8.0] - 2026-11-01

### Added
- `pulsar_data_collection.results.ResultsSink`: writes analysis results (pulsar-metrics `MetricResults` or dicts) to
  the `pulsar_results` measurement, one point per result, with the three-way verdict as a number (`status`), the
  interval, the size and the caller's tags; and Pulse counters to `pulsar_capture_stats`.
- `StatsReporter`: publishes the counters of a Pulse periodically (background thread), never disturbing the capture.

## [0.7.0] - 2026-11-01

### Added
- `pulsar_data_collection.privacy`: detection and masking of personal information with validated rules for Canada,
  in French and English: social insurance number (Luhn), Québec health insurance number (RAMQ), credit card (Luhn,
  issuer prefix), e-mail, North American phone, Canadian postal code, Québec driver's licence, Canadian passport,
  date of birth. Context words raise or lower the confidence; findings never carry the value.
- `CaptureOptions(pii=PiiPolicy())`: personal information masked (or only counted) in the text columns before the
  rows are queued, spooled or written, with a `pii_count` column.
- `PiiSession`: personal information by role (input, context, output) on the spans of LLM applications and agents,
  and leaks (personal information in an output that the user did not provide). New fields `pii_<role>_count` and
  `pii_<role>_types` in the `agent_run`, `llm_call` and `tool_call` records.

### Fixed
- Integration tests: the Flux queries give an explicit stop, so a server clock slightly behind no longer gives an
  empty range (intermittent failure).

## [0.6.0] - 2026-11-01

### Added
- `pulsar_data_collection.genai.probes`: counterfactual probes for LLM applications and agents.
  `CounterfactualProbe` generates sets of requests that differ only by one attribute, plus placebo sets (an attribute
  that must not matter, noise floor of the instrument), sends them in a random order and collects the outcomes,
  keeping failed requests with their error. `ComparativeProbe` asks to choose between two profiles in both orders,
  with or without a tie option (position-effect control).
- Span attributes `pulsar.cf.set_id`, `pulsar.cf.variant`, `pulsar.cf.attribute`, `pulsar.cf.control`, carried to the
  `agent_run` records (`cf_set_id`, `cf_variant`, `cf_attribute`, `cf_control`).

## [0.5.1] - 2026-11-01

### Fixed
- **Rows silently lost when two captures shared a timestamp.** InfluxDB keeps one point per measurement, tags and
  timestamp; with a coarse system clock (15.6 ms on Windows) or concurrent requests, captures got identical timestamps
  and overwrote each other (e.g. 2 agent runs stored out of 3 in the GenAI end-to-end test, intermittently). Each
  `Pulse` now keeps its timestamps strictly increasing, thread-safely.
- Under pandas 3, `pd.date_range(..., periods=n, inclusive="left")` returned n - 1 timestamps and `capture_data()`
  raised `ValueError` in the prediction path.
- Spool round trip keeps the datetime resolution (pandas 3 defaults to microseconds).

### Changed
- **pandas 3 supported** (`pandas>=2.2,<4`, lock file on pandas 3.1); **Python 3.14 and 3.15** supported and tested
  (3.15 requires pandas 3).
- Continuous integration: Python 3.12 to 3.15 with pandas 3, plus Python 3.12 with pandas 2.

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
