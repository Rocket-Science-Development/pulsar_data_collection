# Changelog

All notable changes to pulsar-data-collection. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.11.0] - 2026-11-04

Changes recommended by the scalability rerun and the P2 lab of 3 November 2026 (N1, N3, M32, m18, m19, SC-09,
LA-01, LA-02, LB-02).

### Added

**Writing and spool**

- `encode_lines()`: line protocol built column by column, with the same lines as the influxdb-client serializer (still
  used for unusual frames). A 50-row frame encodes in 0.9 ms instead of 5.3 ms; a `SharedWriter` at 100 models drains
  24.8k rows/s instead of 5.9k (local measurement), as fast as private writers (N1, N3).
- `StorageWriteUncertainError`, parent of `StorageWriteTimeoutError` and of the new `StorageConnectionResetError`.
- README: choosing private writers or `SharedWriter`, the per-process ceiling, the loss window on kill -9
  (`max_queue_rows` plus one batch) and `durability="wal"`, `spool_compression="gzip"`, and the recommended
  `replay_max_rows_per_s` (2 to 6 times the Pulse's live rate; the default stays unlimited).

**Detectors (SC-09)**

- `InjectionDetector` / `TextClassifierDetector`: `score_details(text)` returns a `ClassifierScore` (`n_tokens`,
  `tokens_seen`, `windows`, `truncated`, `window_scores`); `long_text="sliding"` (opt-in) scores long texts on
  overlapping windows of the horizon minus 8 tokens (score = maximum; refit thresholds in that mode); `horizon`, `windows()`.
- `dtype=` on every detector; `FaithfulnessScorer` also passes `**pipeline_kwargs`.
- `ModelPiiDetector`: `window_tokens` / `overlap_tokens` (GLiNER tokens); `scan()` returns a `PiiScan` (a list) with
  `windows` and `truncated_windows`; `scan_many(texts)` batches short texts (2.3x on CPU).
- `FaithfulnessScorer(premise_split="sentences", premise_sentences=2, top_k_premises=4)` (opt-in): each sentence is
  judged against the closest groups of sentences, so its support no longer depends on where the evidence sits.

**Agents and probes (LA-01, LA-02, LB-02)**

- `tool_fields="called" | "allowed" | "top_k" | "all" | "none"` and `tool_fields_k` on `TrajectoryAssembler`,
  `PulsarSpanExporter` and `agent_run_record`.
- Label-free quality fields: `tool_args_checked`, `tool_args_invalid`, `tool_args_invalid_rate`,
  `tool_args_missing_required`, `tool_args_type_errors`, `tool_args_string_numbers`, `tool_args_unparsable` (arguments
  checked against the tool JSON schemas from `gen_ai.tool.definitions`, `agent_span(available_tools=<definitions>)` or
  `llm.request.functions.*`), and `first_step_no_tool`.
- Rich compliance trace for monitors declaring `accepts_rich_trace = True`: tool calls with their parsed arguments and
  status, and conversation events `pulsar.event.<name>` recorded with `record_event()`, in time order
  (`pulsar.sequence`, also set by the `OpenRunTracker` on the tool spans of other instrumentations, breaks timestamp
  ties). Argument values are never written. `not_evaluable_rules` when the monitor
  exposes `not_evaluable`.
- `ComparativeProbe`: `answer_status`, `unusable`, `labels`, `instrument` columns; `refusal=` (default `is_refusal`,
  French and English); neutral label rotation (`labels="neutral"`, `label_choice()`); `run_loglik(probes, loglik)`,
  skipped cleanly without logprobs.

### Changed

- `agent_run` writes per-tool counters only for the tools called in the run (`tool_fields="called"`); a missing counter
  means 0. Before, every catalogue tool got four fields even at 0 (2308 fields for 580 tools); `"all"` restores it.
- `ComparativeProbe` records a refusal as `choice="refusal"` (it was `None`).
- Models load in float32 on CPU when no dtype is given (the NLI of `FaithfulnessScorer`, stored in float16, is about
  10x faster, scores unchanged to about 1e-4); the checkpoint dtype is kept on GPU.
- `ModelPiiDetector` windows default to 384/96 GLiNER tokens instead of 250/50 words; `window_words` is still accepted.
- `InjectionDetector` / `TextClassifierDetector` warn once per instance when they truncate a text.
- A write whose connection is cut after sending counts in `uncertain_rows` (or the spool), not in `dropped_rows`; in
  sync mode it raises `StorageConnectionResetError` instead of urllib3's `ProtocolError`. `write_timeouts` counts only
  timeouts.

### Fixed

- A field whose name is part of the timestamp column's name (`time`, `e`) made the whole batch rejected, and columns
  such as `_t` were dropped.
- Spool file names are strictly increasing within a process: with a coarse clock, `drop_oldest` could evict a recent
  batch and replay did not follow capture order.
- `ModelPiiDetector`'s 250-word windows exceeded GLiNER's 384 tokens, so the end of long texts was never read; its
  docstring said "384 words".

## [0.10.0] - 2026-11-02

Fixes for the limits confirmed by the night lab of 2 November 2026: C3 (data-collection side), C4, M10 to M25,
M27 to M38, m12 to m15 and m18 to m24, plus the findings of the review of these fixes.

### Added

**Writing, spool and stats (SC-01 to SC-14)**

- `SharedWriter` (`pulsar_data_collection.writer`): one writer thread and one HTTP client for every Pulse of a process,
  rows of several models in the same line-protocol requests, bucket checked once per process, up to `concurrency`
  requests in flight. `CaptureOptions(shared_writer=True)` uses `SharedWriter.default()`; `Pulse(..., writer=...)`
  takes an explicit one (M31, SC-01 / SC-02).
- `CaptureOptions(write_concurrency=N)`: several requests in flight for a private writer; 37k rows/s instead of 21k for
  one Pulse with N = 4 against a local InfluxDB (M34).
- `CaptureOptions(probe_interval_s=1.0)`: an unavailable InfluxDB is probed (`/ping`) during the back-off, so that
  writing resumes about a second after an outage instead of up to `max_retry_delay_s` later (M35).
- `CaptureOptions(durability="wal")`: write-ahead copy of each batch in `spool_dir`, deleted once stored and replayed
  after a crash; `kill -9` with the storage available no longer loses the in-memory queue. Durability modes documented
  (M36).
- `CaptureOptions(spool_max_bytes, spool_eviction, spool_compression="gzip")`: size limit of the spool with eviction of
  the oldest (or refusal of the newest) batches, counted in `spool_evicted_rows` and `dropped_rows`; gzip spool files;
  compact JSON (m19).
- `CaptureOptions(replay_max_rows_per_s)`: throttled spool replay; replayed batches are also merged into requests of up
  to `max_batch_rows` rows (m18).
- `CaptureOptions(row_spacing_ns)`, `InfluxdbActions.prepare_frame(row_spacing_ns=...)`.
- Counters `uncertain_rows`, `write_timeouts`, `writer_lag_s`, `last_write_ms`, `max_write_ms`, `spool_bytes`,
  `spool_evicted_rows` in `Pulse.stats()`.
- `StatsReporter(None, sink)` + `add(pulse, tags)`: one reporter thread and one request per interval for every Pulse of
  the process; `ResultsSink.write_stats_many` (m22).
- `StatsReporter(process_health=True)` and `results.process_health()`: threads, and with psutil RSS, OS threads, open
  descriptors (handles on Windows), CPU, plus the writer lag, queued rows and write latency, in the measurement
  `pulsar_process_health` (m23).
- `ResultsSink` writes the optional `MetricResults` fields when present: `n_labelled`, `label_coverage`,
  `missing_share_current`, `missing_share_reference` (always floats), `reliable` (boolean), and the string `status` as
  `result_status`, never colliding with the numeric verdict `status`.
- `StorageWriteTimeoutError`; `StorageRejectedError.dropped`; `InfluxdbActions.serialize_frame`, `write_lines`, `ping`,
  `make_shared_connection`, `release_connection`.

**Personal information (LP-01, LP-02)**

- `PiiDetector(locales=...)`: formats of other countries, opt-in (default `("ca",)`): `us` (`ssn_us`), `fr`
  (French phone numbers, `nir_fr` with its key), `uk` (UK phone numbers, `nino_uk`), `intl` (E.164 phone numbers,
  passports and driver's licences of any country after a context word, `drivers_licence`). Night lab LP-01 (M15).
- `PiiDetector(profiles=("finance",))`: `iban` (mod 97), `bic` (country code, after "SWIFT"/"BIC"), `bank_account`,
  `bank_routing` (transit, institution, EFT and ABA routing numbers) and `card_security_code`, opt-in. LP-01 (M16).
- `PiiDetector(profiles=("address",))`: `street_address` (civic number and street type, French and English), opt-in.
  LP-01 (M13, rules side only).
- `PiiPolicy(locales=[...], profiles=[...])`, passed to its detector; `ALL_TYPES`, `LOCALES`, `PROFILES` exported.
- `PiiDetector.scan(text, business_scope="window" | "adjacent")` and `PiiSession(output_business_scope=...)`.

**Detectors and canary (LI-01, LI-02, LP-01, LR-01, LR-02, LM-01)**

- `InjectionDetector`: per-language alert thresholds (`threshold`, `thresholds`, `language`), `calibrate()` on a benign
  A/A reference (returns the false-alarm rate and the recall it costs), `flag()`, `record()` (writes
  `pulsar.detector.injection_flag`), `score_many()`, `calibrator`, `from_logits` (double-precision scores: the
  single-precision ones saturate and tie near 1), `french_benign_calibration_set()`, `guess_language()` (M10, M11, M12).
- `calibrate_threshold` (A/A order statistic, both directions) and `ScoreCalibrator` (isotonic or Platt), as the
  scores are ranks, not probabilities (m12, m15).
- `TextClassifierDetector` (any Hugging Face text classifier) and `ContentSafetyDetector`, a slot without a default
  model (m13).
- `ModelPiiDetector`: `device` (m14), per-label `thresholds`, overlapping windows (`window_words`, `overlap_words`)
  so that entities beyond GLiNER's 384-word window are found (M14), batched model calls, `locales` and `profiles`
  passed to the default `PiiDetector`, `scan(text, business_scope=...)` forwarded to rules that accept it.
- `FaithfulnessScorer`: overlapping premise windows (`window_tokens`, `stride_tokens`) with the best window per
  sentence (M20), `max_sentences` as a constructor parameter (`None` = all) and `aggregate="min"` (M21), lexical
  shortcut for verbatim copies (`lexical_threshold`) and `max_claim_words` (M22), `details()` returning a
  `FaithfulnessResult` (coverage, windows, truncation), `record()` (M19).
- `CanaryProbe.preset()`, `planned_answers`, `required_answers()`, a warning under 480 answers (M23).
- README: known limits of the detectors with the lab figures (M10-M14, M19-M23, m12-m15).

**Agent tracing (LS-01, LA-03, LI-03)**

- `agent_run`: per-tool counters `tool_<tool>_calls` / `_errors` / `_timeouts` / `_soft_failures` (every tool of
  `allowed_tools` and of the catalogue written, 0 when not called), `tool_errors`, `tool_timeouts`,
  `tool_soft_failures`, `soft_failure_rate`, `max_tool_error_rate`, `tool_status_counts` (lab M27, M28).
  `tool_call`: `status`, `timeout`, `soft_failure`. A failed call is a timeout when its `error.type` names one.
- `record_tool_result(span, output, status, valid, validator)`: records `pulsar.tool.status` and
  `pulsar.tool.output_valid` without writing the output; `retry`, `invalid`, `empty`, `partial`, `garbage` and invalid
  outputs are soft failures (M28).
- `agent_span(..., available_tools=[...])` (`pulsar.agent.available_tools`): `tool_catalog_size`, `tool_catalog_hash`,
  `catalog_missing_tools`, `missing_tools`; `unavailable_tool_calls` for calls outside the catalogue or with status
  `unavailable` / `unknown_tool` / `not_found` (M30).
- `repeated_calls_window`, `max_same_tool_calls`, `tool_cycle_repeats`; parameters `loop_window` (4) and
  `loop_min_repeats` (2) (M29).
- `compliance` and `rules_version` parameters (a pulsar-metrics `ComplianceMonitor`, duck-typed, or a callable):
  `violated_rules`, `violations`, `critical_violations`, `rules_version` in `agent_run`. No dependency on pulsar-metrics
  (C3, data-collection side).
- `OpenRunTracker` / `exporter.run_tracker()`: span processor seeing agent spans start. A run open for longer than
  `max_run_s` (600 s) is reported once by an `agent_run` with `termination="incomplete"` and `incomplete=True`; without
  the tracker, from the first ended span of its trace. `exporter.check_open_runs()` (M25).
- `agent_run.incomplete` (False for complete runs); `exporter.stats()` / `assembler.stats()`: `partial_traces_dropped`,
  `partial_spans_dropped`, `incomplete_runs`, `compliance_errors`, `pending_traces`, `open_runs` (M24).
- `SpanView.parent_remote`, filled from the OpenTelemetry span: a span under a remote parent is a local root.

### Changed

**Writing, spool and stats (SC-01 to SC-14)**

- **Default changed:** the rows of one capture are 1 µs apart instead of 1 ms. The old spacing was the defect: above
  1,000 rows/s a model ran ahead of the clock (+2,760 s after 3 min at 1M rows/min) and queries stopping at `now()`
  missed its rows (M33).
- Writes go through the line protocol (same influxdb-client serializer) when the storage engine is the built-in
  InfluxDB one; a subclass that overrides `write_frame` keeps frame-by-frame writes.

**Personal information (LP-01, LP-02)**

- `PiiSession` outputs: a business word ("dossier", "order"...) lowers a finding only when it is right before the
  value (`output_business_scope="adjacent"`, new default). The previous default (anywhere in the 40 characters
  before) was the defect: it hid every leak of an agent that mentions a file (0 of 2,493 counted in LP-02, C4).
  `output_business_scope="window"` restores it. The values the user gave are taken from both scans, so that the
  narrow rule creates no false leak.
- A RAMQ number with dashes or in lower case is recognized (lower case at confidence 0.55 instead of 0.9); the date
  validation is unchanged. LP-02 (M17: recall 0.80 -> 1.00).
- Words proper to a type, counted only right before the value whatever the scope: "client", "customer", "membre",
  "confirmation", "réservation"... lower a number shaped like a SIN, and so does "#"; "membre", "confirmation"...
  before a phone number beat a generic "numéro". Employee words are not among them (employers hold SINs). LP-02
  (M18: SIN precision 0.81 -> 1.00 on the lab generator).
- Context-word matching uses one compiled pattern per word list (LP-02 throughput about 2.7 times higher).

**Detectors and canary (LI-01, LI-02, LP-01, LR-01, LR-02, LM-01)**

- `FaithfulnessScorer` cuts the passages into windows by default: the former single premise was silently truncated at
  512 tokens (64.9 % of RAGTruth premises), which is the defect; `window_tokens=None` restores it. Verbatim sentences
  now score 1.0 (`lexical_threshold=None` restores the model score). Scores of long contexts rise.
- `ModelPiiDetector` merges by default with `merge="enclosing"`: a model address holding a rule-found postal code or
  phone number replaces them (it used to be dropped), and overlapping addresses are merged; `merge="rules"` restores
  the former rule.
- `CanaryProbe` defaults to 48 prompts x 10 answers (was 12 x 5, which did not detect a quantisation change);
  `DEFAULT_PROMPTS` keeps the former 12 first; `CanaryProbe.preset("quick")` gives the former set.

**Agent tracing (LS-01, LA-03, LI-03)**

- **A trace is no longer assembled before its root arrives** (`wait_for_root=True`, new default). The former default
  assembled a trace after `trace_timeout_s` (30 s) of inactivity even without its root: every run longer than 30 s lost
  its LLM and tool calls in silence (`tool_calls=0`, lab M24). A trace without root is now given up after
  `orphan_timeout_s` (3600 s) or on `max_pending_traces` overflow, and counted. `wait_for_root=False` restores the
  former behaviour. An agent span under a remote parent is assembled as soon as it ends (it used to wait 30 s).
  Consequence: the agent runs under a long-lived *local* root (a chat session span that lasts for the whole
  conversation, for instance) now wait for that root, up to `orphan_timeout_s` (3600 s) of inactivity, with up to
  `max_pending_traces` (10,000) traces in memory. For chat-style roots, lower `orphan_timeout_s` (e.g. 300 s) or end
  the session span per turn.
- Without `OpenRunTracker`, an open run is inferred only from ended spans that carry `gen_ai.agent.name` and are not
  below an agent span already received: a trace of LLM calls without agent, or a completed run under a session root,
  never produces an `incomplete` record. `OpenRunTracker` queues every span start (attributes for agent spans only), so
  that a sub-agent below an intermediate span (a tool that delegates) is part of its run.
- **`loop_detected`** now counts the calls whose normalized tool and arguments (`36000` = `36000.0` = `"36000"`) appeared
  among the 4 previous calls, True from 2 such calls; it used to see only consecutive calls identical byte for byte, so
  ping-pong and type variations were never flagged (M29). `repeated_calls` keeps its former meaning.

### Fixed

**Writing, spool and stats (SC-01 to SC-14)**

- Frames are concatenated only when they share their columns **and dtypes**, and object columns mixing numbers and
  strings are split by kind: valid rows sharing a write with conflicting rows are stored (lab: 100 of 250 valid rows
  lost) (M37, SC-14).
- Exact counters on refused writes: on a 422 partial write the `dropped=N` of InfluxDB splits the batch between
  `written_rows` and `rejected_rows`; without a count, or when the request held several batches or models, the rows are
  written again batch by batch, then in halves down to single rows (M37, M38).
- A write that timed out is no longer counted as lost: spooled and written again (idempotent) with `spool_dir`, counted
  in `uncertain_rows` without it (M32).
- A capture larger than `max_queue_rows` keeps only its newest rows; the others are dropped and counted (m20).
- A spool `.tmp` file left by a crash is recovered at startup when complete, deleted otherwise (m24).
- Review fixes: a batch that cannot be serialized (e.g. integer column names) only rejects its own rows, without
  backing off the shared destination nor dropping the other models' rows; a spooled batch in that case is renamed
  `.bad` (no longer counted in the spool size) instead of blocking the replay. A write-ahead copy refused by a full
  spool no longer counts as an eviction (the batch is kept in memory only, `memory_only_rows`) and never evicts spooled
  batches; files being replayed are never evicted; a failed write of rows partly without write-ahead copy counts only
  what reaches the disk. A replay rate below 1 row/s replays; rows taken by the writer are always resolved after an
  unexpected error; with several requests in flight, an older success no longer clears a newer back-off.

**Personal information (LP-01, LP-02)**

- The e-mail pattern starts a match only at the start of a run of local-part characters: a scan was quadratic on
  long runs without "@" (40,000 characters took about 2.4 s), it is now linear (under 0.03 s).

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
