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
  replayed, oldest first, as soon as a write succeeds again. Without `spool_dir`, these rows are dropped and counted.
- The queue is bounded (`max_queue_rows`): if it is full, the oldest batches are dropped and counted rather than
  blocking the caller or exhausting memory.
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
`write_errors`, `last_error`, `queued_rows`, `spool_files` and `storage_available`. In normal operation
`written_rows + replayed_rows + spooled pending + dropped_rows = captured_rows`; any `dropped_rows` is a measured gap in
the monitoring data. Errors and drops are also logged (logger `pulsar_data_collection`).

`pulse.flush(timeout_s)` waits until the queue is empty, for instance in a notebook before reading the data back.

Measured with InfluxDB 2.6.1 stopped for 15 s in the middle of the service: capture latency stayed at 0.8 ms (1.3 ms
maximum), the rows went to the spool and were replayed when InfluxDB restarted, with **no row lost** (3,000 / 3,000).

`mode="sync"` writes in `capture_data()` and raises on failure: use it for batch jobs, notebooks or tests that need
the data stored before going on, not in a latency-sensitive service.

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

## How data is stored

| InfluxDB concept | Content |
|---|---|
| Bucket | `login["bucket_name"]` (must exist) |
| Measurement | `{model_id}_{model_version}_input_data` |
| Tags | `model_id`, `model_version`, `data_id`, plus every key of `additional_tags` |
| Fields | every column of `data_points` and of `predictions` (numbers, booleans or strings) |
| Timestamp | row *i* of a capture is stored at `timestamp + i` milliseconds |

Consequences:

- InfluxDB keeps a single point per measurement, tags and timestamp. Each `Pulse` therefore keeps its timestamps
  strictly increasing: a capture starting at or before the end of the previous one is shifted just after it. System
  clocks are coarse (15.6 ms on Windows) and concurrent requests share timestamps; without this, rows would silently
  overwrite each other. Several processes writing the same series should add a distinguishing tag (e.g. `instance`).
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
  right after a number ("514 555 0199 $" is an amount). Some types also have words of their own: "client",
  "customer", "membre", "confirmation", "matricule", "réservation"... before a number shaped like a SIN, and "membre",
  "confirmation"... right before a phone number ("Numéro de membre 514-555-0123"); a SIN is never written after "#".
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

## Analysis results and capture statistics for dashboards

The analyses of pulsar-metrics return `MetricResults` in memory. `ResultsSink` writes them to InfluxDB, in one
measurement (`pulsar_results`, one point per result), so that dashboards can show verdicts with their intervals;
`StatsReporter` publishes the counters of a Pulse (`pulsar_capture_stats`) every few seconds.

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
