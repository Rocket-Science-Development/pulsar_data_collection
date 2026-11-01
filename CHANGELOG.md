# Changelog

All notable changes to pulsar-data-collection. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
