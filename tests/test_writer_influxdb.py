"""Write path against a real InfluxDB 2: partial writes, shared writer, probe after an outage.

Runs when PULSAR_TEST_INFLUX_URL, PULSAR_TEST_INFLUX_TOKEN, PULSAR_TEST_INFLUX_ORG and PULSAR_TEST_INFLUX_BUCKET are set
(e.g. the local validation instance), skipped otherwise. Every test writes to its own measurements.
"""

import os
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from influxdb_client import InfluxDBClient

from pulsar_data_collection.models import CaptureOptions, DataWithPrediction, PulseParameters
from pulsar_data_collection.pulse import Pulse
from pulsar_data_collection.writer import SharedWriter

ENV = {k: os.environ.get(f"PULSAR_TEST_INFLUX_{k.upper()}") for k in ("url", "token", "org", "bucket")}
pytestmark = pytest.mark.skipif(not all(ENV.values()), reason="PULSAR_TEST_INFLUX_* not set")
N = 50


def params(model: str) -> PulseParameters:
    return PulseParameters(
        model_id=model,
        model_version="1",
        data_id="test",
        reference_data_storage="ref.csv",
        target_name="y",
        storage_engine="influxdb",
        timestamp_column_name="_time",
        login={"url": ENV["url"], "token": ENV["token"], "org": ENV["org"], "bucket_name": ENV["bucket"]},
    )


def stored(model: str, field: str = "seq") -> int:
    query = f"""
from(bucket: "{ENV['bucket']}") |> range(start: -1d, stop: 1d)
  |> filter(fn: (r) => r._measurement == "{model}_1_input_data" and r._field == "{field}")
  |> group() |> count()"""
    with InfluxDBClient(url=ENV["url"], token=ENV["token"], org=ENV["org"], timeout=30_000) as client:
        tables = client.query_api().query(query)
    return sum(record.get_value() for table in tables for record in table.records)


def batch(seq0: int, f0=None) -> pd.DataFrame:
    frame = pd.DataFrame(np.random.default_rng(seq0).normal(size=(N, 3)), columns=["f0", "f1", "f2"])
    if f0 is not None:
        frame["f0"] = f0
    frame["seq"] = np.arange(seq0, seq0 + N, dtype="int64")
    return frame


def capture(pulse, frame, start):
    pulse.capture_data(
        DataWithPrediction(data_points=frame, predictions=None, timestamp=start, features_names=list(frame.columns))
    )


def name() -> str:
    return f"wscale_{uuid.uuid4().hex[:10]}"


def test_partial_writes_are_counted_exactly():
    model = name()
    start = datetime.now(timezone.utc) - timedelta(minutes=5)
    with Pulse(params(model), CaptureOptions(flush_interval_s=0.5)) as pulse:
        capture(pulse, batch(0), start)
        assert pulse.flush(10)
        for k, frame in enumerate([batch(100), batch(200, f0="high"), batch(300)]):
            capture(pulse, frame, start + timedelta(seconds=k + 1))
        mixed = batch(400)
        mixed["f0"] = mixed["f0"].astype(object)
        mixed.loc[:4, "f0"] = "n/a"
        capture(pulse, mixed, start + timedelta(seconds=10))
        assert pulse.flush(20)
        stats = pulse.stats()
    assert stored(model) == 50 + 100 + 45
    assert stats["written_rows"] == 195 and stats["rejected_rows"] == 55
    assert stats["captured_rows"] == stats["written_rows"] + stats["rejected_rows"]


def test_shared_writer_merges_models_into_shared_requests():
    writer = SharedWriter(flush_interval_s=0.2, concurrency=2)
    models = [name() for _ in range(20)]
    start = datetime.now(timezone.utc) - timedelta(minutes=5)
    pulses = [Pulse(params(m), CaptureOptions(), writer=writer) for m in models]
    try:
        assert len({id(p.db_connection) for p in pulses}) == 1
        for k, pulse in enumerate(pulses):
            capture(pulse, batch(k * 100), start)
        assert writer.flush(20)
        assert all(p.stats()["written_rows"] == N for p in pulses)
    finally:
        for pulse in pulses:
            pulse.close()
        writer.close()
    assert sum(stored(m) for m in models) == 20 * N


def test_no_rows_ahead_of_the_clock_at_high_rate():
    model = name()
    start = datetime.now(timezone.utc)
    with Pulse(params(model), CaptureOptions(mode="sync")) as pulse:
        for _ in range(20):
            capture(pulse, pd.DataFrame({"seq": np.arange(1000, dtype="int64")}), start)
    query_stop_now = f"""
from(bucket: "{ENV['bucket']}") |> range(start: -1h, stop: 5s)
  |> filter(fn: (r) => r._measurement == "{model}_1_input_data" and r._field == "seq") |> group() |> count()"""
    with InfluxDBClient(url=ENV["url"], token=ENV["token"], org=ENV["org"]) as client:
        tables = client.query_api().query(query_stop_now)
    assert (
        sum(r.get_value() for t in tables for r in t.records) == 20_000
    )  # 1 ms spacing: up to 19 s ahead; 5 s of margin for the clock skew of Docker
