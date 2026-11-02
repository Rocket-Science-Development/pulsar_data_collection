"""Non-blocking capture: a slow or unavailable storage engine must not slow down or break predictions.

These tests replace the InfluxDB actions by a fake storage engine whose latency and failures are controlled.
"""

import threading
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

import pulsar_data_collection.pulse as pulse_module
from pulsar_data_collection.database_connectors.influxdb import InfluxdbActions
from pulsar_data_collection.exceptions import StorageConfigurationError, StorageRejectedError, StorageUnavailableError
from pulsar_data_collection.models import CaptureOptions, DataWithPrediction, PulseParameters
from pulsar_data_collection.pulse import Pulse
from pulsar_data_collection.spool import frame_from_json, frame_to_json


class FakeClient:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeActions(InfluxdbActions):
    """InfluxDB actions with a fake connection and a controllable write"""

    def __init__(self):
        self.delay_s = 0.0
        self.down = False
        self.reject = False
        self.startup_error = None
        self.frames = []
        self.lock = threading.Lock()

    def make_connection(self, **kwargs):
        if self.startup_error:
            raise self.startup_error
        return FakeClient()

    def create_client(self, **kwargs):
        return FakeClient()

    def write_frame(self, client, frame, **write_params):
        time.sleep(self.delay_s)
        if self.down:
            raise ConnectionError("storage down")
        if self.reject:
            raise StorageRejectedError("field type conflict")
        with self.lock:
            self.frames.append((frame.copy(), write_params))

    def written(self) -> pd.DataFrame:
        with self.lock:
            return pd.concat([f for f, _ in self.frames], ignore_index=True) if self.frames else pd.DataFrame()


class FakeFactory:
    def __init__(self, actions):
        self.actions = actions

    def get_database_actions(self):
        return self.actions


@pytest.fixture
def fake(monkeypatch):
    actions = FakeActions()
    monkeypatch.setattr(pulse_module, "factories", {"influxdb": FakeFactory(actions)})
    return actions


PARAMS = PulseParameters(
    model_id="m",
    model_version="1",
    data_id="d",
    reference_data_storage="ref.csv",
    target_name="y",
    storage_engine="influxdb",
    timestamp_column_name="_time",
    login={"url": "http://localhost:8086", "token": "t", "org": "o", "bucket_name": "b"},
    additional_tags={"environment": "test"},
)
START = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)


def capture(k: int, rows: int = 10) -> DataWithPrediction:
    rng = np.random.default_rng(k)
    return DataWithPrediction(
        data_points=pd.DataFrame({"x": rng.normal(size=rows), "label": [f"c{k}"] * rows}),
        predictions=pd.DataFrame({"y_pred": rng.integers(0, 2, rows)}),
        timestamp=START + timedelta(seconds=k),
        features_names=["x", "label"],
    )


def fast_options(**kwargs) -> CaptureOptions:
    return CaptureOptions(**{"flush_interval_s": 0.05, "max_retry_delay_s": 0.2, **kwargs})


def test_capture_does_not_wait_for_a_slow_storage(fake):
    fake.delay_s = 0.5
    with Pulse(PARAMS, fast_options()) as pulse:
        durations = []
        for k in range(20):
            t = time.perf_counter()
            pulse.capture_data(capture(k))
            durations.append(time.perf_counter() - t)
        assert max(durations) < 0.05  # each write takes 0.5 s
        assert pulse.flush(10)
        assert pulse.stats()["written_rows"] == 200


def test_unavailable_storage_with_spool_then_replay(fake, tmp_path):
    fake.down = True
    pulse = Pulse(PARAMS, fast_options(spool_dir=tmp_path))
    originals = [capture(k) for k in range(5)]
    t = time.perf_counter()
    for data in originals:
        pulse.capture_data(data)
    assert time.perf_counter() - t < 0.1
    assert pulse.flush(5)
    stats = pulse.stats()
    assert stats["spooled_rows"] == 50 and stats["written_rows"] == 0 and stats["dropped_rows"] == 0
    assert stats["spool_files"] >= 1 and stats["write_errors"] >= 1

    fake.down = False
    deadline = time.monotonic() + 5
    while pulse.stats()["spool_files"] and time.monotonic() < deadline:
        time.sleep(0.05)
    pulse.close()
    stats = pulse.stats()  # final counters remain available after close
    assert stats["replayed_rows"] == 50 and stats["spool_files"] == 0
    assert list(tmp_path.glob("*.json")) == []

    # every captured row reached the storage, with its timestamps and tags
    written = fake.written().sort_values("_time").reset_index(drop=True)
    expected = pd.concat(
        [InfluxdbActions.prepare_frame(d.data_points, d.predictions, d.timestamp) for d in originals], ignore_index=True
    )
    pd.testing.assert_frame_equal(written, expected, check_dtype=False)
    assert all(params["tags"]["environment"] == "test" for _, params in fake.frames)


def test_unavailable_storage_without_spool_counts_dropped_rows(fake, caplog):
    fake.down = True
    with Pulse(PARAMS, fast_options()) as pulse:
        for k in range(3):
            pulse.capture_data(capture(k))
        assert pulse.flush(5)
        stats = pulse.stats()
    assert stats["dropped_rows"] == 30 and stats["written_rows"] == 0
    assert "storage down" in stats["last_error"]
    assert "rows dropped" in caplog.text


def test_full_queue_drops_oldest_rows_without_blocking(fake):
    fake.delay_s = 1.0  # the writer is busy with the first batch
    with Pulse(PARAMS, fast_options(max_queue_rows=50, max_batch_rows=10)) as pulse:
        t = time.perf_counter()
        for k in range(30):
            pulse.capture_data(capture(k))
        assert time.perf_counter() - t < 0.2
        stats = pulse.stats()
        assert stats["dropped_rows"] > 0
        assert stats["queued_rows"] <= 50
        assert pulse.flush(15)
        stats = pulse.stats()
    assert stats["written_rows"] + stats["dropped_rows"] == stats["captured_rows"] == 300


def test_close_writes_pending_rows(fake):
    pulse = Pulse(PARAMS, CaptureOptions(flush_interval_s=60))  # no periodic flush before close
    for k in range(4):
        pulse.capture_data(capture(k))
    pulse.close()
    assert len(fake.written()) == 40
    assert pulse.db_connection.closed
    assert pulse.stats()["written_rows"] == 40
    pulse.close()  # idempotent
    with pytest.raises(RuntimeError, match="closed"):
        pulse.capture_data(capture(5))


def test_caller_dataframes_are_not_modified(fake):
    data = capture(0)
    original = data.data_points.copy()
    with Pulse(PARAMS, fast_options()) as pulse:
        pulse.capture_data(data)
    pd.testing.assert_frame_equal(data.data_points, original)


def test_sync_mode_writes_immediately_and_raises(fake):
    pulse = Pulse(PARAMS, CaptureOptions(mode="sync"))
    pulse.capture_data(capture(0))
    assert len(fake.written()) == 10
    fake.down = True
    with pytest.raises(ConnectionError):
        pulse.capture_data(capture(1))
    assert pulse.stats() == {"mode": "sync"}


def test_startup_with_unavailable_storage(fake, caplog):
    fake.startup_error = StorageUnavailableError("connection refused")
    with Pulse(PARAMS, fast_options()) as pulse:  # background mode starts anyway
        assert "unavailable at startup" in caplog.text
        pulse.capture_data(capture(0))
    for options in (CaptureOptions(strict_startup=True), CaptureOptions(mode="sync")):
        with pytest.raises(StorageUnavailableError):
            Pulse(PARAMS, options)


def test_configuration_errors_always_raise(fake):
    fake.startup_error = StorageConfigurationError("bucket not found")
    with pytest.raises(StorageConfigurationError):
        Pulse(PARAMS, fast_options())


def test_capture_options_reject_unknown_and_invalid_values():
    with pytest.raises(ValueError):
        CaptureOptions(spool=".")
    with pytest.raises(ValueError):
        CaptureOptions(max_queue_rows=0)


def test_spool_round_trip_keeps_types_and_nanoseconds():
    frame = pd.DataFrame(
        {
            "f": [1.5, np.nan],
            "i": [1, 2],
            "s": ["a", None],
            "b": [True, False],
            "_time": pd.date_range("2026-11-01T10:00:00.001234567Z", periods=2, freq="ms"),
            "naive": pd.date_range("2026-11-01", periods=2, freq="D"),
        }
    )
    meta = {"bucket_name": "b", "tags": {"model_id": "m"}}
    back, back_meta = frame_from_json(frame_to_json(frame, meta))
    pd.testing.assert_frame_equal(back, frame)
    assert back_meta == meta


def test_captures_with_the_same_timestamp_never_overwrite_each_other(fake):
    # InfluxDB keeps one point per (measurement, tags, timestamp): equal timestamps would silently lose rows
    with Pulse(PARAMS, CaptureOptions(mode="sync")) as pulse:
        for _ in range(3):
            pulse.capture_data(
                DataWithPrediction(
                    data_points=pd.DataFrame({"x": [1.0, 2.0]}),
                    predictions=None,
                    timestamp=START,  # identical, as with a coarse clock or concurrent requests
                    features_names=["x"],
                )
            )
    times = fake.written()["_time"]
    assert times.is_unique and times.is_monotonic_increasing and len(times) == 6
    assert times.iloc[0] == pd.Timestamp(START)


def test_concurrent_captures_keep_unique_timestamps(fake):
    with Pulse(PARAMS, fast_options()) as pulse:

        def worker():
            for _ in range(50):
                pulse.capture_data(
                    DataWithPrediction(
                        data_points=pd.DataFrame({"x": [1.0]}), predictions=None, timestamp=START, features_names=["x"]
                    )
                )

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert len(fake.written()) == 200 and fake.written()["_time"].is_unique


def test_rejected_rows_are_counted_not_retried(fake, tmp_path):
    # the storage refuses the data itself (e.g. a field type conflict): retrying can never succeed
    fake.reject = True
    pulse = Pulse(PARAMS, fast_options(spool_dir=tmp_path))
    pulse.capture_data(capture(1))
    assert pulse.flush(5)
    stats = pulse.stats()
    assert stats["rejected_rows"] == 10 and stats["spooled_rows"] == 0 and stats["storage_available"]
    assert "StorageRejectedError" in stats["last_error"] and list(tmp_path.glob("*.json")) == []
    fake.reject = False
    pulse.capture_data(capture(2))  # the next batches are written normally, without waiting for a backoff
    pulse.close()
    assert pulse.stats()["written_rows"] == 10


def test_a_rejected_spooled_batch_is_discarded(fake, tmp_path):
    fake.down = True
    pulse = Pulse(PARAMS, fast_options(spool_dir=tmp_path))
    pulse.capture_data(capture(1))
    assert pulse.flush(5) and len(list(tmp_path.glob("*.json"))) == 1
    fake.down, fake.reject = False, True
    deadline = time.monotonic() + 5
    while list(tmp_path.glob("*.json")) and time.monotonic() < deadline:
        time.sleep(0.05)
    pulse.close(timeout_s=1)
    assert list(tmp_path.glob("*.json")) == [] and pulse.stats()["rejected_rows"] == 10
