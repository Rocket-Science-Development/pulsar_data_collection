"""Write path at scale: shared writer, concurrency, exact counters on refused writes, timeouts, recovery after an
outage, write-ahead spool, spool limits and crash leftovers, shared stats reporter and process health.

The storage is a fake that parses the line protocol like InfluxDB: one point per (measurement, tags, timestamp), field
types fixed by the first write, partial writes refused with HTTP 422 "dropped=N".
"""

import gzip
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pulsar_data_collection.pulse as pulse_module
from pulsar_data_collection.database_connectors.influxdb import InfluxdbActions, _dropped, _raise_write_error
from pulsar_data_collection.exceptions import StorageRejectedError, StorageWriteTimeoutError
from pulsar_data_collection.models import CaptureOptions, DataWithPrediction, PulseParameters
from pulsar_data_collection.pulse import Pulse
from pulsar_data_collection.results import StatsReporter, process_health
from pulsar_data_collection.spool import Spool, frame_to_json
from pulsar_data_collection.writer import SharedWriter, Target, _by_columns, split_mixed_objects


# ---------------------------------------------------------------------------------------------------------- fake store
def _parse(line: str):
    head, fields, ts = line.rsplit(" ", 2)
    out = {}
    for part in fields.split(","):
        key, value = part.split("=", 1)
        if value.startswith('"'):
            out[key] = ("string", value.strip('"'))
        elif value in ("true", "false"):
            out[key] = ("boolean", value == "true")
        elif value.endswith("i"):
            out[key] = ("integer", int(value[:-1]))
        else:
            out[key] = ("float", float(value))
    measurement = head.split(",")[0]
    return measurement, head, int(ts), out


class FakeStore:
    """InfluxDB semantics for the tests: idempotent points, typed fields, partial writes"""

    def __init__(self):
        self.points = {}
        self.types = {}
        self.requests = 0
        self.lock = threading.Lock()
        self.down = False
        self.timeout_after_store = False
        self.delay_s = 0.0
        self.whole_rejection = False  # 400-like: refuse the whole request, no count

    def write(self, lines):
        time.sleep(self.delay_s)
        with self.lock:
            self.requests += 1
            if self.down:
                raise ConnectionError("storage down")
            parsed = [_parse(line) for line in lines]
            bad = [p for p in parsed if any(self.types.get((p[0], k), t) != t for k, (t, _) in p[3].items())]
            if bad and self.whole_rejection:
                raise StorageRejectedError("unable to parse")
            for measurement, head, ts, fields in parsed:
                if any(self.types.get((measurement, k), t) != t for k, (t, _) in fields.items()):
                    continue
                for k, (t, _) in fields.items():
                    self.types.setdefault((measurement, k), t)
                self.points[(head, ts)] = fields
            if bad:
                raise StorageRejectedError(f"partial write: field type conflict dropped={len(bad)}", dropped=len(bad))
        if self.timeout_after_store:
            raise StorageWriteTimeoutError("read timed out")

    def count(self, measurement_prefix=""):
        with self.lock:
            return sum(1 for head, _ in self.points if head.startswith(measurement_prefix))


class FakeClient:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class LineActions(InfluxdbActions):
    """Real serialization, fake transport"""

    def __init__(self, store):
        self.store = store
        self.checks = 0
        self.clients = 0
        self.pings = 0

    def make_connection(self, **kwargs):
        self.checks += 1
        self.clients += 1
        return FakeClient()

    def create_client(self, **kwargs):
        self.clients += 1
        return FakeClient()

    _shared = None

    def make_shared_connection(self, **kwargs):
        if self._shared is None:
            self._shared = self.make_connection(**kwargs)
        return self._shared

    def create_shared_client(self, **kwargs):
        return self.make_shared_connection(**kwargs)

    def release_connection(self, client):
        pass

    def write_lines(self, client, bucket_name, lines):
        self.store.write(lines)

    def ping(self, client):
        self.pings += 1
        return not self.store.down


class FakeFactory:
    def __init__(self, actions):
        self.actions = actions

    def get_database_actions(self):
        return self.actions


@pytest.fixture
def store(monkeypatch):
    store = FakeStore()
    actions = LineActions(store)
    store.actions = actions
    monkeypatch.setattr(pulse_module, "factories", {"influxdb": FakeFactory(actions)})
    return store


def params(model="m") -> PulseParameters:
    return PulseParameters(
        model_id=model,
        model_version="1",
        data_id="d",
        reference_data_storage="ref.csv",
        target_name="y",
        storage_engine="influxdb",
        timestamp_column_name="_time",
        login={"url": "http://localhost:8086", "token": "t", "org": "o", "bucket_name": "b"},
    )


START = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
N = 50


def batch(seq0: int, f0=None, extra=None) -> pd.DataFrame:
    rng = np.random.default_rng(seq0)
    frame = pd.DataFrame(rng.normal(size=(N, 3)), columns=["f0", "f1", "f2"])
    if f0 is not None:
        frame["f0"] = f0
    for key, value in (extra or {}).items():
        frame[key] = value
    frame["seq"] = np.arange(seq0, seq0 + N, dtype="int64")
    return frame


def capture(pulse, frame, k=0):
    pulse.capture_data(
        DataWithPrediction(
            data_points=frame, predictions=None, timestamp=START + timedelta(seconds=k), features_names=list(frame.columns)
        )
    )


def options(**kwargs) -> CaptureOptions:
    return CaptureOptions(**{"flush_interval_s": 0.05, "max_retry_delay_s": 0.2, **kwargs})


def accounted(stats) -> int:
    return sum(stats[k] for k in ("written_rows", "spooled_rows", "dropped_rows", "rejected_rows", "uncertain_rows"))


# ---------------------------------------------------------------------------------------------------------- M33
def test_rows_are_spaced_one_microsecond_not_one_millisecond():
    frame = InfluxdbActions.prepare_frame(pd.DataFrame({"x": np.arange(5000.0)}), None, START)
    assert frame["_time"].iloc[1] - frame["_time"].iloc[0] == pd.Timedelta(1, "us")
    assert frame["_time"].iloc[-1] - frame["_time"].iloc[0] < pd.Timedelta(10, "ms")  # 1 ms spacing: 5 s ahead


def test_a_fast_model_does_not_run_ahead_of_the_clock(store):
    # 100 captures of 1000 rows in the same second: with 1 ms spacing the last row was 100 s in the future
    with Pulse(params(), CaptureOptions(mode="sync")) as pulse:
        for _ in range(100):
            capture(pulse, pd.DataFrame({"x": np.arange(1000.0)}))
    last = max(ts for _, ts in store.points)
    assert last - int(pd.Timestamp(START).value) < 1e9  # under 1 s ahead
    assert store.count() == 100_000


def test_row_spacing_is_configurable(store):
    with Pulse(params(), CaptureOptions(mode="sync", row_spacing_ns=1)) as pulse:
        capture(pulse, pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
    times = sorted(ts for _, ts in store.points)
    assert np.diff(times).tolist() == [1, 1]


# ---------------------------------------------------------------------------------------------------------- M37 / M38
def test_grouping_keeps_frames_of_different_dtypes_apart():
    valid, conflicting = batch(0), batch(50, f0="high")
    groups = _by_columns([valid, conflicting, batch(100)])
    assert len(groups) == 2 and groups[0]["f0"].dtype == "float64" and len(groups[0]) == 100


def test_valid_rows_sharing_a_write_with_conflicting_rows_are_stored(store):
    with Pulse(params(), options(flush_interval_s=0.5)) as pulse:  # establishes f0 as a float field
        capture(pulse, batch(0))
        assert pulse.flush(5)
        for k, frame in enumerate([batch(100), batch(200, f0="high"), batch(300)]):
            capture(pulse, frame, k + 1)  # the three land in the same write
        assert pulse.flush(5)
        stats = pulse.stats()
    assert store.count() == 150  # every valid row (the lab lost 100 of them)
    assert stats["written_rows"] == 150 and stats["rejected_rows"] == 50 and stats["spooled_rows"] == 0
    assert accounted(stats) == stats["captured_rows"]


def test_mixed_values_in_one_column_only_lose_the_conflicting_rows(store):
    with Pulse(params(), options()) as pulse:
        capture(pulse, batch(0))
        assert pulse.flush(5)
        mixed = batch(100)
        mixed["f0"] = mixed["f0"].astype(object)
        mixed.loc[:4, "f0"] = "n/a"
        capture(pulse, mixed, 1)
        assert pulse.flush(5)
        stats = pulse.stats()
    assert store.count() == 50 + 45  # the 45 float rows are stored, only the 5 strings are refused
    assert stats["written_rows"] == 95 and stats["rejected_rows"] == 5


def test_split_mixed_objects_converts_each_part():
    frame = pd.DataFrame({"v": pd.Series([1.5, "a", 2, None], dtype=object), "n": [1, 2, 3, 4]})
    parts = split_mixed_objects(frame)
    assert sorted(len(p) for p in parts) == [1, 1, 2]
    numeric = next(p for p in parts if len(p) == 2)
    assert numeric["v"].dtype == "float64"


def test_partial_write_counters_match_the_store(store):
    # an integer field receives a NaN: the batch becomes float; the row without the field is accepted, 49 are refused
    with Pulse(params(), options()) as pulse:
        capture(pulse, batch(0, extra={"count": np.arange(N, dtype="int64")}))
        assert pulse.flush(5)
        with_nan = np.arange(N, dtype="float64")
        with_nan[7] = np.nan
        capture(pulse, batch(100, extra={"count": with_nan}), 1)
        assert pulse.flush(5)
        stats = pulse.stats()
    assert store.count() == 51
    assert stats["written_rows"] == 51 and stats["rejected_rows"] == 49  # used to be 50 written, 50 rejected


def test_a_whole_refused_request_is_split_by_batch_then_by_row(store):
    store.whole_rejection = True  # no dropped count: nothing is stored, the writer must find the bad rows
    with Pulse(params(), options(flush_interval_s=0.5)) as pulse:
        capture(pulse, batch(0))
        assert pulse.flush(5)
        mixed = batch(100)
        mixed["f0"] = mixed["f0"].astype(object)
        mixed.loc[10, "f0"] = "n/a"  # one bad value: its own part after the object split
        capture(pulse, batch(200), 1)
        capture(pulse, batch(300, f0="x"), 2)
        capture(pulse, batch(400), 3)
        assert pulse.flush(10)
        stats = pulse.stats()
    assert store.count() == 50 + 100
    assert stats["written_rows"] == 150 and stats["rejected_rows"] == 50
    assert stats["storage_available"]


def test_rejections_of_several_models_in_one_request_are_attributed(store):
    writer = SharedWriter(flush_interval_s=0.3, concurrency=1)
    try:
        a = Pulse(params("a"), options(), writer=writer)
        b = Pulse(params("b"), options(), writer=writer)
        capture(a, batch(0))
        capture(b, batch(0))
        assert writer.flush(5)
        capture(a, batch(100, f0="bad"), 1)
        capture(b, batch(100), 1)
        assert writer.flush(5)
        sa, sb = a.stats(), b.stats()
        a.close()
        b.close()
    finally:
        writer.close()
    assert sa["rejected_rows"] == 50 and sa["written_rows"] == 50
    assert sb["rejected_rows"] == 0 and sb["written_rows"] == 100


def test_dropped_count_is_read_from_the_422_body():
    from influxdb_client.rest import ApiException

    body = '{"code":"unprocessable entity","message":"partial write: field type conflict ... dropped=49"}'
    assert _dropped(body) == 49 and _dropped("no count") is None
    error = ApiException(status=422)
    error.body = body
    with pytest.raises(StorageRejectedError) as raised:
        _raise_write_error(error, 50)
    assert raised.value.dropped == 49


# ---------------------------------------------------------------------------------------------------------- M32
def test_timeouts_are_uncertain_not_lost(store):
    store.timeout_after_store = True  # InfluxDB stored the rows but the answer came too late
    with Pulse(params(), options(probe_interval_s=None)) as pulse:
        capture(pulse, batch(0))
        assert pulse.flush(5)
        stats = pulse.stats()
    assert stats["dropped_rows"] == 0 and stats["uncertain_rows"] == 50 and stats["write_timeouts"] == 1
    assert store.count() == 50


def test_timed_out_rows_are_spooled_and_replayed_without_duplicates(store, tmp_path):
    store.timeout_after_store = True
    pulse = Pulse(params(), options(spool_dir=tmp_path))
    capture(pulse, batch(0))
    assert pulse.flush(5)
    assert pulse.stats()["spooled_rows"] == 50
    store.timeout_after_store = False
    deadline = time.monotonic() + 5
    while pulse.stats()["spool_files"] and time.monotonic() < deadline:
        time.sleep(0.05)
    pulse.close()
    assert pulse.stats()["replayed_rows"] == 50 and store.count() == 50  # idempotent: same points overwritten


def test_read_timeouts_are_recognised():
    from urllib3.exceptions import MaxRetryError, ReadTimeoutError

    error = MaxRetryError(None, "/api/v2/write", ReadTimeoutError(None, "/api/v2/write", "Read timed out. (read timeout=5)"))
    with pytest.raises(StorageWriteTimeoutError):
        _raise_write_error(error, 10)
    with pytest.raises(ConnectionError):
        _raise_write_error(ConnectionError("refused"), 10)


# ---------------------------------------------------------------------------------------------------------- M35
def test_writing_resumes_within_a_probe_interval_after_an_outage(store):
    store.down = True
    pulse = Pulse(params(), options(max_retry_delay_s=60, probe_interval_s=0.2))
    for k in range(8):  # several failures: the back-off reaches tens of seconds
        capture(pulse, batch(k * 100), k)
        assert pulse.flush(5)
        writer = pulse._writer._channel.endpoint
        writer.retry_at = 0.0  # force the next attempt, so that the delay keeps growing
    capture(pulse, batch(5000), 99)
    assert pulse.flush(5)
    assert pulse._writer._channel.endpoint.failures >= 8  # next attempt in 60 s without the probe
    store.down = False
    time.sleep(0.5)
    started = time.monotonic()
    capture(pulse, batch(9000), 100)
    deadline = time.monotonic() + 3
    while pulse.stats()["written_rows"] < 50 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert pulse.stats()["written_rows"] == 50 and time.monotonic() - started < 1.5
    assert store.actions.pings >= 1
    pulse.close(timeout_s=1)


# ---------------------------------------------------------------------------------------------------------- M36
CRASH_SCRIPT = """
import sys, time, threading
import pandas as pd
sys.path.insert(0, sys.argv[2])
from pulsar_data_collection.writer import BackgroundWriter
from pulsar_data_collection.spool import Spool

block = threading.Event()
def write(frame, meta):
    block.wait()  # the storage never answers: everything stays in the queue
writer = BackgroundWriter(write, {"m": 1}, max_queue_rows=100_000, max_batch_rows=10, flush_interval_s=0.01,
                          max_retry_delay_s=1, spool=Spool(sys.argv[1]), durability="wal")
for k in range(20):
    writer.submit(pd.DataFrame({"x": [float(k)] * 10}))
print("ready", flush=True)
time.sleep(60)
"""


def test_write_ahead_spool_survives_kill_9(tmp_path):
    script = tmp_path / "crash.py"
    script.write_text(CRASH_SCRIPT, encoding="utf-8")
    spool_dir = tmp_path / "spool"
    child = subprocess.Popen(
        [sys.executable, str(script), str(spool_dir), str(Path(__file__).resolve().parents[1])],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "ready"
    finally:
        child.kill()  # SIGKILL / TerminateProcess
        child.wait(10)
    spool = Spool(spool_dir)  # the next start: write-ahead copies become spooled batches
    rows = sum(len(spool.load(p)[0]) for p in spool.files())
    assert rows == 200 and spool.recovered_files == 20


def test_write_ahead_files_are_deleted_once_written(store, tmp_path):
    with Pulse(params(), options(spool_dir=tmp_path, durability="wal")) as pulse:
        for k in range(5):
            capture(pulse, batch(k * 100), k)
        assert pulse.flush(5)
        stats = pulse.stats()
    assert stats["written_rows"] == 250 and stats["spooled_rows"] == 0
    assert list(tmp_path.iterdir()) == []


def test_write_ahead_needs_a_spool_dir():
    with pytest.raises(ValueError, match="spool_dir"):
        CaptureOptions(durability="wal")


# ---------------------------------------------------------------------------------------------------------- m24
def test_leftover_tmp_files_are_recovered_or_deleted(tmp_path):
    good = tmp_path / "1_aaaa_2r.tmp"
    good.write_text(frame_to_json(pd.DataFrame({"x": [1.0, 2.0]}), {"m": 1}), encoding="utf-8")
    (tmp_path / "2_bbbb_2r.tmp").write_text('{"meta": {', encoding="utf-8")  # interrupted write
    spool = Spool(tmp_path)
    assert [p.name for p in spool.files()] == ["1_aaaa_2r.json"]
    assert list(tmp_path.glob("*.tmp")) == [] and spool.recovered_files == 1


# ---------------------------------------------------------------------------------------------------------- m19
def test_spool_size_limit_evicts_the_oldest_and_counts_them(store, tmp_path):
    store.down = True
    one = len(frame_to_json(InfluxdbActions.prepare_frame(batch(0), None, START), {}).encode())
    pulse = Pulse(params(), options(spool_dir=tmp_path, spool_max_bytes=int(one * 3.5), probe_interval_s=None))
    for k in range(6):
        capture(pulse, batch(k * 100), k)
        assert pulse.flush(5)
    stats = pulse.stats()
    pulse.close(timeout_s=1)
    assert stats["spool_bytes"] <= one * 3.5 and stats["spool_files"] <= 3
    assert stats["spool_evicted_rows"] >= 150 and stats["dropped_rows"] == stats["spool_evicted_rows"]


def test_spool_compression_round_trip(tmp_path):
    frame = InfluxdbActions.prepare_frame(batch(0), None, START)
    plain = Spool(tmp_path / "plain").save(frame, {"m": 1})
    spool = Spool(tmp_path / "gz", compression="gzip")
    path = spool.save(frame, {"m": 1})
    assert path.name.endswith(".json.gz") and gzip.decompress(path.read_bytes())
    assert path.stat().st_size < plain.stat().st_size / 2
    back, meta = spool.load(path)
    pd.testing.assert_frame_equal(back, frame) and meta == {"m": 1}


def test_spool_drop_newest_refuses_new_batches(tmp_path):
    frame = pd.DataFrame({"x": np.arange(100.0)})
    spool = Spool(tmp_path, max_bytes=len(frame_to_json(frame, {}).encode()) + 10, eviction="drop_newest")
    assert spool.save(frame, {}) is not None and spool.save(frame, {}) is None
    assert spool.evicted_rows == 100 and len(spool.files()) == 1


# ---------------------------------------------------------------------------------------------------------- m20
def test_a_batch_larger_than_the_queue_is_cut(store):
    with Pulse(params(), options(max_queue_rows=1000, max_batch_rows=100)) as pulse:
        capture(pulse, pd.DataFrame({"x": np.arange(5000.0)}))
        assert pulse.stats()["queued_rows"] <= 1000
        assert pulse.flush(5)
        stats = pulse.stats()
    assert stats["dropped_rows"] == 4000 and stats["written_rows"] == 1000


# ---------------------------------------------------------------------------------------------------------- m18
def test_replay_is_throttled(store, tmp_path):
    store.down = True
    pulse = Pulse(params(), options(spool_dir=tmp_path, replay_max_rows_per_s=100, max_batch_rows=50))
    for k in range(10):
        capture(pulse, batch(k * 100), k)
        assert pulse.flush(5)
    assert pulse.stats()["spooled_rows"] == 500
    store.down = False
    time.sleep(1.5)
    replayed = pulse.stats()["replayed_rows"]
    pulse.close(timeout_s=1)
    assert 50 <= replayed <= 250  # unthrottled: the 500 rows within a few ms


# ---------------------------------------------------------------------------------------------------------- M31 / M34
def test_shared_writer_one_thread_one_client_merged_requests(store):
    before = threading.active_count()
    writer = SharedWriter(flush_interval_s=0.2, concurrency=2)
    pulses = [Pulse(params(f"model{k}"), options(), writer=writer) for k in range(50)]
    try:
        assert threading.active_count() - before <= 3  # was one writer thread per Pulse
        assert store.actions.clients == 1 and store.actions.checks == 1  # one client, bucket checked once
        for k, pulse in enumerate(pulses):
            capture(pulse, batch(k * 100))
        assert writer.flush(5)
        assert store.requests <= 5  # 50 models, a handful of requests
        assert all(p.stats()["written_rows"] == 50 for p in pulses)
        assert store.count() == 2500
    finally:
        for pulse in pulses:
            pulse.close()
        writer.close()


def test_shared_writer_option_uses_the_process_default(store):
    a = Pulse(params("a"), options(shared_writer=True))
    b = Pulse(params("b"), options(shared_writer=True))
    try:
        assert a._writer.writer is b._writer.writer is SharedWriter.default()
        capture(a, batch(0))
        capture(b, batch(0))
        assert a.flush(5) and b.flush(5)
        assert a.stats()["written_rows"] == 50 and b.stats()["written_rows"] == 50
    finally:
        a.close()
        b.close()
    assert store.count() == 100


def test_concurrent_requests_raise_the_throughput_of_one_pulse(store):
    store.delay_s = 0.1  # each request waits 100 ms for the storage
    elapsed = {}
    for concurrency in (1, 4):
        with Pulse(params(f"c{concurrency}"), options(write_concurrency=concurrency, max_batch_rows=50)) as pulse:
            started = time.monotonic()
            for k in range(12):
                capture(pulse, batch(k * 100), k)
            assert pulse.flush(10)
            elapsed[concurrency] = time.monotonic() - started
            assert pulse.stats()["written_rows"] == 600
    assert elapsed[4] < elapsed[1] / 2


def test_frame_targets_are_still_supported():
    written = []
    writer = SharedWriter(flush_interval_s=0.05, concurrency=1)
    channel = writer.register(Target(write_frame=lambda frame, meta: written.append(len(frame))), {"m": 1})
    channel.submit(pd.DataFrame({"x": [1.0, 2.0]}))
    assert channel.flush(5)
    writer.close()
    assert written == [2] and channel.stats()["written_rows"] == 2


def test_writer_gauges_are_reported(store):
    with Pulse(params(), options()) as pulse:
        capture(pulse, batch(0))
        assert pulse.flush(5)
        stats = pulse.stats()
    assert stats["last_write_ms"] is not None and stats["max_write_ms"] >= stats["last_write_ms"]
    assert stats["writer_lag_s"] == 0.0


# ---------------------------------------------------------------------------------------------------------- m22 / m23
class ManySink:
    def __init__(self):
        self.requests = []

    def write_stats(self, stats, tags=None):
        self.requests.append([(stats, tags)])

    def write_stats_many(self, items, measurement=None):
        self.requests.append([(stats, {**tags, "_measurement": measurement}) for stats, tags in items])


class StatsPulse:
    model_version = "1"

    def __init__(self, model_id):
        self.model_id = model_id

    def stats(self):
        return {"captured_rows": 1, "queued_rows": 2, "writer_lag_s": 0.5, "max_write_ms": 3.0}


def test_one_reporter_for_many_pulses():
    sink = ManySink()
    before = threading.active_count()
    reporter = StatsReporter(None, sink, interval_s=0.05, tags={"environment": "test"})
    for k in range(100):
        reporter.add(StatsPulse(f"m{k}"))
    reporter.start()
    assert threading.active_count() - before == 1  # was one thread per Pulse
    time.sleep(0.18)
    reporter.stop()
    assert all(len(request) == 100 for request in sink.requests)  # one request per interval for the 100 Pulse
    assert sink.requests[-1][5][1]["model_id"] == "m5" and sink.requests[-1][5][1]["environment"] == "test"


def test_process_health_point():
    sink = ManySink()
    reporter = StatsReporter(StatsPulse("m"), sink, interval_s=60, process_health=True)
    reporter.report()
    health, tags = sink.requests[-1][0]
    assert tags["_measurement"] == "pulsar_process_health" and "pid" in tags and "host" in tags
    assert health["threads"] >= 1 and health["queued_rows"] == 2 and health["writer_lag_s"] == 0.5
    assert health["max_write_ms"] == 3.0


def test_process_health_uses_psutil_when_installed():
    pytest.importorskip("psutil")
    health = process_health()
    assert health["rss_bytes"] > 0 and ("open_fds" in health or "open_handles" in health)
