"""R1 (lab test SC-04, 0.13.1): without spool_dir, the rows of a write whose connection was cut after the request was
sent (or that timed out) were counted in uncertain_rows and never written again, even with room in the queue. They are
now written once more (line protocol with explicit timestamps: a second write overwrites the same points, no
duplicate), then counted in uncertain_rows only if that write is uncertain too, or if they cannot wait in the queue.

The connection cuts go through a real HTTP server and the real InfluxDB client, for both write profiles: the private
writer of each Pulse, and a SharedWriter with concurrent requests.
"""

import socket
import struct
import threading
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from pulsar_data_collection.exceptions import StorageConnectionResetError
from pulsar_data_collection.models import CaptureOptions, DataWithPrediction, PulseParameters
from pulsar_data_collection.pulse import Pulse
from pulsar_data_collection.writer import SharedWriter, Target

START = datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc)
N = 50


class CuttingServer:
    """InfluxDB-like HTTP server: reads each write request in full (headers and body), then cuts the connection
    without answering for the first `cuts` writes (`style`: "reset" sends a RST, "close" a FIN), and stores the
    others like InfluxDB (one point per series and timestamp, 204). /ping answers 204; any other request is cut."""

    def __init__(self, cuts: int = 1, style: str = "reset"):
        self.cuts = cuts
        self.style = style
        self.socket = socket.socket()
        self.socket.bind(("127.0.0.1", 0))
        self.socket.listen(64)
        self.url = f"http://127.0.0.1:{self.socket.getsockname()[1]}"
        self.writes = []  # bodies of the write requests received in full, cut or not
        self.points = {}  # (series, timestamp) -> fields, of the stored writes
        self.lock = threading.Lock()
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                connection, _ = self.socket.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(connection,), daemon=True).start()

    def _handle(self, connection):
        try:
            while True:  # keep-alive: several requests on one connection
                data = b""
                while b"\r\n\r\n" not in data:
                    chunk = connection.recv(65536)
                    if not chunk:
                        return
                    data += chunk
                head, body = data.split(b"\r\n\r\n", 1)
                lines = head.split(b"\r\n")
                length = next((int(h.split(b":")[1]) for h in lines if h.lower().startswith(b"content-length")), 0)
                while len(body) < length:
                    body += connection.recv(65536)
                if head.startswith(b"GET /ping"):
                    connection.sendall(b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\n\r\n")
                    continue
                if not head.startswith(b"POST /api/v2/write"):
                    self._cut(connection, "reset")
                    return
                with self.lock:
                    self.writes.append(body)
                    cut = len(self.writes) <= self.cuts
                    if not cut:
                        for line in body.decode().split("\n"):
                            series, fields, ts = line.rsplit(" ", 2)
                            self.points[(series, int(ts))] = fields
                if cut:
                    self._cut(connection, self.style)
                    return
                connection.sendall(b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\n\r\n")
        except OSError:
            return
        finally:
            try:
                connection.close()
            except OSError:
                pass

    @staticmethod
    def _cut(connection, style):
        if style == "reset":
            connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))  # RST, not FIN
        connection.close()

    def lines_written(self) -> int:
        with self.lock:
            return sum(body.count(b"\n") + 1 for body in self.writes)

    def stored(self) -> int:
        with self.lock:
            return len(self.points)

    def close(self):
        self.socket.close()


def params(url: str, model: str = "m") -> PulseParameters:
    return PulseParameters(
        model_id=model,
        model_version="1",
        data_id="d",
        reference_data_storage="ref.csv",
        target_name="y",
        storage_engine="influxdb",
        timestamp_column_name="_time",
        login={"url": url, "token": "t", "org": "o", "bucket_name": "b"},
    )


def options(**kwargs) -> CaptureOptions:
    return CaptureOptions(**{"flush_interval_s": 0.05, "max_retry_delay_s": 0.2, "probe_interval_s": None, **kwargs})


def batch(seq0: int) -> pd.DataFrame:
    rng = np.random.default_rng(seq0)
    frame = pd.DataFrame(rng.normal(size=(N, 3)), columns=["f0", "f1", "f2"])
    frame["seq"] = np.arange(seq0, seq0 + N, dtype="int64")
    return frame


def capture(pulse, frame, k=0):
    pulse.capture_data(
        DataWithPrediction(
            data_points=frame, predictions=None, timestamp=START + timedelta(seconds=k), features_names=list(frame.columns)
        )
    )


def accounted(stats) -> int:
    return sum(stats[k] for k in ("written_rows", "spooled_rows", "dropped_rows", "rejected_rows", "uncertain_rows"))


@pytest.fixture(params=["private", "shared"])
def profile(request):
    """How the Pulse write: their own writer thread, or a SharedWriter with 4 requests in flight"""
    writers = []

    def make(server, opts, models=("m",)):
        writer = None
        if request.param == "shared":
            writer = SharedWriter(flush_interval_s=0.05, max_retry_delay_s=0.2, concurrency=4, probe_interval_s=None)
            writers.append(writer)
        return [Pulse(params(server.url, model), opts, writer=writer) for model in models]

    yield make
    for writer in writers:
        writer.close(timeout_s=1)


@pytest.mark.parametrize("style", ["reset", "close"])
def test_a_write_cut_after_sending_is_written_once_more_and_stored(profile, style):
    server = CuttingServer(cuts=1, style=style)
    (pulse,) = profile(server, options())
    try:
        capture(pulse, batch(0))
        assert pulse.flush(10)
        stats = pulse.stats()
    finally:
        pulse.close(timeout_s=1)
        server.close()
    assert len(server.writes) == 2 and server.writes[1] == server.writes[0]  # the same points, once more
    assert server.stored() == N  # no duplicate: the second write overwrote the same points
    assert stats["written_rows"] == N and stats["uncertain_rows"] == 0 and stats["dropped_rows"] == 0
    assert stats["retried_uncertain_rows"] == N and stats["write_errors"] == 1
    assert accounted(stats) == stats["captured_rows"] == N


def test_a_second_uncertain_write_is_counted_and_never_retried_again(profile):
    server = CuttingServer(cuts=10**6)  # every write is cut
    (pulse,) = profile(server, options())
    try:
        capture(pulse, batch(0))
        assert pulse.flush(10)
        time.sleep(0.5)  # longer than the back-off: a third write would have been sent by now
        stats = pulse.stats()
    finally:
        pulse.close(timeout_s=1)
        server.close()
    assert len(server.writes) == 2  # once, then once more; never a third time
    assert stats["uncertain_rows"] == N and stats["written_rows"] == 0 and stats["dropped_rows"] == 0
    assert stats["retried_uncertain_rows"] == N and stats["write_errors"] == 2
    assert accounted(stats) == stats["captured_rows"] == N


def test_several_models_and_batches_are_retried_with_exact_counters(profile):
    """Requests of several models (shared: in the same requests); the first one is cut: its rows, of one or several
    models, are written once more, and every row is stored once"""
    server = CuttingServer(cuts=1)
    pulses = profile(server, options(), models=("a", "b", "c"))
    try:
        for k in range(4):
            for pulse in pulses:
                capture(pulse, batch(k * N), k)
            time.sleep(0.06)
        assert all(pulse.flush(15) for pulse in pulses)
        stats = [pulse.stats() for pulse in pulses]
    finally:
        for pulse in pulses:
            pulse.close(timeout_s=1)
        server.close()
    assert server.stored() == 3 * 4 * N  # every row stored, none twice
    for s in stats:
        assert s["written_rows"] == 4 * N and s["uncertain_rows"] == 0 and s["dropped_rows"] == 0
        assert accounted(s) == s["captured_rows"] == 4 * N
    assert sum(s["retried_uncertain_rows"] for s in stats) > 0


def test_retry_uncertain_false_keeps_the_previous_behaviour(profile):
    server = CuttingServer(cuts=1)
    (pulse,) = profile(server, options(retry_uncertain=False))
    try:
        capture(pulse, batch(0))
        assert pulse.flush(10)
        time.sleep(0.5)
        stats = pulse.stats()
    finally:
        pulse.close(timeout_s=1)
        server.close()
    assert len(server.writes) == 1
    assert stats["uncertain_rows"] == N and stats["retried_uncertain_rows"] == 0 and stats["written_rows"] == 0


def test_with_spool_a_cut_write_is_spooled_and_replayed_not_retried_in_memory(profile, tmp_path):
    server = CuttingServer(cuts=1)
    (pulse,) = profile(server, options(spool_dir=tmp_path))
    try:
        capture(pulse, batch(0))
        assert pulse.flush(10)
        deadline = time.monotonic() + 10
        while pulse.stats()["replayed_rows"] < N and time.monotonic() < deadline:
            time.sleep(0.05)
        stats = pulse.stats()
    finally:
        pulse.close(timeout_s=1)
        server.close()
    assert stats["spooled_rows"] == N and stats["replayed_rows"] == N and stats["retried_uncertain_rows"] == 0
    assert stats["uncertain_rows"] == 0 and stats["dropped_rows"] == 0 and stats["spool_files"] == 0
    assert len(server.writes) == 2 and server.stored() == N  # unchanged: one replay, no duplicate


# ---------------------------------------------------------------------------------------------------------- queue bound
def seq(start: int, rows: int = 10) -> pd.DataFrame:
    return pd.DataFrame({"seq": range(start, start + rows)})


class BlockingCut:
    """write_frame whose first call waits for `release`, then raises a cut connection; the next ones store"""

    def __init__(self, cuts: int = 1):
        self.cuts = cuts
        self.calls = []
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, frame, meta):
        self.calls.append(list(frame["seq"]))
        if len(self.calls) == 1:
            self.started.set()
            assert self.release.wait(10)
        if len(self.calls) <= self.cuts:
            raise StorageConnectionResetError("connection cut after the request was sent")


def test_rows_to_retry_without_room_in_the_queue_are_counted_uncertain_not_dropped():
    """The retry respects max_queue_rows: the rows go back in their place (the oldest), so a full queue drops them
    first; they may be stored, hence counted in uncertain_rows. The caller is never blocked."""
    write = BlockingCut()
    writer = SharedWriter(flush_interval_s=0.02, max_retry_delay_s=0.2, concurrency=1, probe_interval_s=None)
    channel = writer.register(Target(write_frame=write, idempotent=True), {}, max_queue_rows=20)
    channel.submit(seq(0))
    assert write.started.wait(5)  # the first batch is in flight
    channel.submit(seq(10))
    channel.submit(seq(20))  # the queue is full (20 rows)
    write.release.set()
    assert channel.flush(10)
    stats = channel.stats()
    writer.close(timeout_s=1)
    assert write.calls == [list(range(10)), list(range(10, 30))]  # the cut batch was not written again
    assert stats["retried_uncertain_rows"] == 10 and stats["uncertain_rows"] == 10 and stats["dropped_rows"] == 0
    assert stats["written_rows"] == 20 and accounted(stats) == stats["captured_rows"] == 30


def test_rows_waiting_for_their_retry_are_dropped_oldest_first_and_counted_uncertain():
    """Rows waiting for their second write (back-off) are the oldest of the queue: newer captures that overflow it
    drop them first, counted in uncertain_rows; then the oldest fresh batches, counted in dropped_rows"""
    writer = SharedWriter(flush_interval_s=0.02, concurrency=1, probe_interval_s=None)
    calls = []

    def write(frame, meta):
        calls.append(list(frame["seq"]))
        raise StorageConnectionResetError("cut")

    channel = writer.register(Target(write_frame=write, idempotent=True), {}, max_queue_rows=30)
    channel.submit(seq(0))
    deadline = time.monotonic() + 5
    while not channel.stats()["retried_uncertain_rows"] and time.monotonic() < deadline:
        time.sleep(0.01)
    with writer._condition:
        channel.endpoint.retry_at = time.monotonic() + 3600  # the rows to retry wait in the queue
    for k in (10, 20, 30):
        channel.submit(seq(k))  # 40 rows for 30: the rows to retry go first
    waiting = channel.stats()
    channel.submit(seq(40))  # then the oldest fresh batch
    stats = channel.stats()
    writer.close(timeout_s=1)  # closing during the back-off: the rest is dropped and counted, no wait
    final = channel.stats()
    assert calls == [list(range(10))]
    assert waiting["uncertain_rows"] == 10 and waiting["dropped_rows"] == 0 and waiting["queued_rows"] == 30
    assert stats["uncertain_rows"] == 10 and stats["dropped_rows"] == 10
    assert final["dropped_rows"] == 40 and final["uncertain_rows"] == 10 and accounted(final) == final["captured_rows"]


def test_rows_waiting_for_their_retry_at_close_are_counted_uncertain():
    """close() does not wait for the end of a back-off: rows waiting for their second write are counted in
    uncertain_rows (they may be stored), not in dropped_rows"""
    writer = SharedWriter(flush_interval_s=0.02, concurrency=1, probe_interval_s=None)
    calls = []

    def write(frame, meta):
        calls.append(len(frame))
        raise StorageConnectionResetError("cut")

    channel = writer.register(Target(write_frame=write, idempotent=True), {})
    channel.submit(seq(0))
    deadline = time.monotonic() + 5
    while not channel.stats()["retried_uncertain_rows"] and time.monotonic() < deadline:
        time.sleep(0.01)
    with writer._condition:
        channel.endpoint.retry_at = time.monotonic() + 3600
    writer.close(timeout_s=2)
    stats = channel.stats()
    assert calls == [10]
    assert stats["uncertain_rows"] == 10 and stats["dropped_rows"] == 0 and stats["retried_uncertain_rows"] == 10


def test_a_retried_batch_and_a_fresh_one_in_the_same_request_are_treated_apart():
    """Line-protocol batches of one destination share a request: a cut request holding rows already retried and fresh
    rows counts the former in uncertain_rows and gives the latter their own second write"""
    sent = []

    def send(lines):
        sent.append(sorted(lines))
        raise StorageConnectionResetError("cut")

    writer = SharedWriter(flush_interval_s=0.02, concurrency=1, probe_interval_s=None)
    target = Target(encode=lambda frame, meta: [str(v) for v in frame["seq"]], send=send, key="k")
    channel = writer.register(target, {})
    channel.submit(seq(0))
    deadline = time.monotonic() + 5
    while not channel.stats()["retried_uncertain_rows"] and time.monotonic() < deadline:
        time.sleep(0.01)
    with writer._condition:
        channel.endpoint.retry_at = time.monotonic() + 3600
    channel.submit(seq(100))
    with writer._condition:
        channel.endpoint.retry_at = 0.0  # both batches are taken together
        writer._condition.notify_all()
    assert channel.flush(10)
    stats = channel.stats()
    writer.close(timeout_s=1)
    first = sorted(str(v) for v in range(10))
    fresh = sorted(str(v) for v in range(100, 110))
    assert sent == [first, sorted(first + fresh), fresh]
    assert stats["uncertain_rows"] == 20 and stats["retried_uncertain_rows"] == 20 and stats["dropped_rows"] == 0


def test_a_write_frame_target_not_declared_idempotent_is_not_retried():
    """A custom write function may not be idempotent (e.g. appending rows): its uncertain rows are counted at once,
    as before 0.13.1, unless the Target declares idempotent=True"""
    stored = []

    def write(frame, meta):
        stored.extend(frame["seq"])
        raise TimeoutError("answer lost after storing")

    writer = SharedWriter(flush_interval_s=0.02, concurrency=1, probe_interval_s=None)
    channel = writer.register(Target(write_frame=write), {})
    channel.submit(seq(0))
    assert channel.flush(5)
    stats = channel.stats()
    writer.close(timeout_s=1)
    assert stored == list(range(10))  # written once, not twice
    assert stats["uncertain_rows"] == 10 and stats["retried_uncertain_rows"] == 0


def test_requeueing_thousands_of_batches_is_one_merge():
    """Thousands of one-row captures taken in one request, then cut, go back in one merge by submission order: the
    deque was scanned for each batch with the lock held (2 s for 5,000 batches against 5,000 queued ones), blocking
    the callers"""
    writer = SharedWriter(flush_interval_s=60, max_batch_rows=5000, concurrency=1, probe_interval_s=None)
    channel = writer.register(Target(write_frame=lambda f, m: None), {}, max_queue_rows=100_000)
    with writer._condition:
        channel.endpoint.retry_at = time.monotonic() + 3600
    for k in range(5000):
        channel.submit(seq(k, 1))
    with writer._condition:
        channel.endpoint.retry_at = 0.0
        taken = writer._take()
        channel.endpoint.retry_at = time.monotonic() + 3600
    for k in range(5000, 10_000):
        channel.submit(seq(k, 1))
    (job,) = writer._jobs(taken)
    (unit,) = job.units
    assert len(unit.sizes) == 5000
    started = time.perf_counter()
    assert channel.requeue(unit, retry=True)
    elapsed = time.perf_counter() - started
    order = [int(e.frame["seq"].iloc[0]) for e in channel.queue]
    flags = all(e.retried == (e.seq <= 5000) for e in channel.queue)
    stats = channel.stats()
    with writer._condition:
        channel.pending_rows = 0
    writer.close(timeout_s=1)
    assert order == list(range(10_000)) and flags
    assert stats["queued_rows"] == 10_000 and stats["retried_uncertain_rows"] == 5000
    assert elapsed < 1.0  # about 2 s before, a few tens of ms now (most of it slicing the frame, outside the lock)


def test_the_second_write_waits_for_a_storage_that_answers():
    """Right after a cut the back-off often ends while the link is still down (flapping link): the second write of the
    uncertain rows is only sent once the storage answers its ping, so that it is not spent on a dead link"""
    calls, pings = [], []
    answers = iter([False, False, True])

    def ping():
        pings.append(1)
        return next(answers, True)

    def write(frame, meta):
        calls.append(list(frame["seq"]))
        if len(calls) == 1:
            raise StorageConnectionResetError("cut")

    writer = SharedWriter(flush_interval_s=0.02, max_retry_delay_s=0.1, concurrency=1, probe_interval_s=None)
    channel = writer.register(Target(write_frame=write, ping=ping, idempotent=True), {})
    channel.submit(seq(0))
    assert channel.flush(10)
    stats = channel.stats()
    writer.close(timeout_s=1)
    assert calls == [list(range(10)), list(range(10))]  # not sent while the storage did not answer
    assert len(pings) == 3 and stats["write_errors"] == 1  # a ping that fails is not a write error
    assert stats["written_rows"] == 10 and stats["uncertain_rows"] == 0 and stats["retried_uncertain_rows"] == 10


def test_rows_waiting_for_a_storage_that_never_answers_are_counted_uncertain_at_close():
    calls = []

    def write(frame, meta):
        calls.append(len(frame))
        raise StorageConnectionResetError("cut")

    writer = SharedWriter(flush_interval_s=0.02, max_retry_delay_s=0.05, concurrency=1, probe_interval_s=None)
    channel = writer.register(Target(write_frame=write, ping=lambda: False, idempotent=True), {})
    channel.submit(seq(0))
    assert not channel.flush(0.5)  # the rows wait for the storage
    writer.close(timeout_s=1)
    stats = channel.stats()
    assert calls == [10]
    assert stats["uncertain_rows"] == 10 and stats["dropped_rows"] == 0 and accounted(stats) == stats["captured_rows"]
