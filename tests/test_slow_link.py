"""R2, R3 and R4 (lab test SC-04): capture on a slow link.

- R2: a request needing more than bandwidth x write timeout timed out forever: with `max_batch_rows` = 5,000 rows of
  about 560 bytes and a 5 s timeout, nothing was written at 1 Mbit/s. A timed-out request now makes the next ones 8
  times smaller (adaptive batch size), and fast writes make them larger again.
- R3: with a spool, a link that answers, but slower than the traffic, overflowed the queue: the oldest batches were
  dropped, since the spool only took the rows of an unavailable storage. They are now spooled (by a spooler thread, not
  in the caller's thread), and replayed once the queue has drained.
- R4: the rows in flight came on top of `max_queue_rows`. `bound_in_flight=True` counts them against it; the stats
  report the rows in flight and the peak of rows held (queued and in flight).
"""

import threading
import time

import pandas as pd
import pytest

from pulsar_data_collection.exceptions import StorageWriteTimeoutError
from pulsar_data_collection.models import CaptureOptions
from pulsar_data_collection.spool import Spool
from pulsar_data_collection.writer import SharedWriter, Target


class SlowLink:
    """A storage behind a link of `row_s` seconds per row: a request needing more than `timeout_s` times out (after
    `timeout_s`, without storing), the others are stored once (a set of seq: idempotent)"""

    def __init__(self, row_s: float = 0.0, timeout_s: float = 0.2):
        self.row_s = row_s
        self.timeout_s = timeout_s
        self.stored = set()
        self.requests = []  # rows of every request, timed out or not
        self.lock = threading.Lock()

    def send(self, lines):
        with self.lock:
            self.requests.append(len(lines))
            row_s = self.row_s
        need = len(lines) * row_s
        if need > self.timeout_s:
            time.sleep(self.timeout_s)
            raise StorageWriteTimeoutError("write timed out")
        time.sleep(need)
        with self.lock:
            self.stored.update(lines)

    def target(self, key="k") -> Target:
        return Target(encode=lambda frame, meta: frame["seq"].tolist(), send=self.send, ping=lambda: True, key=key)


def frame(start: int, rows: int = 50) -> pd.DataFrame:
    return pd.DataFrame({"seq": range(start, start + rows), "x": [0.5] * rows})


def writer(**kwargs) -> SharedWriter:
    return SharedWriter(**{"max_batch_rows": 1_000, "flush_interval_s": 0.02, "max_retry_delay_s": 0.05, **kwargs})


def wait(condition, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


def accounted(stats) -> int:
    keys = ("written_rows", "spooled_rows", "dropped_rows", "rejected_rows", "uncertain_rows", "queued_rows")
    return sum(stats[k] for k in keys)


# ---------------------------------------------------------------------------------------------------------- R2
def test_a_slow_link_gets_smaller_requests_instead_of_timing_out_forever():
    link = SlowLink(row_s=0.001, timeout_s=0.2)  # 200 rows per request at most
    w = writer()
    channel = w.register(link.target(), {}, max_queue_rows=100_000)
    for k in range(40):
        channel.submit(frame(k * 50))
    assert channel.flush(20)
    stats = channel.stats()
    w.close()
    assert stats["batch_limit_rows"] <= 250 and stats["batch_shrinks"] >= 1
    assert link.stored == set(range(2_000))  # every row written, none lost
    assert stats["written_rows"] == 2_000 and stats["uncertain_rows"] == 0 and stats["dropped_rows"] == 0
    assert stats["retried_uncertain_rows"] > 0  # the timed-out rows were written once more, in smaller requests
    assert link.requests.count(1_000) <= 2  # the two first requests (concurrency 2), never again


def test_without_adaptive_batch_every_request_times_out():
    link = SlowLink(row_s=0.001, timeout_s=0.2)
    w = writer(adaptive_batch=False)
    channel = w.register(link.target(), {}, max_queue_rows=100_000)
    channel.submit(frame(0, 1_000))
    channel.flush(5)
    stats = channel.stats()
    w.close()
    assert stats["batch_limit_rows"] == 1_000 and stats["batch_shrinks"] == 0
    assert stats["written_rows"] == 0 and stats["uncertain_rows"] == 1_000  # timed out twice (R1 retry), counted


def test_requests_grow_back_once_the_link_is_fast():
    link = SlowLink(row_s=0.002, timeout_s=0.2)
    w = writer()
    channel = w.register(link.target(), {}, max_queue_rows=100_000)
    for k in range(20):
        channel.submit(frame(k * 50))
    assert channel.flush(20)
    assert channel.stats()["batch_limit_rows"] < 1_000
    link.row_s = 0.0
    k = 20
    while channel.stats()["batch_limit_rows"] < 1_000 and k < 400:
        for _ in range(25):
            channel.submit(frame(k * 50))
            k += 1
        channel.flush(5)
    stats = channel.stats()
    w.close()
    assert stats["batch_limit_rows"] == 1_000
    assert link.stored == set(range(k * 50)) and stats["uncertain_rows"] == 0


def test_the_adaptive_batch_size_stops_at_min_batch_rows():
    link = SlowLink(row_s=1.0, timeout_s=0.05)  # not even one row passes
    w = writer(min_batch_rows=30, max_batch_rows=10_000)
    channel = w.register(link.target(), {}, max_queue_rows=100_000, retry_uncertain=False)
    for k in range(8):  # 2,000 rows, then requests of 250, 31 and 30 rows
        channel.submit(frame(k * 250, 250))
        channel.flush(10)
    stats = channel.stats()
    w.close()
    assert stats["batch_limit_rows"] == 30
    assert 30 in link.requests and max(link.requests[-3:]) <= 30
    assert stats["uncertain_rows"] == 2_000 and accounted(stats) == stats["captured_rows"]


def test_a_batch_larger_than_the_request_is_split_and_written_once():
    link = SlowLink(row_s=0.001, timeout_s=0.2)
    w = writer()
    channel = w.register(link.target(), {}, max_queue_rows=100_000)
    channel.submit(frame(0, 1_000))  # one batch of 1,000 rows: times out, then goes in requests of at most 250
    channel.submit(frame(1_000, 1_000))
    assert channel.flush(20)
    stats = channel.stats()
    w.close()
    assert link.stored == set(range(2_000)) and stats["written_rows"] == 2_000
    assert stats["in_flight_rows"] == 0 and stats["queued_rows"] == 0
    assert link.requests.count(1_000) <= 2 and max(link.requests[2:]) <= 250


def test_a_shared_destination_shrinks_once_for_concurrent_timeouts():
    link = SlowLink(row_s=0.001, timeout_s=0.2)
    w = writer(concurrency=4)
    channels = [w.register(link.target(), {}, max_queue_rows=100_000) for _ in range(4)]
    for k in range(80):
        channels[k % 4].submit(frame(k * 50))
    assert all(c.flush(20) for c in channels)
    stats = [c.stats() for c in channels]
    w.close()
    assert link.stored == set(range(4_000))
    assert len({s["batch_limit_rows"] for s in stats}) == 1 and stats[0]["batch_limit_rows"] <= 250  # one destination
    assert sum(s["written_rows"] for s in stats) == 4_000


def test_a_spooled_batch_larger_than_the_request_is_replayed_in_smaller_requests(tmp_path):
    link = SlowLink(row_s=0.001, timeout_s=0.2)
    spool = Spool(tmp_path)
    spool.save(frame(0, 1_000), {})  # e.g. a 1,000-row request that timed out and was spooled
    w = writer()
    channel = w.register(link.target(), {}, max_queue_rows=100_000, spool=spool)
    channel.endpoint.batch_rows = 250  # the destination is known to be slow
    channel.endpoint.timeout_s = 0.2
    assert wait(lambda: not spool.files(), 10)
    stats = channel.stats()
    w.close()
    assert link.stored == set(range(1_000)) and stats["replayed_rows"] == 1_000
    assert max(link.requests) <= 250


def test_capture_options_reach_the_private_writer():
    options = CaptureOptions(adaptive_batch=False, min_batch_rows=7, spool_overflow=False, bound_in_flight=True)
    assert (options.adaptive_batch, options.min_batch_rows, options.spool_overflow, options.bound_in_flight) == (
        False,
        7,
        False,
        True,
    )
    defaults = CaptureOptions()
    assert defaults.adaptive_batch and defaults.spool_overflow and not defaults.bound_in_flight
    with pytest.raises(ValueError):
        CaptureOptions(min_batch_rows=0)


# ---------------------------------------------------------------------------------------------------------- R3
class Gate:
    """A storage that answers every request after `release` is set (slow, never unavailable)"""

    def __init__(self):
        self.release = threading.Event()
        self.stored = set()
        self.lock = threading.Lock()

    def send(self, lines):
        self.release.wait(10)
        with self.lock:
            self.stored.update(lines)

    def target(self) -> Target:
        return Target(encode=lambda frame, meta: frame["seq"].tolist(), send=self.send, ping=lambda: True, key="g")


def test_with_a_spool_the_queue_overflow_is_spooled_then_replayed(tmp_path):
    gate = Gate()
    spool = Spool(tmp_path)
    saved_by = []
    save = spool.save

    def recording_save(frame, meta, wal=False):
        saved_by.append(threading.current_thread().name)
        return save(frame, meta, wal=wal)

    spool.save = recording_save
    w = writer(max_batch_rows=100)
    channel = w.register(gate.target(), {}, max_queue_rows=200, spool=spool)
    for k in range(20):  # 1,000 rows while the first requests are stuck: 200 queued, the rest beyond the bound
        assert wait(lambda: channel.overflow_rows == 0)  # the spooler keeps up (a faster burst is bounded too)
        channel.submit(frame(k * 50))
    assert wait(lambda: channel.overflow_rows == 0 and channel.stats()["overflow_spooled_rows"] >= 600)
    during = channel.stats()
    assert during["dropped_rows"] == 0 and during["queued_rows"] <= 200
    assert saved_by and "MainThread" not in saved_by  # never saved in the caller's thread
    gate.release.set()
    assert channel.flush(10)
    assert wait(lambda: not spool.files())
    stats = channel.stats()
    w.close()
    assert gate.stored == set(range(1_000))
    assert stats["dropped_rows"] == 0 and stats["written_rows"] + stats["replayed_rows"] == 1_000
    assert stats["spooled_rows"] == stats["overflow_spooled_rows"] == stats["replayed_rows"]


def test_without_spool_overflow_the_oldest_batches_are_dropped_as_before(tmp_path):
    gate = Gate()
    w = writer(max_batch_rows=100)
    channel = w.register(gate.target(), {}, max_queue_rows=200, spool=Spool(tmp_path), spool_overflow=False)
    for k in range(20):
        channel.submit(frame(k * 50))
    during = channel.stats()
    gate.release.set()
    assert channel.flush(10)
    stats = channel.stats()
    w.close()
    assert during["dropped_rows"] >= 600 and stats["overflow_spooled_rows"] == 0
    assert accounted(stats) == 1_000


def test_a_full_spool_counts_the_overflow_it_refuses(tmp_path):
    gate = Gate()
    spool = Spool(tmp_path, max_bytes=1, eviction="drop_newest")
    w = writer(max_batch_rows=100)
    channel = w.register(gate.target(), {}, max_queue_rows=200, spool=spool)
    for k in range(20):
        channel.submit(frame(k * 50))
    assert wait(lambda: channel.overflow_rows == 0 and channel.stats()["spool_evicted_rows"] > 0)
    gate.release.set()
    assert channel.flush(10)
    stats = channel.stats()
    w.close()
    assert stats["overflow_spooled_rows"] == 0
    assert stats["written_rows"] + stats["dropped_rows"] == 1_000  # dropped_rows includes spool_evicted_rows


def test_a_batch_larger_than_the_queue_goes_to_the_spool(tmp_path):
    gate = Gate()
    gate.release.set()
    spool = Spool(tmp_path)
    w = writer()
    channel = w.register(gate.target(), {}, max_queue_rows=100, spool=spool)
    channel.submit(frame(0, 300))
    assert channel.flush(10)
    assert wait(lambda: not spool.files())
    stats = channel.stats()
    w.close()
    assert gate.stored == set(range(300)) and stats["dropped_rows"] == 0
    assert stats["overflow_spooled_rows"] == 200 and stats["replayed_rows"] == 200


# ---------------------------------------------------------------------------------------------------------- R4
def test_rows_in_flight_are_reported_and_held_rows_peak():
    gate = Gate()
    w = writer(max_batch_rows=100, concurrency=1)
    channel = w.register(gate.target(), {}, max_queue_rows=200)
    channel.submit(frame(0, 100))  # one request of 100 rows, stuck
    assert wait(lambda: channel.stats()["in_flight_rows"] == 100)
    for k in range(2, 20):
        channel.submit(frame(k * 50))
    during = channel.stats()
    gate.release.set()
    assert channel.flush(10)
    stats = channel.stats()
    w.close()
    assert during["queued_rows"] == 200 and during["in_flight_rows"] == 100  # in flight on top of the bound
    assert stats["max_held_rows"] == 300 and stats["in_flight_rows"] == 0


def test_bound_in_flight_keeps_queued_and_in_flight_rows_within_max_queue_rows():
    gate = Gate()
    w = writer(max_batch_rows=100, concurrency=1)
    channel = w.register(gate.target(), {}, max_queue_rows=200, bound_in_flight=True)
    channel.submit(frame(0, 100))  # one request of 100 rows, stuck
    assert wait(lambda: channel.stats()["in_flight_rows"] == 100)
    for k in range(2, 20):
        channel.submit(frame(k * 50))
    during = channel.stats()
    gate.release.set()
    assert channel.flush(10)
    stats = channel.stats()
    w.close()
    assert during["queued_rows"] + during["in_flight_rows"] <= 250  # the newest batch is always kept, as without
    assert stats["max_held_rows"] <= 250
    assert accounted(stats) == 1_000 and stats["dropped_rows"] == 1_000 - stats["written_rows"]


def test_requeued_rows_are_no_longer_counted_in_flight():
    link = SlowLink(row_s=0.001, timeout_s=0.05)  # 50 rows pass, a request of 100 times out
    w = writer(max_batch_rows=100, adaptive_batch=False)
    channel = w.register(link.target(), {}, max_queue_rows=1_000, bound_in_flight=True)
    channel.submit(frame(0, 100))  # times out, put back in the queue for its second write
    assert wait(lambda: channel.stats()["retried_uncertain_rows"] == 100)
    assert wait(lambda: channel.stats()["in_flight_rows"] in (0, 100) and channel.stats()["queued_rows"] in (0, 100))
    assert channel.flush(5)
    stats = channel.stats()
    w.close()
    # counted once: never queued and in flight at the same time (the bound would otherwise have dropped them)
    assert stats["max_held_rows"] == 100 and stats["in_flight_rows"] == 0
    assert stats["uncertain_rows"] == 100 and stats["dropped_rows"] == 0
