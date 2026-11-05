"""Background writer: keeps writes to the storage engine out of the prediction path.

`SharedWriter` drains the in-memory queues of one or more channels (one channel per Pulse) with a single thread, and
sends up to `concurrency` requests at once. Channels writing to the same InfluxDB bucket with the same client have their
rows sent in the same requests (line protocol), whatever their model. `BackgroundWriter` is the private, single-channel
form used by default by each Pulse.
"""

import logging
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd

from .exceptions import StorageRejectedError, StorageWriteTimeoutError, StorageWriteUncertainError
from .spool import Spool

logger = logging.getLogger("pulsar_data_collection")

GATE_PINGS = 3  # failed pings after which the second write of uncertain rows is sent anyway (ping may be unreliable)
SHRINK_FACTOR = 4  # a request that timed out: the rows per request of its destination are divided by this much

WriteFunction = Callable[[pd.DataFrame, dict], None]


# ---------------------------------------------------------------------------------------------------------- grouping
def _signature(frame: pd.DataFrame) -> tuple:
    return tuple(frame.columns), tuple(str(t) for t in frame.dtypes)


def _by_columns(batch: list) -> list:
    """Concatenate the frames that share the same columns AND dtypes, keeping the others apart

    Concatenating frames whose columns differ fills the gaps with NaN: an integer or boolean column then becomes float
    or object, and the storage refuses it against the type it already holds. Concatenating frames whose dtypes differ
    (a float column in one, strings in the other) turns the column into strings: valid rows were then refused with the
    conflicting ones. Records with optional fields are therefore written in one write per set of columns and dtypes.
    """
    groups = {}
    for frame in batch:
        groups.setdefault(_signature(frame), []).append(frame)
    return [pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0] for frames in groups.values()]


_MIXED = ("mixed", "mixed-integer")  # strings (or other objects) together with numbers or booleans


def _kind(value) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "n"
    if isinstance(value, (bool, np.bool_)):
        return "b"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return "f"
    return "s"


def split_mixed_objects(frame: pd.DataFrame, signature: Optional[tuple] = None) -> list:
    """Split a frame whose object columns mix numbers, booleans and strings into frames of one kind per column

    The storage serializes an object column as strings: one string in a float column made every row of the batch
    conflict with the stored float field. Each part gets its column converted (float, bool or str), so that only the
    rows really in conflict are refused. `signature` (from `_signature`) spares reading the dtypes again.
    """
    columns, dtypes = signature or _signature(frame)
    objects = [c for c, dtype in zip(columns, dtypes) if dtype == "object"]
    mixed = [c for c in objects if pd.api.types.infer_dtype(frame[c], skipna=True) in _MIXED]
    if not mixed:
        return [frame]
    groups: dict = {}
    for position, key in enumerate(zip(*(map(_kind, frame[c].tolist()) for c in mixed))):
        groups.setdefault(key, []).append(position)
    parts = []
    for key, positions in groups.items():
        part = frame.iloc[positions].copy()
        for column, kind in zip(mixed, key):
            if kind == "f":
                part[column] = part[column].astype("float64")
            elif kind == "b":
                part[column] = part[column].astype(bool)
            elif kind == "s":
                part[column] = part[column].astype(str)
        parts.append(part.reset_index(drop=True))
    return parts


# ---------------------------------------------------------------------------------------------------------- targets
class Target:
    """How a channel writes: frame by frame (`write_frame`), or as line protocol (`encode` + `send`) that can share a
    request with the other channels of the same `key` (same client and bucket)

    `ping` (optional) is used to detect the end of an outage without waiting for the back-off delay. `idempotent`:
    writing the same rows twice stores them once (InfluxDB points with their timestamp), so the rows of an uncertain
    write can be written once more; None (default): True for line-protocol targets, False for `write_frame` ones.
    """

    def __init__(
        self,
        write_frame: Optional[WriteFunction] = None,
        encode: Optional[Callable[[pd.DataFrame, dict], list]] = None,
        send: Optional[Callable[[list], None]] = None,
        ping: Optional[Callable[[], bool]] = None,
        key=None,
        idempotent: Optional[bool] = None,
    ):
        if write_frame is None and (encode is None or send is None):
            raise ValueError("a Target needs write_frame, or encode and send")
        self.write_frame = write_frame
        self.encode = encode
        self.send = send
        self.ping = ping
        self.key = key if encode is not None else None
        self.idempotent = self.lines if idempotent is None else idempotent

    @property
    def lines(self) -> bool:
        return self.encode is not None


class _Endpoint:
    """Availability of one storage destination: exponential back-off after a failure, probed while waiting"""

    def __init__(self, ping, batch_rows: int):
        self.ping = ping
        # adaptive batch size: rows per request, lowered after a timed-out request, raised again after fast writes
        self.batch_rows = batch_rows
        self.timeout_s = None  # duration of the last request that timed out (the write timeout, as seen)
        self.shrinks = 0
        self.failures = 0
        self.retry_at = 0.0
        self.next_probe = 0.0
        self.probing = False
        self.generation = 0  # incremented by every failure: an older success must not clear a newer back-off
        self.held_rows = 0  # queued rows of its channels that wait for the end of a back-off (no spool, not closing)
        # second writes of uncertain rows wait for an answer to ping (see SharedWriter._may_retry)
        self.gate_failures = 0
        self.gate_until = 0.0
        self.answered_at = None


@dataclass
class _Entry:
    frame: pd.DataFrame
    meta: dict
    queued_at: float
    wal: Optional[Path] = None
    seq: int = 0  # submission order in the channel, to put rows back in their place
    retried: bool = False  # sent once with an uncertain outcome (may be stored): never retried again, lost = uncertain


@dataclass
class _Unit:
    """Rows of one channel sent together; `sizes` keeps the original batches, so that a refused unit can be split"""

    channel: "Channel"
    frame: pd.DataFrame
    meta: dict
    sizes: list
    replay: bool = False
    files: list = field(default_factory=list)  # write-ahead or spool files holding exactly these rows
    whole: bool = True  # False once split: its files then hold more rows than it does
    lines: Optional[list] = None
    origins: Optional[list] = None  # (seq, queued_at) of each original batch (one per size), to put it back in order
    retried: bool = False  # as _Entry.retried
    top: Optional["_Unit"] = None  # unit taken from the queues that holds these rows (None: this one)
    released: int = 0  # rows of this (top) unit put back in the queue: no longer in flight

    @property
    def rows(self) -> int:
        return len(self.frame)

    def encoded(self) -> list:
        if self.lines is None:
            self.lines = self.channel.target.encode(self.frame, self.meta)
        return self.lines

    def split(self) -> list:
        """By original batch, then in halves down to single rows"""
        origins = self.origins if self.origins and len(self.origins) == len(self.sizes) else None
        if len(self.sizes) > 1:
            bounds, start = [], 0
            for k, size in enumerate(self.sizes):
                bounds.append((start, start + size, [size], [origins[k]] if origins else self.origins))
                start += size
        else:
            half = self.rows // 2
            bounds = [(0, half, [half], self.origins), (half, self.rows, [self.rows - half], self.origins)]
        return [
            _Unit(
                self.channel,
                self.frame.iloc[a:b].reset_index(drop=True),
                self.meta,
                sizes,
                self.replay,
                whole=False,
                origins=part_origins,
                retried=self.retried,
                top=self.top or self,
            )
            for a, b, sizes, part_origins in bounds
            if b > a
        ]


class Channel:
    """The queue and the counters of one Pulse inside a SharedWriter"""

    def __init__(
        self,
        writer: "SharedWriter",
        target: Target,
        meta: dict,
        max_queue_rows: int,
        spool: Optional[Spool],
        durability: Literal["memory", "wal"],
        replay_max_rows_per_s: Optional[float],
        endpoint: _Endpoint,
        retry_uncertain: bool = True,
        spool_overflow: bool = True,
        bound_in_flight: bool = False,
    ):
        if durability == "wal" and spool is None:
            raise ValueError('durability="wal" needs a spool directory')
        self.writer = writer
        self.target = target
        self.meta = meta
        self.max_queue_rows = max_queue_rows
        self.spool = spool
        self.durability = durability
        self.replay_max_rows_per_s = replay_max_rows_per_s
        self.endpoint = endpoint
        self.retry_uncertain = retry_uncertain
        self.spool_overflow = spool_overflow and spool is not None
        self.bound_in_flight = bound_in_flight
        self.queue: deque = deque()
        self.queued_rows = 0
        self.pending_rows = 0  # taken by the writer, not resolved yet (in flight)
        self.max_held_rows = 0  # peak of the rows queued and in flight
        # batches beyond max_queue_rows waiting for the spooler thread (spool_overflow), bounded by max_queue_rows too
        self.overflow: deque = deque()
        self.overflow_rows = 0  # rows in `overflow`, plus those of the batch being saved
        self.closing = False  # close() started: rows are no longer kept in the queue during a back-off
        self.closed = False
        self._seq = 0
        self._replay_budget = 0.0
        self._replay_at = time.monotonic()
        self._stats = {
            "captured_rows": 0,
            "written_rows": 0,
            "spooled_rows": 0,
            "replayed_rows": 0,
            "dropped_rows": 0,
            "rejected_rows": 0,
            "uncertain_rows": 0,
            "retried_uncertain_rows": 0,
            "overflow_spooled_rows": 0,
            "write_errors": 0,
            "write_timeouts": 0,
            "memory_only_rows": 0,
            "last_error": None,
            "last_write_ms": None,
            "max_write_ms": None,
        }

    # ------------------------------------------------------------------ caller side
    def submit(self, frame: pd.DataFrame) -> None:
        rows = len(frame)
        if rows == 0:
            return
        wal = self.spool.save(frame, self.meta, wal=True) if self.durability == "wal" else None
        writer = self.writer
        with writer._condition:
            if self.closed:
                raise RuntimeError("writer channel is closed")
            self._stats["captured_rows"] += rows
            if self.durability == "wal" and wal is None:  # spool full: this batch is kept in memory only
                self._stats["memory_only_rows"] += rows
                logger.warning("Spool full: batch of %d rows kept in memory only (no write-ahead copy)", rows)
            if rows > self.max_queue_rows:
                if wal is not None:  # kept on disk, replayed later
                    self.spool.promote(wal)
                    self._stats["spooled_rows"] += rows
                    logger.warning("Batch of %d rows larger than max_queue_rows (%d): spooled", rows, self.max_queue_rows)
                    return
                extra = rows - self.max_queue_rows
                head = frame.iloc[:extra].reset_index(drop=True)
                frame = frame.iloc[extra:].reset_index(drop=True)
                rows = len(frame)
                if self.spool_overflow:
                    self._overflow(_Entry(head, self.meta, time.monotonic(), seq=self._seq + 1))
                else:
                    self._stats["dropped_rows"] += extra
                    logger.warning("Batch larger than max_queue_rows (%d): %d oldest rows dropped", self.max_queue_rows, extra)
            self._seq += 1
            self.queue.append(_Entry(frame, self.meta, time.monotonic(), wal, self._seq))
            self._queued(rows)
            self._bound()
            if writer._ready(self, time.monotonic()) and writer._ready_rows() >= writer._trigger_rows:
                writer._condition.notify_all()

    def _queued(self, rows: int) -> None:
        """Count rows entering (or, negative, leaving) the queue; called with the writer's condition held"""
        self.queued_rows += rows
        self.writer._queued_rows += rows
        if self.spool is None and not self.closing:
            self.endpoint.held_rows += rows

    def _held(self) -> int:
        """Rows counted against max_queue_rows: those queued, plus those in flight with bound_in_flight"""
        return self.queued_rows + self.pending_rows if self.bound_in_flight else self.queued_rows

    def _bound(self) -> None:
        """Drop (or spool) the oldest batches beyond max_queue_rows; called with the writer's condition held"""
        while self._held() > self.max_queue_rows and len(self.queue) > 1:
            dropped = self.queue.popleft()
            size = len(dropped.frame)
            self._queued(-size)
            if dropped.wal is not None:
                self.spool.promote(dropped.wal)
                self._stats["spooled_rows"] += size
                self._stats["overflow_spooled_rows"] += size
            elif self.spool_overflow and not self.closed:
                self._overflow(dropped)
            else:
                self._lost(size, dropped.retried)
        if self.queued_rows + self.pending_rows > self.max_held_rows:
            self.max_held_rows = self.queued_rows + self.pending_rows

    def _overflow(self, entry: "_Entry") -> None:
        """A batch beyond max_queue_rows, with a spool: saved to the spool by the spooler thread (never in the caller's
        thread). The batches waiting for it are bounded by max_queue_rows too: beyond, the oldest are dropped and
        counted. Called with the writer's condition held"""
        self.overflow.append(entry)
        self.overflow_rows += len(entry.frame)
        while self.overflow_rows > self.max_queue_rows and len(self.overflow) > 1:
            lost = self.overflow.popleft()
            self.overflow_rows -= len(lost.frame)
            self._lost(len(lost.frame), lost.retried)
        self.writer._wake_spooler()

    def _save_overflow(self, entry: "_Entry") -> None:
        """Save one overflow batch to the spool (without the writer's condition held), then count it"""
        size = len(entry.frame)
        try:
            saved = self.spool.save(entry.frame, entry.meta) is not None  # refused by a full spool: spool_evicted_rows
        except Exception:  # noqa: BLE001 - disk error: counted, the writer goes on
            logger.exception("Overflow batch of %d rows could not be spooled", size)
            saved = None
        with self.writer._condition:
            self.overflow_rows -= size
            if saved:
                self._stats["spooled_rows"] += size
                self._stats["overflow_spooled_rows"] += size
            elif saved is None:
                self._lost(size, entry.retried)
            self.writer._condition.notify_all()

    def _lost(self, rows: int, retried: bool) -> None:
        """Rows dropped by the queue bound; those of an uncertain write (`retried`: they may be stored) are counted in
        uncertain_rows, the others in dropped_rows. Called with the writer's condition held"""
        if retried:
            self._stats["uncertain_rows"] += rows
            logger.warning("Capture queue full (%d rows): %d rows of an uncertain write not retried", self.max_queue_rows, rows)
        else:
            self._stats["dropped_rows"] += rows
            logger.warning("Capture queue full (%d rows): %d oldest rows dropped", self.max_queue_rows, rows)

    def _take_head(self, rows: int) -> "_Entry":
        """The first `rows` rows of the oldest batch, the rest staying at the head of the queue (a request limited by
        the adaptive batch size); called with the writer's condition held"""
        entry = self.queue[0]
        head = _Entry(entry.frame.iloc[:rows].reset_index(drop=True), entry.meta, entry.queued_at, None, entry.seq, entry.retried)
        entry.frame = entry.frame.iloc[rows:].reset_index(drop=True)
        return head

    def requeue(self, unit: "_Unit", retry: bool = False) -> bool:
        """Put rows back in the queue: rows certainly not stored (not sent because of a back-off, or refused
        connection), or, with `retry`, the rows of an uncertain write for their single second write. Each original
        batch goes back in its place (by submission order), within max_queue_rows like any batch (beyond, the oldest
        batches are dropped and counted; rows that may be stored in uncertain_rows); False when the channel is closing
        (the caller then counts them)

        Rows of object columns mixing kinds were split by kind before the write: they go back as one batch per kind,
        in the place of their oldest batch.
        """
        origins = unit.origins or [(0, time.monotonic())]
        if len(origins) == len(unit.sizes) and len(origins) > 1:
            pieces, start = [], 0
            for size, origin in zip(unit.sizes, origins):
                pieces.append((unit.frame.iloc[start : start + size].reset_index(drop=True), origin))
                start += size
        else:
            pieces = [(unit.frame, origins[0])]
        with self.writer._condition:
            if self.closing or self.closed:
                return False
            if not unit.replay:  # no longer in flight (the cleanup of its top unit does not count them again)
                (unit.top or unit).released += unit.rows
                self.pending_rows -= unit.rows
            retried = unit.retried or retry
            if retry:
                self._stats["retried_uncertain_rows"] += unit.rows
            entries = []
            for frame, (seq, queued_at) in sorted(pieces, key=lambda piece: piece[1][0]):  # stable: same seq kept in order
                rows = len(frame)
                if rows > self.max_queue_rows:  # as in submit(): only the newest rows fit
                    extra = rows - self.max_queue_rows
                    frame = frame.iloc[extra:].reset_index(drop=True)
                    rows = len(frame)
                    self._lost(extra, retried)
                entries.append(_Entry(frame, unit.meta, queued_at, seq=seq, retried=retried))
                self._queued(rows)
            # one merge by submission order (each piece after the queued batches of the same or a lower seq): inserting
            # piece by piece scanned the deque for each, seconds with the lock held for thousands of one-row captures
            merged, queued = deque(), iter(self.queue)
            current = next(queued, None)
            for entry in entries:
                while current is not None and current.seq <= entry.seq:
                    merged.append(current)
                    current = next(queued, None)
                merged.append(entry)
            if current is not None:
                merged.append(current)
                merged.extend(queued)
            self.queue = merged
            self._bound()
            return True

    def flush(self, timeout_s: float = 30.0) -> bool:
        """Wait until every submitted row has been written, spooled or dropped; True if done within the timeout"""
        deadline = time.monotonic() + timeout_s
        writer = self.writer
        with writer._condition:
            writer._flush_requests += 1
            writer._condition.notify_all()
            try:
                while self.queue or self.pending_rows or self.overflow_rows:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    writer._condition.wait(min(remaining, 0.05))  # woken by the writer when rows are resolved
            finally:
                writer._flush_requests -= 1
        return True

    def close(self, timeout_s: float = 30.0) -> None:
        """Write (or spool) what is pending, then detach from the writer; rows kept in the queue during a back-off are
        tried once more, then spooled or dropped and counted, without waiting for the end of the back-off"""
        with self.writer._condition:
            if self.spool is None and not self.closing:
                self.endpoint.held_rows -= self.queued_rows  # its rows no longer wait for the end of a back-off
            self.closing = True
            self.writer._condition.notify_all()
        self.flush(timeout_s)
        with self.writer._condition:
            self.closed = True
            entries = list(self.queue)
            self.queue.clear()
            self._queued(-self.queued_rows)
            overflow = list(self.overflow)
            self.overflow.clear()
        for entry in overflow:  # the spooler did not finish in time: saved here
            self._save_overflow(entry)
        for entry in entries:  # the writer did not finish in time
            self.writer._fallback(
                _Unit(
                    self,
                    entry.frame,
                    entry.meta,
                    [len(entry.frame)],
                    files=[entry.wal] if entry.wal else [],
                    retried=entry.retried,
                )
            )
        self.writer._unregister(self)

    def stats(self) -> dict:
        writer = self.writer
        with writer._condition:
            stats = dict(self._stats)
            stats["queued_rows"] = self.queued_rows
            stats["in_flight_rows"] = self.pending_rows
            stats["max_held_rows"] = self.max_held_rows
            stats["batch_limit_rows"] = writer._limit(self.endpoint)
            stats["batch_shrinks"] = self.endpoint.shrinks
            stats["storage_available"] = time.monotonic() >= self.endpoint.retry_at
            stats["writer_lag_s"] = time.monotonic() - self.queue[0].queued_at if self.queue else 0.0
        if self.spool:
            stats["spool_files"] = len(self.spool.files())
            stats["spool_bytes"] = self.spool.size_bytes()
            stats["spool_evicted_rows"] = self.spool.evicted_rows
            stats["dropped_rows"] += self.spool.evicted_rows
        else:
            stats["spool_files"] = 0
        return stats

    # ------------------------------------------------------------------ replay throttle
    def replay_allowance(self, now: float, wanted: int) -> int:
        if self.replay_max_rows_per_s is None:
            return wanted
        rate = self.replay_max_rows_per_s
        self._replay_budget = min(max(rate, 1.0), self._replay_budget + (now - self._replay_at) * rate)
        self._replay_at = now
        return int(self._replay_budget) if self._replay_budget >= 1 else 0

    def replay_spent(self, rows: int) -> None:
        if self.replay_max_rows_per_s is not None:
            self._replay_budget -= rows


# ---------------------------------------------------------------------------------------------------------- writer
@dataclass
class _Job:
    units: list
    tops: list  # units whose files are cleaned once the job is over


class SharedWriter:
    """One writer thread (and up to `concurrency` requests in flight) for several Pulse instances of a process

    - Each channel (Pulse) keeps its own bounded queue, spool and counters; the thread takes up to `max_batch_rows` rows
      across channels, every `flush_interval_s` seconds or as soon as that many rows are waiting.
    - Frames are grouped by channel, columns and dtypes; channels with the same line-protocol target share requests.
    - After a failed write, the destination is considered unavailable for an exponential delay (1 s, 2 s, 4 s... up to
      `max_retry_delay_s`); every `probe_interval_s` a cheap probe (InfluxDB /ping) checks whether it is back, so that
      writing resumes within about a second of the end of an outage. Meanwhile the batches of its channels go to their
      spool or, without spool, stay in their queue (within `max_queue_rows`) until it is back.
    - A refused write (HTTP 400 / 422) is counted exactly: from the `dropped=N` of a partial write when the request
      held one batch, otherwise by writing again batch by batch, then in halves down to single rows.
    - A write that timed out, or whose connection was cut before the answer, may have been stored: spooled (replaying is
      idempotent) or, without spool, put back in the queue for one second write, sent once the destination answers its
      ping (`retry_uncertain`, default; idempotent targets only; counted in
      `retried_uncertain_rows`), then counted in `uncertain_rows` if that one is uncertain too or cannot be made. A
      write that failed otherwise stored nothing: its rows go back to the queue (or to the spool) like those of the
      back-off.
    - Rows waiting for the end of a back-off neither count towards `max_batch_rows` (they do not trigger requests of
      the other channels) nor delay the replay of the other spools. Without spool, the memory held during an outage is
      bounded per channel only: number of channels x `max_queue_rows`.
    - Adaptive batch size (`adaptive_batch`, default): a request that timed out divides the rows per request of its
      destination by 4 (not below `min_batch_rows`), so that a slow link (bandwidth x write timeout smaller than a
      request) still carries rows instead of timing out forever; a write fast enough to be doubled within half the
      observed timeout doubles it again, up to `max_batch_rows`. Live requests and spool replays both follow it.
    - With a spool, the batches beyond `max_queue_rows` of a channel (storage answering, but slower than the traffic)
      are saved to the spool by a spooler thread instead of being dropped (`spool_overflow`), then replayed once the
      queue has drained.
    """

    _default = None
    _default_lock = threading.Lock()

    def __init__(
        self,
        max_batch_rows: int = 5_000,
        flush_interval_s: float = 1.0,
        max_retry_delay_s: float = 60.0,
        concurrency: int = 2,
        probe_interval_s: Optional[float] = 1.0,
        name: str = "pulsar-data-collection-writer",
        adaptive_batch: bool = True,
        min_batch_rows: int = 10,
    ):
        if concurrency < 1 or max_batch_rows < 1 or flush_interval_s <= 0 or min_batch_rows < 1:
            raise ValueError("concurrency, max_batch_rows, min_batch_rows and flush_interval_s must be positive")
        self.max_batch_rows = max_batch_rows
        self.adaptive_batch = adaptive_batch
        self.min_batch_rows = min(min_batch_rows, max_batch_rows)
        self._trigger_rows = max_batch_rows  # rows that start a request at once: the smallest adaptive batch size
        self.flush_interval_s = flush_interval_s
        self.max_retry_delay_s = max_retry_delay_s
        self.concurrency = concurrency
        self.probe_interval_s = probe_interval_s
        self._condition = threading.Condition()
        self._channels: list = []
        self._endpoints: dict = {}  # shared endpoints, by target key
        self._destinations: dict = {}  # every endpoint of the channels, by id: [endpoint, channels]
        self._queued_rows = 0
        self._inflight = 0
        self._flush_requests = 0
        self._busy = False
        self._stopping = False
        self._next = 0
        self._name = name
        self._spooler = None  # thread saving the overflow batches of the channels with a spool, started on first use
        self._executor = (
            ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix=f"{name}-send") if concurrency > 1 else None
        )
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ process default
    @classmethod
    def default(cls, **kwargs) -> "SharedWriter":
        """The process-wide shared writer, created on first use (with `kwargs`, ignored afterwards)"""
        with cls._default_lock:
            if cls._default is None or cls._default._stopping:
                cls._default = cls(**kwargs)
            return cls._default

    # ------------------------------------------------------------------ channels
    def register(
        self,
        target: Target,
        meta: dict,
        max_queue_rows: int = 100_000,
        spool: Optional[Spool] = None,
        durability: Literal["memory", "wal"] = "memory",
        replay_max_rows_per_s: Optional[float] = None,
        retry_uncertain: bool = True,
        spool_overflow: bool = True,
        bound_in_flight: bool = False,
    ) -> Channel:
        with self._condition:
            if self._stopping:
                raise RuntimeError("writer is closed")
            if target.key is not None:
                endpoint = self._endpoints.get(target.key)
                if endpoint is None:
                    endpoint = self._endpoints[target.key] = _Endpoint(target.ping, self.max_batch_rows)
            else:
                endpoint = _Endpoint(target.ping, self.max_batch_rows)
            channel = Channel(
                self,
                target,
                meta,
                max_queue_rows,
                spool,
                durability,
                replay_max_rows_per_s,
                endpoint,
                retry_uncertain,
                spool_overflow=spool_overflow,
                bound_in_flight=bound_in_flight,
            )
            self._channels.append(channel)
            self._destinations.setdefault(id(endpoint), [endpoint, 0])[1] += 1
            self._condition.notify_all()
            return channel

    def _unregister(self, channel: Channel) -> None:
        with self._condition:
            if channel in self._channels:
                self._channels.remove(channel)
                destination = self._destinations.get(id(channel.endpoint))
                if destination is not None:
                    destination[1] -= 1
                    if not destination[1]:
                        del self._destinations[id(channel.endpoint)]
                        self._update_trigger()
            if channel.target.key is not None and not any(c.target.key == channel.target.key for c in self._channels):
                self._endpoints.pop(channel.target.key, None)

    def channels(self) -> int:
        with self._condition:
            return len(self._channels)

    def flush(self, timeout_s: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            channels = list(self._channels)
        return all(c.flush(max(0.0, deadline - time.monotonic())) for c in channels)

    def close(self, timeout_s: float = 30.0) -> None:
        """Write (or spool) what is pending in every channel, then stop the thread"""
        deadline = time.monotonic() + timeout_s
        with self._condition:
            channels = list(self._channels)
        for channel in channels:
            channel.close(max(0.0, deadline - time.monotonic()))
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._thread.join(max(0.1, deadline - time.monotonic()))
        if self._spooler is not None:
            self._spooler.join(max(0.1, deadline - time.monotonic()))
        if self._executor is not None:
            self._executor.shutdown(wait=True)

    # ------------------------------------------------------------------ writer thread
    def _take(self) -> list:
        """Up to max_batch_rows rows, taking channels in turn (not those whose rows wait for the end of a back-off)"""
        taken, rows = [], 0
        now = time.monotonic()
        channels = [c for c in self._channels if c.queue and self._ready(c, now)]
        if not channels:
            return taken
        start = self._next % len(channels)
        self._next += 1
        order = channels[start:] + channels[:start]
        per_endpoint: dict = {}  # rows taken for each destination whose adaptive batch size is lowered
        while rows < self.max_batch_rows and order:
            for channel in list(order):
                if not channel.queue:
                    order.remove(channel)
                    continue
                entry = channel.queue[0]
                size = len(entry.frame)
                limit = self._limit(channel.endpoint)
                if limit < self.max_batch_rows:
                    used = per_endpoint.get(id(channel.endpoint), 0)
                    if used + size > limit:
                        if used:  # this request of its destination is full
                            order.remove(channel)
                            continue
                        if entry.wal is None:  # a batch larger than the request: its oldest rows now, the rest later
                            entry = channel._take_head(limit)
                            size = limit
                    per_endpoint[id(channel.endpoint)] = used + size
                if entry is channel.queue[0]:
                    channel.queue.popleft()
                channel._queued(-size)
                channel.pending_rows += size
                taken.append((channel, entry))
                rows += size
                if rows >= self.max_batch_rows:
                    break
        return taken

    @staticmethod
    def _ready(channel: Channel, now: float) -> bool:
        """The queued rows of a channel can be taken: its storage is available, or they go to its spool anyway, or it
        is closing (its rows are then written, spooled or dropped without waiting)"""
        return now >= channel.endpoint.retry_at or channel.spool is not None or channel.closing

    def _ready_rows(self) -> int:
        """Queued rows that can be taken now (not those waiting for the end of a back-off); with the condition held.
        One term per destination, not per channel: it runs on every submit"""
        if not self._queued_rows:
            return 0
        now = time.monotonic()
        return self._queued_rows - sum(e.held_rows for e, _ in self._destinations.values() if e.held_rows and now < e.retry_at)

    def _run(self) -> None:
        while True:
            with self._condition:
                ready = self._ready_rows()
                if not self._stopping and ready < self._trigger_rows and not (self._flush_requests and ready):
                    self._condition.wait(self.flush_interval_s)
                while self._inflight >= self.concurrency and not self._stopping:
                    self._condition.wait(0.05)
                taken = self._take()
                self._busy = True
                stopping = self._stopping and not self._queued_rows
            try:
                if taken:
                    try:
                        jobs = self._jobs(taken)
                    except Exception:  # noqa: BLE001 - never leave taken rows unresolved
                        logger.exception("Unexpected error while preparing a write")
                        self._abandon(taken)
                    else:
                        self._dispatch(jobs)
                self._probe_endpoints()
                self._replay_spools()
            except Exception:  # noqa: BLE001 - the writer thread must never die
                logger.exception("Unexpected error in the capture writer")
            finally:
                with self._condition:
                    self._busy = False
                    if taken:  # nothing changed otherwise: waking the flush() waiters would only spin with them
                        self._condition.notify_all()
            if stopping:
                with self._condition:
                    while self._inflight:
                        self._condition.wait(0.05)
                return

    def _jobs(self, taken: list, replay: bool = False) -> list:
        """Group the entries by channel, columns and dtypes, then put the line-protocol units of a target together"""
        groups: dict = {}
        for channel, entry in taken:
            key = (id(channel), id(entry.meta), _signature(entry.frame), entry.retried)
            groups.setdefault(key, (channel, entry.meta, []))[2].append(entry)
        tops, units = [], []
        for (_, _, signature, retried), (channel, meta, entries) in groups.items():
            frames = [e.frame for e in entries]
            frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
            files = [e.wal for e in entries if e.wal is not None]
            # whole only if its files hold all its rows: otherwise a fallback saves the rows instead of promoting
            whole = not files or len(files) == len(entries)
            origins = [(e.seq, e.queued_at) for e in entries]
            sizes = [len(f) for f in frames]
            top = _Unit(channel, frame, meta, sizes, replay=replay, files=files, whole=whole, origins=origins, retried=retried)
            tops.append(top)
            parts = split_mixed_objects(frame, signature)
            if len(parts) == 1:
                units.append(top)
            else:
                units.extend(
                    _Unit(
                        channel,
                        part,
                        meta,
                        [len(part)],
                        replay=replay,
                        whole=False,
                        origins=origins[:1],
                        retried=retried,
                        top=top,
                    )
                    for part in parts
                )
        jobs, shared = [], {}
        for unit in units:
            key = unit.channel.target.key
            if key is None:
                jobs.append(_Job([unit], []))
            else:
                shared.setdefault(key, []).append(unit)
        jobs.extend(_Job(group, []) for group in shared.values())
        if jobs:
            jobs[-1].tops.extend(tops)  # cleaned once every job of this batch is over (jobs run in order per batch)
        return jobs

    def _abandon(self, taken: list) -> None:
        """Entries that cannot be written: spooled if possible, otherwise dropped, and always resolved"""
        for channel, entry in taken:
            size = len(entry.frame)
            try:
                self._fallback(
                    _Unit(
                        channel,
                        entry.frame,
                        entry.meta,
                        [size],
                        files=[entry.wal] if entry.wal else [],
                        retried=entry.retried,
                    ),
                    "unexpected writer error",
                )
            except Exception:  # noqa: BLE001
                logger.exception("Rows lost after an unexpected writer error")
                with self._condition:
                    channel._stats["uncertain_rows" if entry.retried else "dropped_rows"] += size
            finally:
                with self._condition:
                    channel.pending_rows -= size
                    self._condition.notify_all()

    def _dispatch(self, jobs: list) -> None:
        if self._executor is None or len(jobs) == 0:
            for job in jobs:
                self._run_job(job)
            return
        batch = _Batch(len(jobs), [top for job in jobs for top in job.tops])
        for job in jobs:
            job.tops = []
            with self._condition:
                while self._inflight >= self.concurrency:
                    self._condition.wait(0.05)
                self._inflight += 1
            try:
                self._executor.submit(self._run_job_async, job, batch)
            except RuntimeError:  # executor shut down: run it here, so that its rows are resolved
                self._run_job_async(job, batch)

    def _run_job_async(self, job: "_Job", batch: "_Batch") -> None:
        try:
            self._run_job(job)
        except Exception:  # noqa: BLE001
            logger.exception("Unexpected error in the capture writer")
        finally:
            if batch.done():
                self._cleanup(batch.tops)
            with self._condition:
                self._inflight -= 1
                self._condition.notify_all()

    def _run_job(self, job: "_Job") -> None:
        try:
            self._attempt(job.units)
        except Exception:  # noqa: BLE001
            logger.exception("Unexpected error in the capture writer")
            for unit in job.units:
                self._fallback(unit)
        finally:
            if job.tops:
                self._cleanup(job.tops)

    def _cleanup(self, tops: list) -> None:
        """Delete the write-ahead / replayed files of rows now resolved (kept files were promoted or left to replay)"""
        for top in tops:
            for path in top.files:
                if top.channel.spool is not None:
                    top.channel.spool.remove(path)
            top.files = []
            with self._condition:
                if not top.replay:
                    top.channel.pending_rows -= top.rows - top.released
                self._condition.notify_all()

    # ------------------------------------------------------------------ one request
    def _available(self, endpoint: _Endpoint) -> bool:
        now = time.monotonic()
        with self._condition:
            if now >= endpoint.retry_at:
                return True
            probe = (
                endpoint.ping is not None
                and self.probe_interval_s is not None
                and not endpoint.probing
                and now >= endpoint.next_probe
            )
            if probe:
                endpoint.probing = True
                endpoint.next_probe = now + self.probe_interval_s
        if probe:
            threading.Thread(target=self._probe, args=(endpoint,), name="pulsar-storage-probe", daemon=True).start()
        return False

    def _probe_endpoints(self) -> None:
        """Probe the destinations in back-off even without traffic, so that the next rows find them available"""
        with self._condition:
            endpoints = {id(c.endpoint): c.endpoint for c in self._channels}.values()
        for endpoint in endpoints:
            if endpoint.retry_at:
                self._available(endpoint)

    def _probe(self, endpoint: _Endpoint) -> None:
        try:
            up = endpoint.ping()
        except Exception:  # noqa: BLE001
            up = False
        with self._condition:
            endpoint.probing = False
            if up:
                self._answered(endpoint)
            if up and endpoint.retry_at:
                endpoint.retry_at = 0.0
                logger.info("Storage answers again: writing resumes")
                self._condition.notify_all()

    def _send(self, units: list) -> None:
        target = units[0].channel.target
        if target.lines:
            lines = []
            for unit in units:
                lines.extend(unit.encoded())
            target.send(lines)
        else:
            target.write_frame(units[0].frame, units[0].meta)

    def _encodable(self, units: list) -> list:
        """Encode each unit on its own: rows that cannot be serialized are refused, never blamed on the storage"""
        if not units[0].channel.target.lines:
            return units
        good = []
        for unit in units:
            try:
                unit.encoded()
            except Exception as error:  # noqa: BLE001 - e.g. non-string column names
                self._unencodable(unit, error)
            else:
                good.append(unit)
        return good

    def _unencodable(self, unit: _Unit, error: Exception) -> None:
        with self._condition:
            unit.channel._stats["write_errors"] += 1
            unit.channel._stats["last_error"] = f"{type(error).__name__}: {error}"
        if unit.replay and unit.whole and unit.files and unit.channel.spool is not None:
            for path in unit.files:  # kept for inspection, never replayed again
                unit.channel.spool.set_aside(path)
            unit.files = []
        self._reject(unit, unit.rows, error)

    def _attempt(self, units: list) -> None:
        endpoint = units[0].channel.endpoint
        if not self._available(endpoint):
            for unit in units:
                self._fallback(unit, "storage unavailable", unsent=True)
            return
        units = self._encodable(units)
        if not units:
            return
        if endpoint.ping is not None and any(u.retried for u in units) and not self._may_retry(endpoint):
            # the single second write of uncertain rows is only spent on a storage that answers: right after a cut,
            # the back-off often ends while the link is still down (flapping link, backend behind a port forwarder).
            # Only those rows wait: the other rows of the request are sent now
            for unit in units:
                if unit.retried:
                    self._fallback(unit, "storage not answering", unsent=True)
            units = [u for u in units if not u.retried]
            if not units:
                return
        with self._condition:
            generation = endpoint.generation
        started = time.perf_counter()
        try:
            self._send(units)
        except StorageRejectedError as error:
            self._timing(units, started)
            self._rejected(units, error, generation)
            return
        except Exception as error:  # noqa: BLE001
            elapsed = self._timing(units, started)
            uncertain = _uncertain(error)
            # a request that timed out and is larger than the (new) adaptive batch size: its rows are written again in
            # smaller requests, even if they had their second write already (bounded: the size goes down each time)
            resized = uncertain and _timeout(error) and self._shrink(endpoint, sum(u.rows for u in units), elapsed)
            self._record_failure(units, error, uncertain)
            for unit in units:  # certainly not stored unless uncertain: kept in the queue like the rows of the back-off
                self._fallback(unit, str(error), uncertain=uncertain, unsent=not uncertain, resized=resized)
            return
        elapsed = self._timing(units, started)
        self._record_success(endpoint, generation)
        self._grow(endpoint, sum(u.rows for u in units), elapsed)
        for unit in units:
            self._count(unit, unit.rows)

    def _rejected(self, units: list, error: StorageRejectedError, generation: int) -> None:
        with self._condition:
            for channel in {id(u.channel): u.channel for u in units}.values():
                channel._stats["write_errors"] += 1
                channel._stats["last_error"] = f"{type(error).__name__}: {error}"
        self._record_success(units[0].channel.endpoint, generation)  # the storage answered: it is up
        if len(units) > 1:  # several channels or groups: the refused rows cannot be attributed, write them one by one
            for unit in units:
                self._attempt([unit])
            return
        unit = units[0]
        dropped = getattr(error, "dropped", None)
        if dropped is not None and dropped <= unit.rows:
            self._count(unit, unit.rows - dropped)
            self._reject(unit, dropped, error)
            return
        if unit.rows > 1:
            for part in unit.split():
                self._attempt([part])
            return
        self._reject(unit, unit.rows, error)

    # ------------------------------------------------------------------ outcomes
    def _count(self, unit: _Unit, rows: int) -> None:
        if rows <= 0:
            return
        with self._condition:
            unit.channel._stats["replayed_rows" if unit.replay else "written_rows"] += rows
        if unit.replay:
            logger.info("Replayed %d spooled rows", rows)

    def _reject(self, unit: _Unit, rows: int, error: Exception) -> None:
        """Rows the storage refused: counted, never spooled nor retried"""
        if rows <= 0:
            return
        with self._condition:
            unit.channel._stats["rejected_rows"] += rows
        logger.error("%d rows rejected by the storage, not retried: %s", rows, error)

    def _fallback(
        self,
        unit: _Unit,
        reason: str = "storage unavailable",
        uncertain: bool = False,
        unsent: bool = False,
        resized: bool = False,
    ) -> None:
        """Rows not written: left to replay, spooled, kept in the queue (`unsent`: certainly not stored, without
        spool), put back in the queue once for a second write (`uncertain`, without spool: line-protocol writes are
        idempotent, a second write overwrites the same points), or counted in uncertain_rows or dropped_rows

        Rows already written once with an uncertain outcome (`unit.retried`) are never retried again after another
        uncertain write, unless that write timed out because its request was larger than the adaptive batch size
        (`resized`), and are counted in uncertain_rows whenever they are lost (they may be stored).
        """
        channel, spool = unit.channel, unit.channel.spool
        if unit.replay:
            if not unit.whole and spool is not None:  # its original files are deleted: keep these rows in a new one
                spool.save(unit.frame, unit.meta)
            else:
                unit.files = []  # left in the spool, replayed later
            return
        if spool is not None:
            if unit.whole and unit.files:
                for path in unit.files:
                    spool.promote(path)
                unit.files = []
                saved = True
            else:
                saved = spool.save(unit.frame, unit.meta) is not None
            with self._condition:
                if saved:
                    channel._stats["spooled_rows"] += unit.rows
                # refused by a full spool: counted in spool_evicted_rows (part of dropped_rows)
            return
        if unsent and channel.requeue(unit):  # not stored: kept in memory until the storage is back
            return
        retry = uncertain and (resized or not unit.retried) and channel.retry_uncertain and channel.target.idempotent
        if retry and channel.requeue(unit, retry=True):
            logger.warning("%d rows may or may not be stored (%s); written once more (idempotent)", unit.rows, reason)
            return
        uncertain = uncertain or unit.retried
        with self._condition:
            channel._stats["uncertain_rows" if uncertain else "dropped_rows"] += unit.rows
        if uncertain:
            logger.warning("%d rows may or may not be stored (%s); set spool_dir to keep them", unit.rows, reason)
        else:
            logger.warning("%d rows dropped (%s); set spool_dir to keep them", unit.rows, reason)

    def _timing(self, units: list, started: float) -> float:
        elapsed_ms = (time.perf_counter() - started) * 1000
        with self._condition:
            for channel in {id(u.channel): u.channel for u in units}.values():
                channel._stats["last_write_ms"] = round(elapsed_ms, 3)
                current = channel._stats["max_write_ms"]
                channel._stats["max_write_ms"] = round(max(current or 0.0, elapsed_ms), 3)
        return elapsed_ms / 1000

    # ------------------------------------------------------------------ adaptive batch size
    def _limit(self, endpoint: _Endpoint) -> int:
        """Rows per request to this destination now"""
        return endpoint.batch_rows if self.adaptive_batch else self.max_batch_rows

    def _update_trigger(self) -> None:
        """Called with the condition held, after an adaptive batch size changed"""
        limits = [e.batch_rows for e, _ in self._destinations.values()] if self.adaptive_batch else []
        self._trigger_rows = min([self.max_batch_rows, *limits])

    def _shrink(self, endpoint: _Endpoint, rows: int, elapsed_s: float) -> bool:
        """A request of `rows` rows timed out after `elapsed_s`: the next ones to this destination are 4 times smaller
        (a request needing more than bandwidth x timeout can never succeed, however often it is retried). True when
        the request was larger than the new size (its rows will go in smaller requests)"""
        if not self.adaptive_batch:
            return False
        with self._condition:
            endpoint.timeout_s = elapsed_s
            smaller = max(self.min_batch_rows, min(endpoint.batch_rows, rows // SHRINK_FACTOR))
            if smaller >= endpoint.batch_rows:
                return rows > endpoint.batch_rows  # shrunk already by a concurrent request
            endpoint.batch_rows = smaller
            endpoint.shrinks += 1
            self._update_trigger()
        logger.warning("Write of %d rows timed out after %.1f s: requests now hold %d rows", rows, elapsed_s, smaller)
        return True

    def _grow(self, endpoint: _Endpoint, rows: int, elapsed_s: float) -> None:
        """A request of `rows` rows succeeded in `elapsed_s`: double the rows per request of its destination when the
        doubled request is expected (linearly) to take less than half the timeout that was observed"""
        if not self.adaptive_batch or rows <= 0:
            return
        with self._condition:
            limit = endpoint.batch_rows
            if limit >= self.max_batch_rows or endpoint.timeout_s is None:
                return
            if elapsed_s * 2 * limit / rows > endpoint.timeout_s / 2:
                return
            endpoint.batch_rows = min(self.max_batch_rows, 2 * limit)
            self._update_trigger()
            grown = endpoint.batch_rows
        logger.info("Writes are fast again: requests to this storage now hold %d rows", grown)

    # ------------------------------------------------------------------ overflow spooler
    def _wake_spooler(self) -> None:
        """Called with the condition held by a channel that queued an overflow batch"""
        if self._spooler is None or not self._spooler.is_alive():
            self._spooler = threading.Thread(target=self._spool_overflow, name=f"{self._name}-spooler", daemon=True)
            self._spooler.start()
        self._condition.notify_all()

    def _spool_overflow(self) -> None:
        """Save the overflow batches of the channels to their spool, oldest first, outside the caller's thread"""
        while True:
            with self._condition:
                work = None
                while work is None:
                    work = next(((c, c.overflow.popleft()) for c in self._channels if c.overflow), None)
                    if work is None:
                        if self._stopping:
                            return
                        self._condition.wait(0.5)
            channel, entry = work
            try:
                channel._save_overflow(entry)
            except Exception:  # noqa: BLE001 - the spooler must never die
                logger.exception("Unexpected error in the capture spooler")

    def _may_retry(self, endpoint: _Endpoint) -> bool:
        """Whether the second write of uncertain rows can be sent: the destination answered its ping (within the last
        probe interval, or now). A failed ping holds these rows for an exponential delay of their own (the back-off of
        the destination and the other rows are left alone); after GATE_PINGS failed pings in a row they are sent anyway,
        since a ping can fail while writes work (a proxy exposing only the write API)"""
        now = time.monotonic()
        with self._condition:
            if endpoint.gate_failures >= GATE_PINGS:
                return True
            recent = self.probe_interval_s or 1.0
            if endpoint.answered_at is not None and now - endpoint.answered_at < recent:
                return True
            if now < endpoint.gate_until:
                return False
        try:
            answered = bool(endpoint.ping())
        except Exception:  # noqa: BLE001
            answered = False
        with self._condition:
            if answered:
                self._answered(endpoint)
                return True
            endpoint.gate_failures += 1
            endpoint.gate_until = time.monotonic() + min(2 ** (endpoint.gate_failures - 1), self.max_retry_delay_s)
            return endpoint.gate_failures >= GATE_PINGS

    @staticmethod
    def _answered(endpoint: _Endpoint) -> None:
        """The destination answered its ping; called with the condition held"""
        endpoint.answered_at = time.monotonic()
        endpoint.gate_failures = 0
        endpoint.gate_until = 0.0

    def _back_off_locked(self, endpoint: _Endpoint) -> float:
        endpoint.failures += 1
        endpoint.generation += 1
        delay = min(2 ** (endpoint.failures - 1), self.max_retry_delay_s)
        endpoint.retry_at = time.monotonic() + delay
        endpoint.next_probe = time.monotonic() + (self.probe_interval_s or delay)
        return delay

    def _record_failure(self, units: list, error: Exception, uncertain: bool) -> None:
        endpoint = units[0].channel.endpoint
        with self._condition:
            delay = self._back_off_locked(endpoint)
            for channel in {id(u.channel): u.channel for u in units}.values():
                channel._stats["write_errors"] += 1
                if uncertain and _timeout(error):
                    channel._stats["write_timeouts"] += 1
                channel._stats["last_error"] = f"{type(error).__name__}: {error}"
        logger.warning("Write to storage failed (%s); next attempt in %.0f s", error, delay)

    def _record_success(self, endpoint: _Endpoint, generation: int) -> None:
        with self._condition:
            if endpoint.generation != generation:  # a request started later failed: keep its back-off
                return
            if endpoint.failures:
                logger.info("Storage available again")
            endpoint.failures = 0
            endpoint.retry_at = 0.0

    # ------------------------------------------------------------------ spool replay
    def _replay_spools(self) -> None:
        with self._condition:
            channels = [c for c in self._channels if c.spool is not None]
        for channel in channels:
            for _ in range(10):  # bounded, so that live batches are not delayed for long
                with self._condition:
                    # live rows first: a full request of them is waiting (adaptive size included)
                    if self._ready_rows() >= self._trigger_rows or self._stopping:
                        return
                if not self._available(channel.endpoint):
                    break
                if not self._replay_once(channel):
                    break

    def _replay_once(self, channel: Channel) -> bool:
        """Write the oldest spooled batches of a channel (up to max_batch_rows rows); False when nothing was done"""
        limit = self._limit(channel.endpoint)
        allowance = channel.replay_allowance(time.monotonic(), limit)
        if allowance <= 0:
            return False
        taken, rows = [], 0
        spool = channel.spool
        for path in spool.files():
            if rows >= min(allowance, limit):
                break
            if not spool.hold(path):  # evicted meanwhile
                continue
            try:
                frame, meta = spool.load(path)
            except Exception:  # noqa: BLE001 - unreadable file: set aside, never retried
                logger.exception("Spool file %s cannot be read: renamed .bad", path.name)
                spool.set_aside(path)
                continue
            taken.append((path, frame, meta))
            rows += len(frame)
        if not taken:
            return False
        try:
            return self._replay_taken(channel, taken, rows)
        finally:
            spool.release([path for path, _, _ in taken])

    def _replay_taken(self, channel: Channel, taken: list, rows: int) -> bool:
        channel.replay_spent(rows)
        groups: dict = {}
        for path, frame, meta in taken:
            key = (repr(sorted(meta.items(), key=str)), _signature(frame))
            groups.setdefault(key, (meta, []))[1].append((path, frame))
        tops, units = [], []
        for meta, items in groups.values():
            frames = [f for _, f in items]
            frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
            top = _Unit(channel, frame, meta, [len(f) for f in frames], replay=True, files=[p for p, _ in items])
            tops.append(top)
            parts = split_mixed_objects(frame)
            limit = self._limit(channel.endpoint)
            if len(parts) == 1 and top.rows <= limit:
                units.append(top)
            else:  # a spooled batch larger than the adaptive batch size is replayed in requests of that size
                chunks = [p.iloc[a : a + limit].reset_index(drop=True) for p in parts for a in range(0, len(p), limit)]
                units.extend(_Unit(channel, c, meta, [len(c)], replay=True, whole=False) for c in chunks)
        failed = False
        for unit in units:
            before = channel.endpoint.retry_at
            self._attempt([unit])
            failed = failed or channel.endpoint.retry_at > before
        self._cleanup(tops)
        return not failed


class _Batch:
    """Jobs taken together: their write-ahead files are cleaned when the last one is over"""

    def __init__(self, jobs: int, tops: list):
        self._left = jobs
        self.tops = tops
        self._lock = threading.Lock()

    def done(self) -> bool:
        with self._lock:
            self._left -= 1
            return self._left == 0


def _uncertain(error: BaseException) -> bool:
    """A timeout or a connection cut after the request was sent: the storage may have stored the rows"""
    if isinstance(error, (StorageWriteUncertainError, TimeoutError, ConnectionResetError, ConnectionAbortedError)):
        return True  # builtin TimeoutError: socket read timeout
    try:
        from .database_connectors.influxdb import is_connection_cut, is_read_timeout
    except ImportError:  # pragma: no cover
        return False
    return is_read_timeout(error) or is_connection_cut(error)


def _timeout(error: BaseException) -> bool:
    if isinstance(error, (StorageWriteTimeoutError, TimeoutError)):
        return True
    try:
        from .database_connectors.influxdb import is_read_timeout
    except ImportError:  # pragma: no cover
        return False
    return is_read_timeout(error)


class BackgroundWriter:
    """Bounded in-memory queue drained by a private daemon thread (one Pulse)

    - submit() never blocks: when the queue holds more than `max_queue_rows`, the oldest batches are dropped and counted;
      a single batch larger than `max_queue_rows` keeps only its newest rows.
    - The thread writes every `flush_interval_s` seconds, or as soon as `max_batch_rows` rows are waiting, with up to
      `concurrency` requests in flight.
    - After a failed write, the storage is considered unavailable for an exponential delay (1 s, 2 s, 4 s... up to
      `max_retry_delay_s`), probed every `probe_interval_s` when the target can be pinged: batches go straight to the
      spool or, without spool, stay in the queue (within `max_queue_rows`) until the storage is back.
    - Spooled batches are replayed, oldest first, once the storage is available, at most `replay_max_rows_per_s`.
    - Without spool, the rows of a write that timed out or was cut are written once more (`retry_uncertain`), then
      counted in `uncertain_rows` if the outcome is still uncertain.
    - A write that timed out makes the next requests smaller (`adaptive_batch`, see SharedWriter); with a spool, the
      batches beyond `max_queue_rows` are spooled instead of dropped (`spool_overflow`); `bound_in_flight` counts the
      rows in flight against `max_queue_rows` too.
    - `durability="wal"`: each batch is also written to the spool directory before submit() returns, so that a crash
      (kill -9) loses nothing; the file is deleted once the batch is written.

    `write` (a function of the frame and `meta`) or `target` gives the way to write.
    """

    def __init__(
        self,
        write: Optional[WriteFunction],
        meta: dict,
        max_queue_rows: int,
        max_batch_rows: int,
        flush_interval_s: float,
        max_retry_delay_s: float,
        spool: Optional[Spool] = None,
        target: Optional[Target] = None,
        concurrency: int = 1,
        probe_interval_s: Optional[float] = 1.0,
        durability: Literal["memory", "wal"] = "memory",
        replay_max_rows_per_s: Optional[float] = None,
        retry_uncertain: bool = True,
        adaptive_batch: bool = True,
        min_batch_rows: int = 10,
        spool_overflow: bool = True,
        bound_in_flight: bool = False,
    ):
        self._writer = SharedWriter(
            max_batch_rows=max_batch_rows,
            flush_interval_s=flush_interval_s,
            max_retry_delay_s=max_retry_delay_s,
            concurrency=concurrency,
            probe_interval_s=probe_interval_s,
            adaptive_batch=adaptive_batch,
            min_batch_rows=min_batch_rows,
        )
        self._channel = self._writer.register(
            target or Target(write_frame=write),
            meta,
            max_queue_rows=max_queue_rows,
            spool=spool,
            durability=durability,
            replay_max_rows_per_s=replay_max_rows_per_s,
            retry_uncertain=retry_uncertain,
            spool_overflow=spool_overflow,
            bound_in_flight=bound_in_flight,
        )

    def submit(self, frame: pd.DataFrame) -> None:
        self._channel.submit(frame)

    def flush(self, timeout_s: float = 30.0) -> bool:
        return self._channel.flush(timeout_s)

    def close(self, timeout_s: float = 30.0) -> None:
        """Write (or spool) what is pending, then stop the thread"""
        self._writer.close(timeout_s)

    def stats(self) -> dict:
        return self._channel.stats()
