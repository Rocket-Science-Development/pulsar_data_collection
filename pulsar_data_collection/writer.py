"""Background writer: keeps writes to the storage engine out of the prediction path."""

import logging
import threading
import time
from collections import deque
from typing import Callable, Optional

import pandas as pd

from .spool import Spool

logger = logging.getLogger("pulsar_data_collection")

WriteFunction = Callable[[pd.DataFrame, dict], None]


class BackgroundWriter:
    """Bounded in-memory queue drained by a daemon thread

    - submit() never blocks: when the queue holds more than `max_queue_rows`, the oldest batches are dropped and counted.
    - The thread writes every `flush_interval_s` seconds, or as soon as `max_batch_rows` rows are waiting.
    - After a failed write, the storage is considered unavailable for an exponential delay (1 s, 2 s, 4 s... up to
      `max_retry_delay_s`): batches go straight to the spool (or are dropped and counted without spool) instead of
      waiting for the storage.
    - Spooled batches are replayed, oldest first, once a write succeeds again.
    """

    def __init__(
        self,
        write: WriteFunction,
        meta: dict,
        max_queue_rows: int,
        max_batch_rows: int,
        flush_interval_s: float,
        max_retry_delay_s: float,
        spool: Optional[Spool] = None,
    ):
        self._write = write
        self._meta = meta
        self._max_queue_rows = max_queue_rows
        self._max_batch_rows = max_batch_rows
        self._flush_interval_s = flush_interval_s
        self._max_retry_delay_s = max_retry_delay_s
        self._spool = spool

        self._queue: deque = deque()
        self._queued_rows = 0
        self._busy = False
        self._stopping = False
        self._condition = threading.Condition()
        self._failures = 0
        self._retry_at = 0.0
        self._stats = {
            "captured_rows": 0,
            "written_rows": 0,
            "spooled_rows": 0,
            "replayed_rows": 0,
            "dropped_rows": 0,
            "write_errors": 0,
            "last_error": None,
        }
        self._thread = threading.Thread(target=self._run, name="pulsar-data-collection-writer", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ caller side
    def submit(self, frame: pd.DataFrame) -> None:
        rows = len(frame)
        with self._condition:
            self._stats["captured_rows"] += rows
            self._queue.append(frame)
            self._queued_rows += rows
            while self._queued_rows > self._max_queue_rows and len(self._queue) > 1:
                dropped = self._queue.popleft()
                self._queued_rows -= len(dropped)
                self._stats["dropped_rows"] += len(dropped)
                logger.warning("Capture queue full (%d rows): %d oldest rows dropped", self._max_queue_rows, len(dropped))
            if self._queued_rows >= self._max_batch_rows:
                self._condition.notify()

    def flush(self, timeout_s: float = 30.0) -> bool:
        """Wait until every submitted row has been written, spooled or dropped; True if done within the timeout"""
        deadline = time.monotonic() + timeout_s
        with self._condition:
            self._condition.notify()
            while self._queue or self._busy:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(min(remaining, 0.05))
                self._condition.notify()
        return True

    def close(self, timeout_s: float = 30.0) -> None:
        """Write (or spool) what is pending, then stop the thread"""
        with self._condition:
            self._stopping = True
            self._condition.notify()
        self._thread.join(timeout_s)
        with self._condition:
            for frame in self._queue:  # thread did not finish in time
                self._fallback(frame, "writer stopped before writing")
            self._queue.clear()
            self._queued_rows = 0

    def stats(self) -> dict:
        with self._condition:
            stats = dict(self._stats)
            stats["queued_rows"] = self._queued_rows
            stats["storage_available"] = time.monotonic() >= self._retry_at
            stats["spool_files"] = len(self._spool.files()) if self._spool else 0
            return stats

    # ------------------------------------------------------------------ writer thread
    def _run(self) -> None:
        while True:
            with self._condition:
                if not self._stopping and self._queued_rows < self._max_batch_rows:
                    self._condition.wait(self._flush_interval_s)
                batch = []
                rows = 0
                while self._queue and rows < self._max_batch_rows:
                    frame = self._queue.popleft()
                    batch.append(frame)
                    rows += len(frame)
                self._queued_rows -= rows
                self._busy = bool(batch)
                stopping = self._stopping and not self._queue
            try:
                if batch:
                    self._write_or_fallback(pd.concat(batch, ignore_index=True))
                self._replay_spool()
            except Exception:  # noqa: BLE001 - the writer thread must never die
                logger.exception("Unexpected error in the capture writer")
            finally:
                with self._condition:
                    self._busy = False
                    self._condition.notify_all()
            if stopping:
                return

    def _write_or_fallback(self, frame: pd.DataFrame) -> None:
        if time.monotonic() < self._retry_at:
            self._fallback(frame, "storage unavailable")
            return
        try:
            self._write(frame, self._meta)
        except Exception as error:  # noqa: BLE001
            self._record_failure(error)
            self._fallback(frame, str(error))
            return
        self._record_success(len(frame), "written_rows")

    def _replay_spool(self) -> None:
        if not self._spool or time.monotonic() < self._retry_at:
            return
        for _ in range(10):  # bounded, so that live batches are not delayed for long
            path = self._spool.oldest()
            if path is None:
                return
            frame, meta = self._spool.load(path)
            try:
                self._write(frame, meta)
            except Exception as error:  # noqa: BLE001
                self._record_failure(error)
                return
            path.unlink()
            self._record_success(len(frame), "replayed_rows")
            logger.info("Replayed %d spooled rows from %s", len(frame), path.name)

    def _fallback(self, frame: pd.DataFrame, reason: str) -> None:
        with self._condition:
            if self._spool:
                self._spool.save(frame, self._meta)
                self._stats["spooled_rows"] += len(frame)
            else:
                self._stats["dropped_rows"] += len(frame)
        if not self._spool:
            logger.warning("%d rows dropped (%s); set spool_dir to keep them", len(frame), reason)

    def _record_failure(self, error: Exception) -> None:
        with self._condition:
            self._failures += 1
            delay = min(2 ** (self._failures - 1), self._max_retry_delay_s)
            self._retry_at = time.monotonic() + delay
            self._stats["write_errors"] += 1
            self._stats["last_error"] = f"{type(error).__name__}: {error}"
        logger.warning("Write to storage failed (%s); next attempt in %.0f s", error, delay)

    def _record_success(self, rows: int, counter: str) -> None:
        with self._condition:
            if self._failures:
                logger.info("Storage available again")
            self._failures = 0
            self._retry_at = 0.0
            self._stats[counter] += rows
