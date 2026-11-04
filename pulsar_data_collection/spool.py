"""Local spool: batches that could not be written are saved to disk and replayed when the storage is back.

File names: `<time_ns>_<id>_<rows>r` followed by a suffix (`time_ns` strictly increasing within a process, so that the
names sort in the order of the batches even when the clock does not move between two of them):

- `.json` / `.json.gz`: spooled batch, replayed oldest first then deleted;
- `.wal` / `.wal.gz`: write-ahead copy of a batch still in the in-memory queue (`durability="wal"`), deleted once written.
  After a crash (kill -9), the next start turns them into spooled batches, so that they are replayed;
- `.tmp`: file being written. A `.tmp` left by a crash is recovered at startup if it is complete, deleted otherwise.

A spool directory belongs to one Pulse of one process at a time.
"""

import gzip
import json
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

logger = logging.getLogger("pulsar_data_collection")

SPOOL_SUFFIXES = (".json", ".json.gz")
WAL_SUFFIXES = (".wal", ".wal.gz")
_ROWS = re.compile(r"_(\d+)r\.")


def _datetime_unit(dtype: str) -> str:
    """Resolution of a datetime dtype string: "datetime64[us, UTC]" -> "us" (pandas 3 defaults to us, pandas 2 to ns)"""
    return dtype[dtype.index("[") + 1 :].split(",")[0].rstrip("]").strip()


def frame_to_json(frame: pd.DataFrame, meta: dict) -> str:
    """Serialize a batch and its write parameters; dates are stored as int64 nanoseconds (exact round trip)"""
    dtypes = {str(c): str(t) for c, t in frame.dtypes.items()}
    out = frame.reset_index(drop=True).copy()
    for column, dtype in dtypes.items():
        if dtype.startswith("datetime64"):
            out[column] = out[column].dt.as_unit("ns").astype("int64")
    payload = {"meta": meta, "dtypes": dtypes, "columns": list(dtypes), "data": json.loads(out.to_json(orient="values"))}
    return json.dumps(payload, separators=(",", ":"))


def frame_from_json(text: str) -> tuple[pd.DataFrame, dict]:
    """Inverse of frame_to_json"""
    payload = json.loads(text)
    frame = pd.DataFrame(payload["data"], columns=payload["columns"])
    for column, dtype in payload["dtypes"].items():
        if dtype.startswith("datetime64"):
            values = pd.to_datetime(frame[column].astype("int64"), unit="ns", utc=True)
            timezone = dtype[dtype.index(",") + 1 : -1].strip() if "," in dtype else None
            values = values.dt.tz_convert(timezone) if timezone else values.dt.tz_localize(None)
            frame[column] = values.dt.as_unit(_datetime_unit(dtype))
        else:
            frame[column] = frame[column].astype(dtype)
    return frame, payload["meta"]


def _suffix(path: Path) -> str:
    name = path.name
    for suffix in (".json.gz", ".wal.gz", ".json", ".wal", ".tmp"):
        if name.endswith(suffix):
            return suffix
    return path.suffix


class Spool:
    """Directory of pending batches, replayed oldest first

    Parameters
    ----------
    directory : path
        Created if needed
    max_bytes : int, optional
        Size limit of the directory. When a new batch does not fit, `eviction` decides: "drop_oldest" (default) deletes
        the oldest spooled batches, "drop_newest" refuses the new one. Either way the rows are counted as evicted.
    compression : "gzip" or None
        gzip (level 1) divides the size by about 2 on random floats, more on repetitive values
    """

    def __init__(
        self,
        directory: os.PathLike,
        max_bytes: Optional[int] = None,
        eviction: Literal["drop_oldest", "drop_newest"] = "drop_oldest",
        compression: Optional[Literal["gzip"]] = None,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.eviction = eviction
        self.compression = compression
        self.evicted_rows = 0
        self.evicted_files = 0
        self.recovered_files = 0
        self._lock = threading.Lock()
        self._held: set = set()  # files being replayed: never evicted
        self._last_stamp = 0
        self._recover()
        self._bytes = sum(self._size(p) for p in self.files())

    # ------------------------------------------------------------------ startup recovery
    def _recover(self) -> None:
        """Leftovers of a crash: complete `.tmp` files and write-ahead copies become spooled batches"""
        for path in sorted(self.directory.glob("*.tmp")):
            try:
                text = path.read_bytes()
                frame_from_json(gzip.decompress(text).decode("utf-8") if text[:2] == b"\x1f\x8b" else text.decode("utf-8"))
            except Exception:  # noqa: BLE001 - incomplete file: its rows were never acknowledged as spooled
                logger.warning("Spool: incomplete file %s from an interrupted write deleted", path.name)
                path.unlink(missing_ok=True)
                continue
            target = path.with_name(path.name[: -len(".tmp")] + (".json.gz" if text[:2] == b"\x1f\x8b" else ".json"))
            path.replace(target)
            self.recovered_files += 1
            logger.warning("Spool: file %s from an interrupted write recovered", target.name)
        for path in self._glob(WAL_SUFFIXES):
            self.promote(path)
            self.recovered_files += 1
        if self.recovered_files:
            logger.warning("Spool: %d batches left by a previous run will be replayed", self.recovered_files)

    # ------------------------------------------------------------------ files
    def _glob(self, suffixes) -> list[Path]:
        return sorted(p for suffix in suffixes for p in self.directory.glob(f"*{suffix}"))

    @staticmethod
    def _size(path: Path) -> int:
        try:
            return path.stat().st_size
        except OSError:
            return 0

    def files(self) -> list[Path]:
        """Spooled batches (write-ahead copies excluded), oldest first"""
        return self._glob(SPOOL_SUFFIXES)

    def oldest(self) -> Optional[Path]:
        files = self.files()
        return files[0] if files else None

    def size_bytes(self) -> int:
        with self._lock:
            return self._bytes

    @staticmethod
    def rows(path: Path) -> Optional[int]:
        """Rows of a batch, from its name (None for files written by older versions)"""
        match = _ROWS.search(path.name)
        return int(match.group(1)) if match else None

    def load(self, path: Path) -> tuple[pd.DataFrame, dict]:
        data = path.read_bytes()
        if path.name.endswith(".gz"):
            data = gzip.decompress(data)
        return frame_from_json(data.decode("utf-8"))

    # ------------------------------------------------------------------ writing
    def save(self, frame: pd.DataFrame, meta: dict, wal: bool = False) -> Optional[Path]:
        """Save a batch (atomic: a crash never leaves a half-written batch under its final name)

        Returns None when the batch was refused because the spool is full. A spooled batch refused that way is counted in
        `evicted_rows`; a write-ahead copy (`wal=True`) is never saved at the expense of spooled batches and its refusal
        is not counted: the batch stays in memory only, the caller keeps track of it.
        """
        data = frame_to_json(frame, meta).encode("utf-8")
        if self.compression == "gzip":
            data = gzip.compress(data, compresslevel=1)
        suffix = (WAL_SUFFIXES if wal else SPOOL_SUFFIXES)[1 if self.compression == "gzip" else 0]
        if not self._make_room(len(data), len(frame), wal=wal):
            return None
        path = self.directory / f"{self._stamp()}_{uuid.uuid4().hex[:8]}_{len(frame)}r{suffix}"
        temporary = path.with_name(path.name[: -len(suffix)] + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
        return path

    def _stamp(self) -> int:
        """time_ns, made strictly increasing: the clock of Windows can stay on the same value for milliseconds, and two
        batches with the same time were ordered by their random id (the oldest could be replayed after, or evicted
        instead of, a newer one)"""
        with self._lock:
            self._last_stamp = max(time.time_ns(), self._last_stamp + 1)
            return self._last_stamp

    def _refuse(self, rows: int) -> bool:
        self.evicted_rows += rows
        self.evicted_files += 1
        logger.warning("Spool full (%d bytes): new batch of %d rows dropped", self.max_bytes, rows)
        return False

    def _make_room(self, size: int, rows: int, wal: bool = False) -> bool:
        with self._lock:
            if self.max_bytes is None or self._bytes + size <= self.max_bytes:
                self._bytes += size
                return True
            if wal:  # a write-ahead copy never evicts spooled batches
                return False
            if self.eviction == "drop_newest" or size > self.max_bytes:
                return self._refuse(rows)
            evictable = [p for p in self.files() if p not in self._held]
            needed = self._bytes + size - self.max_bytes
            if sum(self._size(p) for p in evictable) < needed:
                return self._refuse(rows)
            for path in evictable:
                if self._bytes + size <= self.max_bytes:
                    break
                evicted = self.rows(path)
                if evicted is None:
                    try:
                        evicted = len(self.load(path)[0])
                    except Exception:  # noqa: BLE001
                        evicted = 0
                self._bytes -= self._size(path)
                path.unlink(missing_ok=True)
                self.evicted_rows += evicted
                self.evicted_files += 1
                logger.warning("Spool full (%d bytes): oldest batch of %d rows dropped", self.max_bytes, evicted)
            self._bytes += size
            return True

    def promote(self, path: Path) -> Path:
        """Turn a write-ahead copy into a spooled batch (replayed later)"""
        suffix = _suffix(path)
        target = path.with_name(path.name[: -len(suffix)] + (".json.gz" if suffix.endswith(".gz") else ".json"))
        path.replace(target)
        return target

    def remove(self, path: Path) -> None:
        with self._lock:
            self._bytes -= self._size(path)
            path.unlink(missing_ok=True)
            self._held.discard(path)

    def hold(self, path: Path) -> bool:
        """Protect a file from eviction while it is replayed; False if it no longer exists"""
        with self._lock:
            if not path.exists():
                return False
            self._held.add(path)
            return True

    def release(self, paths) -> None:
        with self._lock:
            for path in paths:
                self._held.discard(path)

    def set_aside(self, path: Path) -> Path:
        """Rename an unusable batch `.bad`: kept for inspection, never replayed, no longer counted in the spool size"""
        target = path.with_name(path.name + ".bad")
        with self._lock:
            self._bytes -= self._size(path)
            self._held.discard(path)
            try:
                path.replace(target)
            except OSError as error:
                logger.warning("Spool: %s cannot be set aside: %s", path.name, error)
        return target
