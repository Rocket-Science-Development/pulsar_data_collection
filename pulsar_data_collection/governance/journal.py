"""Append-only, hash-chained audit journal of the data control operations (retention changes, deletions, exports).

One JSON object per line:

    {"seq": 3, "at": "2026-11-04T10:00:00.123456Z", "actor": "jdoe", "action": "delete",
     "details": {...}, "prev": "<hash of entry 2>", "hash": "<hash of this entry>"}

`hash` is SHA-256 (or HMAC-SHA256 with a key, see `key_env`) of the canonical JSON of the entry without `hash`, so that
modifying, inserting, removing or reordering an entry breaks the chain from that entry on. Removing the last entries
or rewriting the whole file can only be detected against a head kept elsewhere (`verify(expected_head=...)`), or with
the key. Entries hold who, what, when, the predicate and the counts: never data values, never tokens (details whose
keys look like secrets are refused).
"""

import contextlib
import hashlib
import hmac
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

GENESIS = "0" * 64
_SECRET_WORDS = ("token", "password", "secret", "authorization", "api_key", "apikey")


def _canonical(entry: dict) -> bytes:
    return json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")


def _check_no_secret(value, path: str = "details") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if any(word in str(key).lower() for word in _SECRET_WORDS):
                raise ValueError(f"{path}.{key}: secrets are never written to the audit journal")
            _check_no_secret(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _check_no_secret(item, f"{path}[{index}]")


@contextlib.contextmanager
def _file_lock(path: Path):
    """Exclusive lock on a side file, across processes (fcntl on POSIX, msvcrt on Windows)"""
    with open(path, "a+b") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            for _ in range(600):  # LK_LOCK retries 10 times per call (about 10 s)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    continue
            else:  # pragma: no cover - another process holds the lock for hours
                raise TimeoutError(f"audit journal lock {path} is held by another process")
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class AuditJournal:
    """Append-only journal file (JSON lines), each entry chained to the previous one

    Parameters
    ----------
    path : path
        Journal file, created with its directory if needed
    key_env : str, optional
        Environment variable holding a key: entries are chained with HMAC-SHA256, so that rewriting the journal needs
        the key. The key is read at each use and never written.
    """

    def __init__(self, path: os.PathLike, key_env: Optional[str] = None):
        self.path = Path(path)
        self.key_env = key_env
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"AuditJournal(path={str(self.path)!r}, keyed={self.key_env is not None})"

    def _key(self) -> Optional[bytes]:
        if self.key_env is None:
            return None
        key = os.environ.get(self.key_env)
        if not key:
            raise RuntimeError(f"audit journal key: environment variable {self.key_env} is not set")
        return key.encode("utf-8")

    def _digest(self, entry: dict) -> str:
        key, data = self._key(), _canonical(entry)
        return hmac.new(key, data, hashlib.sha256).hexdigest() if key else hashlib.sha256(data).hexdigest()

    def _last(self) -> Optional[dict]:
        if not self.path.exists():
            return None
        last = None
        with open(self.path, "rb") as handle:
            for line in handle:
                if line.strip():
                    last = line
        return json.loads(last) if last else None

    def append(self, action: str, actor: str, details: dict) -> dict:
        """Add one entry (written and fsynced before returning); returns it with its hash"""
        _check_no_secret(details)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, _file_lock(self.path.with_name(self.path.name + ".lock")):
            last = self._last()
            entry = {
                "seq": (last["seq"] + 1) if last else 1,
                "at": datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z"),
                "actor": str(actor),
                "action": str(action),
                "details": json.loads(_canonical(details)),
                "prev": last["hash"] if last else GENESIS,
            }
            entry["hash"] = self._digest(entry)
            with open(self.path, "ab") as handle:
                handle.write(_canonical(entry) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
        return entry

    def entries(self) -> list[dict]:
        """Every entry, oldest first (unverified: see verify)"""
        if not self.path.exists():
            return []
        with open(self.path, "rb") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def head(self) -> Optional[str]:
        """Hash of the last entry (keep it elsewhere to detect a truncation later), None when empty"""
        last = self._last()
        return last["hash"] if last else None

    def verify(self, expected_head: Optional[str] = None) -> dict:
        """Check the whole chain: {"ok", "entries", "head", "error", "line"}; `line` is the first broken line (1-based)

        `expected_head`: a head recorded earlier; the journal must still contain it (removed entries are detected).
        """
        result = {"ok": True, "entries": 0, "head": None, "error": None, "line": None}

        def broken(line: int, error: str) -> dict:
            result.update(ok=False, error=error, line=line)
            return result

        if not self.path.exists():
            return broken(0, "journal file not found") if expected_head else result
        previous, seen_head = GENESIS, expected_head is None
        with open(self.path, "rb") as handle:
            lines = [line for line in handle]
        number = 0
        for number, raw in enumerate(lines, start=1):
            if not raw.strip():
                return broken(number, "empty line")
            try:
                entry = json.loads(raw)
                stored = entry.pop("hash")
            except (ValueError, KeyError, AttributeError):
                return broken(number, "not a journal entry")
            if entry.get("seq") != number:
                return broken(number, f"sequence {entry.get('seq')} where {number} was expected (entry removed or moved)")
            if entry.get("prev") != previous:
                return broken(number, "previous hash does not match (entry removed, inserted or modified before it)")
            if not hmac.compare_digest(self._digest(entry), str(stored)):
                return broken(number, "hash does not match the content (entry modified)")
            previous = stored
            seen_head = seen_head or stored == expected_head
            result.update(entries=number, head=stored)
        if not seen_head:
            return broken(number, "expected head not found: entries were removed or the journal was rewritten")
        return result
