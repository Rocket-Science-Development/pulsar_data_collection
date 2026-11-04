"""Evidence outbox: durable, at-least-once delivery of evidence records to a remote HTTPS service.

Evidence records are aggregated results (e.g. the "pulsar.evidence/1" records of pulsar-metrics) that are sent from
where the data lives to a central service, while the raw data stays local. `EvidenceOutbox` takes them as plain dicts
and only checks that each one carries a `schema_version` and an `evidence_id`: it does not depend on pulsar-metrics.

Delivery:

- `put()` writes the record to a local SQLite file (WAL journal, synchronous=FULL) before returning: once it returns
  True, the record survives a crash or a kill of the process;
- a background thread POSTs the pending records in batches, oldest first, as one JSON object
  `{"records": [...], "sent_at": "<UTC ISO 8601>"}`. A record leaves the store only when the receiver answered 2xx;
- delivery is at least once: a record whose answer was lost (timeout, crash after the send) is sent again. The
  receiver deduplicates on `evidence_id`. A record put again while still pending replaces the pending copy (same id;
  the new version takes a new place at the end of the queue, and the answer to a send of the old version, delivered
  or rejected, never deletes nor quarantines the new one);
- 2xx and 409 (the receiver already holds this `evidence_id`) count as delivered;
- network errors, timeouts, 408, 429 and 5xx are retried for ever (or `max_attempts`) with exponential backoff and
  jitter, honouring `Retry-After`. 401, 403, 404, 405, 410, 421 and redirects are endpoint or configuration errors
  (credentials, URL, method): retried with backoff and logged, never discarded. Any other 4xx may reject the record
  itself ("poison"): the batch is split and the records sent one by one. A record refused on its own is moved to the
  quarantine table (kept for inspection, counted in `failed`, never retried) when another record of the same batch
  was accepted, when it was alone in its batch, or when it is too large (413). When every record of a batch is
  refused one by one, the receiver is taken to refuse the request rather than the records (schema or endpoint
  error): they are retried with backoff;
- the store is bounded (`max_records` pending): beyond it `overflow` decides, "drop_oldest" (default) or
  "drop_newest", and every record lost that way is counted in `dropped` and logged.

The bearer token (or the header callable) is never logged, never shown by `repr()` and redacted from stored errors.
TLS certificates are verified by default (`ca_bundle` for a private CA); every request has a timeout.
"""

import json
import logging
import os
import random
import sqlite3
import ssl
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable, Iterable, Literal, Optional
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger("pulsar_data_collection")

DATABASE_NAME = "evidence_outbox.sqlite3"
RETRYABLE_STATUS = (408, 429)  # with every 5xx
CONFIGURATION_STATUS = (401, 403)  # wrong or expired credentials: retried, never discarded
ENDPOINT_STATUS = (404, 405, 410, 421)  # wrong URL or method, or endpoint gone: retried like credentials errors
DELIVERED_STATUS = (409,)  # conflict: the receiver already holds this evidence_id
MAX_BACKOFF_EXPONENT = 60  # 2 ** (failures - 1) is capped: no float overflow after a long outage
SPLIT_STATUS = (413,)  # payload too large: the batch is split, a single record too large is quarantined
_SCHEMA = """
CREATE TABLE IF NOT EXISTS outbox (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    evidence_id TEXT NOT NULL UNIQUE,
    body TEXT NOT NULL,
    queued_at REAL NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT
);
CREATE TABLE IF NOT EXISTS quarantine (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    evidence_id TEXT NOT NULL,
    body TEXT NOT NULL,
    queued_at REAL NOT NULL,
    quarantined_at REAL NOT NULL,
    status_code INTEGER,
    reason TEXT
);
"""


class EvidenceRejected(Exception):
    """The receiver refused a request for its content (4xx other than 401, 403, 404, 405, 408, 409, 410, 421, 429)"""


class _Secret:
    """Holds the token: never shown by repr() or str()"""

    __slots__ = ("_value",)

    def __init__(self, value: Optional[str]):
        self._value = value

    def get(self) -> Optional[str]:
        return self._value

    def __repr__(self) -> str:
        return "<secret>" if self._value else "<none>"

    __str__ = __repr__


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Redirects are refused: urllib would forward the Authorization header to the new host"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _safe_url(url: str) -> str:
    """URL without user info, query or fragment, for logs and repr"""
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def _retry_after(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
    except (TypeError, ValueError):
        return None


class EvidenceOutbox:
    """Durable outbox of evidence records, sent in batches to an HTTPS endpoint by a background thread

    Parameters
    ----------
    url : str
        Endpoint receiving the POSTed batches; must be https unless allow_http=True (tests, loopback receivers)
    directory : path
        Folder of the SQLite store (created if needed). One outbox per directory and process at a time
    token : str, optional
        Bearer token, sent as "Authorization: Bearer <token>"
    token_env : str, optional
        Name of the environment variable holding the token, read at each request (rotation without restart)
    headers : callable, optional
        Function returning extra request headers (e.g. a signed or short-lived Authorization), called at each request
    max_records : int, optional
        Records kept pending at most (default 100 000); beyond, `overflow` applies
    overflow : "drop_oldest" or "drop_newest"
        drop_oldest (default) deletes the oldest pending record to make room; drop_newest refuses the new one (put
        returns False). Either way the record is counted in `dropped`
    batch_size, max_batch_bytes : int, optional
        Records (default 100) and bytes (default 1 MiB) per request at most
    timeout_s : float, optional
        Timeout of every request (default 10 s)
    verify : bool, optional
        Verify the TLS certificate and host name (default True; False is for tests only and logged as a warning)
    ca_bundle : path, optional
        CA bundle file to verify the receiver's certificate (private or ministry CA)
    retry_initial_s, retry_max_s, jitter : float, optional
        Backoff after the n-th consecutive failure: min(retry_max_s, retry_initial_s * 2^(n-1)), reduced by a random
        share of at most `jitter` (default 1 s, 300 s, 0.5). retry_initial_s > 0, retry_max_s >= retry_initial_s and
        0 <= jitter < 1
    max_attempts : int, optional
        Attempts after which a record is moved to the quarantine (default None: retried until delivered)
    poll_interval_s : float, optional
        Delay before sending a partial batch, to gather records put together (default 0.5 s)
    allow_http : bool, optional
        Accept an http:// URL (default False)
    start : bool, optional
        Start the sending thread now (default True)
    """

    def __init__(
        self,
        url: str,
        directory: os.PathLike,
        token: Optional[str] = None,
        token_env: Optional[str] = None,
        headers: Optional[Callable[[], dict]] = None,
        max_records: int = 100_000,
        overflow: Literal["drop_oldest", "drop_newest"] = "drop_oldest",
        batch_size: int = 100,
        max_batch_bytes: int = 1 << 20,
        timeout_s: float = 10.0,
        verify: bool = True,
        ca_bundle: Optional[os.PathLike] = None,
        retry_initial_s: float = 1.0,
        retry_max_s: float = 300.0,
        jitter: float = 0.5,
        max_attempts: Optional[int] = None,
        poll_interval_s: float = 0.5,
        allow_http: bool = False,
        start: bool = True,
    ):
        scheme = urlsplit(url).scheme.lower()
        if urlsplit(url).username is not None or urlsplit(url).password is not None:
            raise ValueError("credentials in the evidence URL are not supported: pass token=, token_env= or headers=")
        if scheme != "https" and not (allow_http and scheme == "http"):
            raise ValueError(f"evidence URL must be https (got {scheme or 'no'} scheme); allow_http=True for tests only")
        if overflow not in ("drop_oldest", "drop_newest"):
            raise ValueError("overflow must be 'drop_oldest' or 'drop_newest'")
        if max_records < 1 or batch_size < 1 or timeout_s <= 0:
            raise ValueError("max_records and batch_size must be >= 1, timeout_s > 0")
        if not 0 <= jitter < 1:
            raise ValueError("jitter must be >= 0 and < 1")
        if not retry_initial_s > 0 or not retry_max_s >= retry_initial_s:
            raise ValueError("retry_initial_s must be > 0 and retry_max_s >= retry_initial_s")
        self._url = url
        self.url = _safe_url(url)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._token = _Secret(token)
        self._token_env = token_env
        self._headers = headers
        self.max_records = max_records
        self.overflow = overflow
        self.batch_size = batch_size
        self.max_batch_bytes = max_batch_bytes
        self.timeout_s = timeout_s
        self.retry_initial_s = retry_initial_s
        self.retry_max_s = retry_max_s
        self.jitter = jitter
        self.max_attempts = max_attempts
        self.poll_interval_s = poll_interval_s

        handlers = [_NoRedirect()]
        if scheme == "https":
            context = ssl.create_default_context(cafile=str(ca_bundle) if ca_bundle else None)
            if not verify:
                logger.warning("Evidence outbox: TLS verification disabled for %s (tests only)", self.url)
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
            handlers.append(urllib.request.HTTPSHandler(context=context))
        self._opener = urllib.request.build_opener(*handlers)

        self._db = sqlite3.connect(self.directory / DATABASE_NAME, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript(_SCHEMA)
        self._lock = threading.Lock()  # the SQLite connection and the counters
        self._wake = threading.Condition(self._lock)
        self._stats = {"queued": 0, "sent": 0, "retried": 0, "failed": 0, "dropped": 0, "replaced": 0, "requests": 0}
        self._failures = 0  # consecutive failed requests
        self._retry_at = 0.0  # monotonic time before which no request is made
        self._last_error: Optional[str] = None
        self._flush_requested = False
        self._closing = False
        self._closed = False  # the store is closed: the thread exits, start() is refused
        self._unexpected_logged = False  # an unexpected error of the thread is logged once until the next delivery
        self._thread: Optional[threading.Thread] = None
        pending = self._pending_count()
        if pending:
            logger.warning("Evidence outbox: %d records left by a previous run will be sent to %s", pending, self.url)
        if start:
            self.start()

    def __repr__(self) -> str:
        return f"EvidenceOutbox(url={self.url!r}, directory={str(self.directory)!r}, token={self._token!r})"

    # ------------------------------------------------------------------ public API
    def start(self) -> None:
        """Start the sending thread (done by the constructor unless start=False); RuntimeError once closed"""
        if self._closed:
            raise RuntimeError("evidence outbox is closed: open a new one on the same directory")
        if self._thread is None or not self._thread.is_alive():
            self._closing = False
            self._thread = threading.Thread(target=self._run, name="pulsar-evidence-outbox", daemon=True)
            self._thread.start()

    def put(self, record: dict) -> bool:
        """Store one record durably for sending; False when refused because the store is full (drop_newest)"""
        return self.put_many([record]) == 1

    def put_many(self, records: Iterable[dict]) -> int:
        """Store records durably (one transaction); returns how many were accepted

        Raises ValueError, before storing any, when a record is not a dict with non-empty string `schema_version` and
        `evidence_id`, or is not valid JSON (NaN, infinities, objects that do not serialize).
        """
        rows = []
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"an evidence record must be a dict, got {type(record).__name__}")
            for key in ("schema_version", "evidence_id"):
                if not isinstance(record.get(key), str) or not record[key]:
                    raise ValueError(f"evidence record without {key}")
            try:
                body = json.dumps(record, allow_nan=False, separators=(",", ":"), ensure_ascii=False)
            except (TypeError, ValueError) as error:
                raise ValueError(f"evidence record {record['evidence_id']} is not valid JSON: {error}") from None
            rows.append((record["evidence_id"], body))
        accepted = 0
        with self._wake:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                for evidence_id, body in rows:
                    accepted += self._insert(evidence_id, body)
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            self._wake.notify_all()
        return accepted

    def flush(self, timeout_s: float = 30.0) -> bool:
        """Send now what is pending (skipping the backoff once) and wait; True when nothing is left pending"""
        deadline = time.monotonic() + timeout_s
        with self._wake:
            self._flush_requested = True
            self._retry_at = 0.0
            self._wake.notify_all()
        try:
            while time.monotonic() < deadline:
                with self._wake:
                    if self._pending_count() == 0:
                        return True
                    if self._thread is None or not self._thread.is_alive():
                        return False
                    self._wake.wait(min(0.05, max(0.0, deadline - time.monotonic())))
            with self._wake:
                return self._pending_count() == 0
        finally:
            with self._wake:
                self._flush_requested = False  # a flush that timed out must not disable the batching of later records

    def close(self, timeout_s: float = 10.0, flush: bool = True) -> None:
        """Flush (unless flush=False), stop the thread and close the store; what is still pending stays on disk and is
        sent by the next outbox opened on the same directory"""
        if flush and self._thread is not None:
            self.flush(timeout_s)
        with self._wake:
            self._closing = True
            self._wake.notify_all()
        if self._thread is not None:
            self._thread.join(max(timeout_s, self.timeout_s + 1))  # a request in progress ends within timeout_s
            if self._thread.is_alive():
                logger.warning("Evidence outbox: the sending thread is still busy at close; it stops after its request")
        with self._lock:
            if self._closed:
                return
            pending = self._pending_count()
            self._closed = True  # set before the close: a thread still in a request then exits without the store
            self._db.close()
        if pending:
            logger.warning("Evidence outbox closed with %d records pending: they will be sent at the next start", pending)

    def __enter__(self) -> "EvidenceOutbox":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def stats(self) -> dict:
        """Counters since this outbox was opened, and the state of the store

        queued: records accepted by put; sent: delivered (2xx); retried: record sends that failed and will be retried;
        failed: records moved to the quarantine (rejected by the receiver, or max_attempts reached); dropped: records
        lost to the overflow policy; replaced: records put again while pending (same evidence_id); pending and
        quarantined: records in the store now; oldest_pending_age_s: age of the oldest pending record (0 when none).
        """
        with self._lock:
            stats = dict(self._stats)
            stats["pending"] = self._pending_count()
            stats["quarantined"] = self._db.execute("SELECT COUNT(*) FROM quarantine").fetchone()[0]
            oldest = self._db.execute("SELECT MIN(queued_at) FROM outbox").fetchone()[0]
            stats["oldest_pending_age_s"] = max(0.0, time.time() - oldest) if oldest is not None else 0.0
            stats["consecutive_failures"] = self._failures
            stats["receiver_available"] = self._failures == 0
            stats["last_error"] = self._last_error
        return stats

    def quarantined(self) -> list[dict]:
        """Records moved aside, with the status code and the reason (for inspection; they are never sent again)"""
        with self._lock:
            rows = self._db.execute(
                "SELECT evidence_id, body, status_code, reason, quarantined_at FROM quarantine ORDER BY seq"
            ).fetchall()
        return [
            {"evidence_id": i, "record": json.loads(b), "status_code": c, "reason": r, "quarantined_at": t}
            for i, b, c, r, t in rows
        ]

    # ------------------------------------------------------------------ store (called with the lock held)
    def _pending_count(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]

    def _insert(self, evidence_id: str, body: str) -> int:
        old = self._db.execute("SELECT seq, queued_at FROM outbox WHERE evidence_id = ?", (evidence_id,)).fetchone()
        if old:
            # a new row (new seq), not an UPDATE: the answer to a send of the old version, which may be in flight,
            # deletes or quarantines the old seq only, never the new version
            self._db.execute("DELETE FROM outbox WHERE seq = ?", (old[0],))
            self._db.execute("INSERT INTO outbox (evidence_id, body, queued_at) VALUES (?, ?, ?)", (evidence_id, body, old[1]))
            self._stats["replaced"] += 1
            return 1
        if self._pending_count() >= self.max_records:
            self._stats["dropped"] += 1
            if self.overflow == "drop_newest":
                logger.warning("Evidence outbox full (%d records): new record %s dropped", self.max_records, evidence_id)
                return 0
            oldest = self._db.execute("SELECT seq, evidence_id FROM outbox ORDER BY seq LIMIT 1").fetchone()
            self._db.execute("DELETE FROM outbox WHERE seq = ?", (oldest[0],))
            logger.warning("Evidence outbox full (%d records): oldest record %s dropped", self.max_records, oldest[1])
        self._db.execute("INSERT INTO outbox (evidence_id, body, queued_at) VALUES (?, ?, ?)", (evidence_id, body, time.time()))
        self._stats["queued"] += 1
        return 1

    def _batch(self) -> list[tuple]:
        rows = self._db.execute(
            "SELECT seq, evidence_id, body, attempts FROM outbox ORDER BY seq LIMIT ?", (self.batch_size,)
        ).fetchall()
        batch, size = [], 0
        for row in rows:
            size += len(row[2].encode("utf-8")) + 1
            if batch and size > self.max_batch_bytes:
                break
            batch.append(row)
        return batch

    def _delete(self, seqs: list[int]) -> int:
        """Delete delivered rows; returns how many were still pending (not replaced nor dropped meanwhile)"""
        return self._db.executemany("DELETE FROM outbox WHERE seq = ?", [(s,) for s in seqs]).rowcount

    def _quarantine(self, rows: list[tuple], status_code: Optional[int], reason: str) -> None:
        """Move rows to the quarantine; a row no longer pending (replaced or dropped meanwhile) is skipped"""
        now, moved = time.time(), []
        self._db.execute("BEGIN IMMEDIATE")
        try:
            for seq, evidence_id, _, _ in rows:
                row = self._db.execute("SELECT body, queued_at FROM outbox WHERE seq = ?", (seq,)).fetchone()
                if row is None:
                    continue
                self._db.execute(
                    "INSERT INTO quarantine (evidence_id, body, queued_at, quarantined_at, status_code, reason) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (evidence_id, row[0], row[1], now, status_code, reason),
                )
                self._db.execute("DELETE FROM outbox WHERE seq = ?", (seq,))
                moved.append(evidence_id)
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        if not moved:
            return
        self._stats["failed"] += len(moved)
        logger.error("Evidence outbox: %d record(s) moved to the quarantine (%s): %s", len(moved), reason, ", ".join(moved[:5]))

    # ------------------------------------------------------------------ sending
    def _bearer_token(self) -> Optional[str]:
        return self._token.get() or (os.environ.get(self._token_env) if self._token_env else None)

    def _request_headers(self, token: Optional[str]) -> dict:
        headers = {"Content-Type": "application/json", "User-Agent": "pulsar-data-collection evidence outbox"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if self._headers is not None:
            headers.update(self._headers() or {})
        return headers

    def _secrets(self, headers: dict, token: Optional[str] = None) -> list[str]:
        values = [v for k, v in headers.items() if k.lower() in ("authorization", "proxy-authorization", "x-api-key")]
        values += [v.split(" ", 1)[1] for v in values if " " in v]
        tokens = [t for t in (token, self._token.get()) if t]
        return [s for s in values + tokens if isinstance(s, str) and s]

    @staticmethod
    def _redact(text: str, secrets: list[str]) -> str:
        for secret in sorted(secrets, key=len, reverse=True):
            text = text.replace(secret, "***")
        return text

    def _post(self, rows: list[tuple]) -> tuple:
        """(outcome, status, retry_after, error): outcome "sent", "retry", "rejected" or "split" """
        body = '{"records":[' + ",".join(r[2] for r in rows) + '],"sent_at":'
        body += json.dumps(datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")) + "}"
        secrets: list[str] = []
        try:
            token = self._bearer_token()
            secrets = self._secrets({}, token)  # before the header callable: its error message may hold the token
            headers = self._request_headers(token)
            secrets = self._secrets(headers, token)
            request = urllib.request.Request(self._url, data=body.encode("utf-8"), headers=headers, method="POST")
            # the scheme is checked in __init__ (https, or http when allowed): no file:// or custom scheme
            with self._opener.open(request, timeout=self.timeout_s) as response:  # nosec B310
                response.read()
                return "sent", response.status, None, None
        except urllib.error.HTTPError as error:
            status = error.code
            retry_after = _retry_after(error.headers.get("Retry-After") if error.headers else None)
            error.close()
            reason = f"HTTP {status} {self._redact(str(error.reason or ''), secrets)}".strip()
            if status in DELIVERED_STATUS:
                return "sent", status, None, None
            if status in RETRYABLE_STATUS or status >= 500:
                return "retry", status, retry_after, reason
            if status in CONFIGURATION_STATUS or status in ENDPOINT_STATUS or 300 <= status < 400:
                return "retry", status, retry_after, reason + " (configuration: credentials or URL)"
            if status in SPLIT_STATUS and len(rows) > 1:
                return "split", status, None, reason
            return ("split" if len(rows) > 1 else "rejected"), status, None, reason
        except Exception as error:  # noqa: BLE001 - network errors, timeouts, TLS failures, header callable errors
            text = self._redact(f"{type(error).__name__}: {error}", secrets)
            return "retry", None, None, text[:500]

    def _backoff(self, retry_after: Optional[float]) -> float:
        exponent = min(max(self._failures - 1, 0), MAX_BACKOFF_EXPONENT)
        delay = min(self.retry_max_s, self.retry_initial_s * 2.0**exponent)
        # jitter, not cryptography
        delay *= 1 - self.jitter * random.random()  # nosec B311
        if retry_after is not None:
            delay = max(delay, min(retry_after, self.retry_max_s))
        return delay

    def _send(self, rows: list[tuple]) -> None:
        """Send rows; on a content rejection of several rows, send them one by one to find the poison records"""
        outcome, status, retry_after, error = self._post(rows)
        with self._lock:
            self._stats["requests"] += 1
            if outcome == "sent":
                self._delivered(rows)
                return
            self._last_error = error
            if outcome == "rejected":
                self._quarantine(rows, status, error)
                return
            if outcome == "retry":
                self._retry(rows, status, retry_after, error)
                return
        accepted, refused = 0, []  # split: one request per record
        for row in rows:
            if self._closing or time.monotonic() < self._retry_at:
                break  # the records not sent yet stay pending
            outcome, status, retry_after, error = self._post([row])
            with self._lock:
                self._stats["requests"] += 1
                if outcome == "sent":
                    self._delivered([row])
                    accepted += 1
                    continue
                self._last_error = error
                if outcome == "retry":
                    self._retry([row], status, retry_after, error)
                else:
                    refused.append((row, status, error))
        with self._lock:
            # poison: refused on its own while another record of the batch was accepted, or too large on its own
            poison = [r for r in refused if accepted or r[1] in SPLIT_STATUS]
            for row, status, error in poison:
                self._quarantine([row], status, error)
            rest = [r for r in refused if r not in poison]
            if rest:
                # every record refused on its own and none accepted: the request is refused, not the records
                error = f"{rest[-1][2]} (every record of the batch refused: endpoint or schema error?)"
                self._last_error = error
                self._retry([r[0] for r in rest], rest[-1][1], None, error)

    def _delivered(self, rows: list[tuple]) -> None:
        self._stats["sent"] += self._delete([r[0] for r in rows])  # rows replaced or dropped meanwhile are not counted
        if self._failures:
            logger.info("Evidence outbox: %s reachable again", self.url)
        self._failures, self._retry_at, self._last_error = 0, 0.0, None
        self._unexpected_logged = False

    def _retry(self, rows: list[tuple], status: Optional[int], retry_after: Optional[float], error: Optional[str]) -> None:
        self._failures += 1
        delay = self._backoff(retry_after)
        self._retry_at = time.monotonic() + delay
        self._stats["retried"] += len(rows)
        self._db.executemany(
            "UPDATE outbox SET attempts = attempts + 1, last_error = ? WHERE seq = ?", [(error, r[0]) for r in rows]
        )
        log = logger.error if status in CONFIGURATION_STATUS or status in ENDPOINT_STATUS else logger.warning
        log("Evidence outbox: %d records not sent to %s (%s), retry in %.1f s", len(rows), self.url, error, delay)
        if self.max_attempts is not None:
            spent = [r for r in rows if r[3] + 1 >= self.max_attempts]
            if spent:
                self._quarantine(spent, status, f"{self.max_attempts} attempts: {error}")

    def _run(self) -> None:
        """Sending loop: no error ends it, only close()"""
        while True:
            try:
                if not self._step():
                    return
            except Exception:  # noqa: BLE001 - the thread must survive any error; the rows stay pending
                if self._closed:
                    return  # close() stopped waiting for a request in progress and closed the store
                self._on_unexpected_error()

    def _on_unexpected_error(self) -> None:
        if not self._unexpected_logged:
            self._unexpected_logged = True
            logger.exception("Evidence outbox: unexpected error while sending (logged once until the next delivery)")
        else:
            logger.debug("Evidence outbox: unexpected error while sending, again", exc_info=True)
        with self._wake:
            self._failures += 1
            try:
                delay = self._backoff(None)
            except Exception:  # noqa: BLE001 - this path must not fail either
                delay = self.retry_max_s
            self._retry_at = time.monotonic() + delay

    def _step(self) -> bool:
        """Wait for a batch and send it; False when the outbox is closing"""
        with self._wake:
            seen = None  # when pending records were first seen: a partial batch waits poll_interval_s for more
            while not self._closing:
                now = time.monotonic()
                pending = self._pending_count()
                if pending and now >= self._retry_at:
                    seen = now if seen is None else seen
                    if pending >= self.batch_size or self._flush_requested or now >= seen + self.poll_interval_s:
                        break
                    self._wake.wait(seen + self.poll_interval_s - now)
                    continue
                seen = None
                self._wake.wait(min(max(self._retry_at - now, 0.01), 1.0) if pending else 1.0)
            if self._closing:
                return False
            rows = self._batch()
            if not rows:
                self._flush_requested = False
                return True
        self._send(rows)
        with self._wake:
            if self._pending_count() == 0:
                self._flush_requested = False
            self._wake.notify_all()
        return True
