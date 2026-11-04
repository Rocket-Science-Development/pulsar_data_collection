"""EvidenceOutbox: durable at-least-once delivery to a local HTTP receiver (http.server on the loopback)"""

import json
import logging
import ssl
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from pulsar_data_collection.evidence import DATABASE_NAME, EvidenceOutbox

FIXTURE = Path(__file__).parent / "data" / "evidence_records.json"  # records in the "pulsar.evidence/1" format
TOKEN = "tok-3f9a1c7e-synthetic-secret"  # synthetic test token


class Receiver:
    """Loopback receiver: answers each POST with the next scripted status (default 200), records what it got"""

    def __init__(self, port: int = 0):
        self.batches: list[list[dict]] = []
        self.headers: list[dict] = []
        self.script: list = []  # statuses (or (status, headers)) to answer, in order, before falling back to 200
        self.reject_ids: set = set()  # evidence_id answered 400 (poison) whenever present in a batch
        self.lock = threading.Lock()
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with receiver.lock:
                    status, extra = 200, {}
                    if receiver.script:
                        item = receiver.script.pop(0)
                        status, extra = item if isinstance(item, tuple) else (item, {})
                    elif receiver.reject_ids & {r["evidence_id"] for r in body["records"]}:
                        status = 400
                    if status == 200:
                        receiver.batches.append(body["records"])
                    receiver.headers.append(dict(self.headers))
                self.send_response(status)
                for key, value in extra.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}/evidence"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def ids(self) -> list[str]:
        with self.lock:
            return [r["evidence_id"] for batch in self.batches for r in batch]

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def receiver():
    r = Receiver()
    yield r
    r.stop()


def records(n: int, prefix: str = "ev") -> list[dict]:
    return [{"schema_version": "pulsar.evidence/1", "evidence_id": f"{prefix}-{i:04d}", "value": i} for i in range(n)]


def outbox(url, directory, **kwargs) -> EvidenceOutbox:
    options = dict(allow_http=True, retry_initial_s=0.05, retry_max_s=0.2, poll_interval_s=0.02, timeout_s=2.0)
    return EvidenceOutbox(url, directory, **{**options, **kwargs})


def wait_for(condition, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


def test_records_are_delivered_in_batches_with_the_bearer_token(receiver, tmp_path):
    with outbox(receiver.url, tmp_path, token=TOKEN, batch_size=10) as box:
        assert box.put_many(records(25)) == 25
        assert box.flush(10)
        stats = box.stats()
    assert receiver.ids() == [f"ev-{i:04d}" for i in range(25)]  # oldest first, each once
    assert max(len(b) for b in receiver.batches) == 10
    assert receiver.headers[0]["Authorization"] == f"Bearer {TOKEN}"
    assert receiver.headers[0]["Content-Type"] == "application/json"
    assert stats["queued"] == stats["sent"] == 25
    assert (stats["pending"], stats["failed"], stats["dropped"], stats["oldest_pending_age_s"]) == (0, 0, 0, 0.0)


def test_receiver_down_then_up_loses_nothing(tmp_path):
    down = Receiver()
    port, url = down.port, down.url
    down.stop()  # nothing listens on the port: connection refused
    box = outbox(url, tmp_path, batch_size=5)
    box.put_many(records(12))
    assert wait_for(lambda: box.stats()["retried"] > 0)
    stats = box.stats()
    assert stats["pending"] == 12 and stats["sent"] == 0 and not stats["receiver_available"]
    assert stats["oldest_pending_age_s"] > 0 and stats["last_error"]
    up = Receiver(port)
    try:
        assert box.flush(10)
        assert sorted(up.ids()) == [f"ev-{i:04d}" for i in range(12)]
        assert box.stats()["receiver_available"]
    finally:
        box.close()
        up.stop()


def test_records_written_before_a_crash_are_sent_after_restart(receiver, tmp_path):
    first = outbox(receiver.url, tmp_path, start=False)  # crash between the write and the send
    first.put_many(records(7))
    first.close(flush=False)
    assert receiver.ids() == []
    assert (tmp_path / DATABASE_NAME).exists()

    with outbox(receiver.url, tmp_path) as second:
        assert second.stats()["pending"] == 7
        assert second.flush(10)
    assert sorted(receiver.ids()) == [f"ev-{i:04d}" for i in range(7)]


def test_lost_answer_resends_the_same_ids_only(tmp_path):
    """At least once: a batch whose answer was lost is sent again; duplicates are only ever the same evidence_id"""

    receiver = Receiver()
    handler = receiver.server.RequestHandlerClass
    original = handler.do_POST
    calls = {"n": 0}

    def slow_first(self):
        calls["n"] += 1
        if calls["n"] == 1:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with receiver.lock:
                receiver.batches.append(body["records"])  # stored by the receiver...
            time.sleep(1.0)  # ...but the answer comes after the client timeout
            return
        original(self)

    handler.do_POST = slow_first
    try:
        with outbox(receiver.url, tmp_path, timeout_s=0.3, batch_size=50) as box:
            box.put_many(records(5))
            box.put(records(1)[0])  # put again while pending: replaced, not queued twice
            assert box.flush(10)
            stats = box.stats()
        ids = receiver.ids()
        assert sorted(set(ids)) == [f"ev-{i:04d}" for i in range(5)]
        assert len(ids) == 10  # every record twice, never another id
        assert stats["replaced"] == 1 and stats["queued"] == 5 and stats["retried"] == 5
    finally:
        receiver.stop()


@pytest.mark.parametrize("status", [429, 503, 408, 401])
def test_transient_and_configuration_errors_are_retried(receiver, tmp_path, status):
    receiver.script = [(status, {"Retry-After": "0"}), status]
    with outbox(receiver.url, tmp_path) as box:
        box.put_many(records(3))
        assert wait_for(lambda: len(receiver.ids()) == 3)
        stats = box.stats()
    assert stats["retried"] == 6 and stats["sent"] == 3 and stats["failed"] == 0


def test_poison_record_is_quarantined_and_the_others_delivered(receiver, tmp_path):
    receiver.reject_ids = {"ev-0002"}
    with outbox(receiver.url, tmp_path, batch_size=10) as box:
        box.put_many(records(5))
        assert box.flush(10)
        stats = box.stats()
        quarantined = box.quarantined()
    assert sorted(receiver.ids()) == ["ev-0000", "ev-0001", "ev-0003", "ev-0004"]
    assert stats["failed"] == 1 and stats["quarantined"] == 1 and stats["pending"] == 0
    assert quarantined[0]["evidence_id"] == "ev-0002" and quarantined[0]["status_code"] == 400
    assert quarantined[0]["record"]["value"] == 2


def test_max_attempts_moves_records_aside(receiver, tmp_path):
    receiver.script = [503] * 10
    with outbox(receiver.url, tmp_path, max_attempts=2) as box:
        box.put(records(1)[0])
        assert wait_for(lambda: box.stats()["failed"] == 1)
        assert box.stats()["pending"] == 0


@pytest.mark.parametrize(
    "overflow, kept", [("drop_oldest", ["ev-0002", "ev-0003", "ev-0004"]), ("drop_newest", ["ev-0000", "ev-0001", "ev-0002"])]
)
def test_bounded_store_counts_what_overflows(receiver, tmp_path, overflow, kept):
    box = outbox(receiver.url, tmp_path, max_records=3, overflow=overflow, start=False)
    accepted = box.put_many(records(5))
    assert accepted == (5 if overflow == "drop_oldest" else 3)
    assert box.stats()["dropped"] == 2 and box.stats()["pending"] == 3
    box.start()
    assert box.flush(10)
    box.close()
    assert receiver.ids() == kept


def test_invalid_records_are_refused_before_storing(tmp_path):
    box = outbox("http://127.0.0.1:9/evidence", tmp_path, start=False)
    with pytest.raises(ValueError, match="evidence_id"):
        box.put_many([records(1)[0], {"schema_version": "pulsar.evidence/1"}])
    with pytest.raises(ValueError, match="schema_version"):
        box.put({"evidence_id": "x"})
    with pytest.raises(ValueError, match="valid JSON"):
        box.put({"schema_version": "pulsar.evidence/1", "evidence_id": "x", "value": float("nan")})
    assert box.stats()["pending"] == 0
    box.close(flush=False)


def test_https_is_required_and_tls_verified_by_default(tmp_path):
    with pytest.raises(ValueError, match="https"):
        EvidenceOutbox("http://collector.example/evidence", tmp_path)
    box = EvidenceOutbox("https://collector.example/evidence", tmp_path, start=False)
    https = [h for h in box._opener.handlers if hasattr(h, "_context")][0]
    assert https._context.verify_mode == ssl.CERT_REQUIRED and https._context.check_hostname
    box.close(flush=False)


def test_tls_failure_is_retried_not_lost(receiver, tmp_path):
    """An https URL on a plain-HTTP receiver fails the TLS handshake: the record stays pending"""
    url = receiver.url.replace("http://", "https://")
    box = EvidenceOutbox(url, tmp_path, retry_initial_s=0.05, retry_max_s=0.1, poll_interval_s=0.01, timeout_s=2)
    box.put(records(1)[0])
    assert wait_for(lambda: box.stats()["retried"] > 0)
    assert box.stats()["pending"] == 1
    box.close(flush=False)


def test_token_never_logged_nor_shown(tmp_path, caplog, monkeypatch):
    caplog.set_level(logging.DEBUG, logger="pulsar_data_collection")
    receiver = Receiver()
    receiver.script = [401, (503, {"Retry-After": "0"})]
    monkeypatch.setenv("PULSAR_TEST_EVIDENCE_TOKEN", TOKEN)
    try:
        with pytest.raises(ValueError) as refused:
            outbox(f"http://user:{TOKEN}@127.0.0.1:{receiver.port}/evidence", tmp_path / "userinfo")
        assert TOKEN not in str(refused.value)
        url = f"http://127.0.0.1:{receiver.port}/evidence?key={TOKEN}"
        box = outbox(url, tmp_path, token_env="PULSAR_TEST_EVIDENCE_TOKEN")
        assert TOKEN not in repr(box) and TOKEN not in box.url
        box.put_many(records(2))
        assert box.flush(10)
        assert receiver.headers[-1]["Authorization"] == f"Bearer {TOKEN}"
        stats = box.stats()
        box.close()
        down = outbox("http://127.0.0.1:9/evidence", tmp_path / "down", token=TOKEN, headers=lambda: {"X-Api-Key": TOKEN})
        down.put(records(1)[0])
        assert wait_for(lambda: down.stats()["retried"] > 0)
        stats_down = down.stats()
        assert TOKEN not in repr(down)
        down.close(flush=False)
    finally:
        receiver.stop()
    assert stats["retried"] == 4
    assert caplog.records, "the failures are logged"
    assert TOKEN not in caplog.text
    assert TOKEN not in json.dumps(stats) and TOKEN not in json.dumps(stats_down)


def test_header_callable_is_called_per_request(receiver, tmp_path):
    calls = []

    def headers():
        calls.append(1)
        return {"Authorization": f"Bearer rotated-{len(calls)}"}

    with outbox(receiver.url, tmp_path, headers=headers, batch_size=1) as box:
        box.put_many(records(2))
        assert box.flush(10)
    assert [h["Authorization"] for h in receiver.headers] == ["Bearer rotated-1", "Bearer rotated-2"]


def test_evidence_records_of_pulsar_metrics_shape_are_delivered_unchanged(receiver, tmp_path):
    """Integration: drift, performance, genai and compliance records in the "pulsar.evidence/1" format (fixture copied
    from pulsar-metrics output), sent twice as a ministry would after a re-run: the receiver dedups on evidence_id"""
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert {r["metric"]["family"] for r in fixture} >= {"drift", "genai", "compliance"}
    with outbox(receiver.url, tmp_path, token=TOKEN) as box:
        assert box.put_many(fixture) == len(fixture)
        assert box.flush(10)
        assert box.put_many(fixture) == len(fixture)  # the next run sends the same findings again
        assert box.flush(10)
    received = [r for batch in receiver.batches for r in batch]
    assert len(received) == 2 * len(fixture)
    by_id = {}
    for record in received:
        by_id.setdefault(record["evidence_id"], []).append(record)
    assert set(by_id) == {r["evidence_id"] for r in fixture}
    for original in fixture:
        assert by_id[original["evidence_id"]] == [original, original]  # byte-for-byte content, deduplicable on id
