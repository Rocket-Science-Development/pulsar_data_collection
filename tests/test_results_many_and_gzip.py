"""ResultsSink.write_results_many (several timestamps in one request) and gzip of the write requests, against a local
fake of the InfluxDB 2 HTTP API (no database needed)"""

import gzip
import json
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd
import pytest

from pulsar_data_collection.database_connectors.influxdb import InfluxdbActions, shared_clients
from pulsar_data_collection.models import CaptureOptions, PulseParameters
from pulsar_data_collection.pulse import Pulse
from pulsar_data_collection.results import ResultsSink

T = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
DIGEST = "sha256:" + "ab" * 32


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def _answer(self, code, body=b"", content_type="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/ping"):
            self._answer(204)
        elif self.path.startswith("/api/v2/buckets"):
            body = {"buckets": [{"id": "0000000000000001", "orgID": "0000000000000002", "name": "b", "retentionRules": []}]}
            self._answer(200, json.dumps(body).encode())
        else:
            self._answer(404, b"{}")

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        encoding = self.headers.get("Content-Encoding")
        body = gzip.decompress(raw) if encoding == "gzip" else raw
        self.server.writes.append(
            {"path": self.path, "encoding": encoding, "raw_bytes": len(raw), "lines": body.decode("utf-8").splitlines()}
        )
        self._answer(204)


@pytest.fixture
def influx():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.writes = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def url(server):
    return f"http://127.0.0.1:{server.server_address[1]}"


def results(n, metric="ks"):
    return [
        {
            "metric_type": "drift",
            "metric_name": metric,
            "feature_name": f"f{i}",
            "metric_value": 0.1 * i,
            "drift_status": i % 2 == 0,
        }
        for i in range(n)
    ]


def sink(server, **kwargs):
    return ResultsSink(url(server), "token", "org", "bucket", **kwargs)


def writes(server):
    return [w for w in server.writes if w["path"].startswith("/api/v2/write")]


# ------------------------------------------------------------------------------------------------ write_results_many
def test_many_timestamps_in_one_request(influx):
    items = [
        (results(3), T, {"model_id": "a"}),
        (results(2), T + timedelta(minutes=1), {"model_id": "b"}),
        (results(1, "psi"), T + timedelta(minutes=2)),
    ]
    with sink(influx) as s:
        assert s.write_results_many(items) == 6
    (request,) = writes(influx)
    assert len(request["lines"]) == 6
    stamps = sorted({line.rsplit(" ", 1)[1] for line in request["lines"]})
    assert len(stamps) == 3
    assert sum("model_id=a" in line for line in request["lines"]) == 3
    assert sum("model_id=b" in line for line in request["lines"]) == 2


def test_same_points_as_one_write_results_per_batch(influx):
    items = [(results(4), T + timedelta(minutes=k), {"model_id": f"m{k}", "campaign": "c"}) for k in range(5)]
    with sink(influx) as s:
        for r, ts, tags in items:
            s.write_results(r, ts, tags)
        one_by_one = sorted(line for w in writes(influx) for line in w["lines"])
        influx.writes.clear()
        s.write_results_many(items)
    assert len(writes(influx)) == 1
    assert sorted(writes(influx)[0]["lines"]) == one_by_one


def test_requests_are_bounded_and_large_batches_split(influx):
    items = [(results(7), T), (results(3), T + timedelta(minutes=1))]
    with sink(influx, max_points_per_request=4) as s:
        assert s.write_results_many(items) == 10
    assert [len(w["lines"]) for w in writes(influx)] == [4, 4, 2]
    influx.writes.clear()
    with sink(influx) as s:
        assert s.write_results_many(items, max_points_per_request=6) == 10
    assert [len(w["lines"]) for w in writes(influx)] == [6, 4]


def test_empty_and_malformed_items(influx):
    with sink(influx) as s:
        assert s.write_results_many([]) == 0
        assert s.write_results_many([([], T)]) == 0
        with pytest.raises(ValueError):
            s.write_results_many([results(1)])  # not a (results, timestamp[, tags]) tuple
        with pytest.raises(ValueError):
            s.write_results_many([(results(1), T)], max_points_per_request=0)
    assert writes(influx) == []
    with pytest.raises(ValueError):
        sink(influx, max_points_per_request=0)


def test_model_digest_checked_on_every_batch_before_any_request(influx):
    with sink(influx, model_fingerprint=DIGEST, max_points_per_request=2) as s:
        bad = [(results(3), T), (results(1), T + timedelta(minutes=1), {"model_digest": "sha256:" + "cd" * 32})]
        with pytest.raises(ValueError, match="model_digest"):
            s.write_results_many(bad)
        assert writes(influx) == []  # nothing written, not even the first requests
        assert s.write_results_many([(results(3), T), (results(1), T, {"model_digest": DIGEST, "model_id": "a"})]) == 4
    lines = [line for w in writes(influx) for line in w["lines"]]
    assert len(lines) == 4 and all(f"model_digest={DIGEST}" in line for line in lines)


def test_timestamp_none_means_now(influx):
    with sink(influx) as s:
        s.write_results_many([(results(2), None)])
    stamp = int(writes(influx)[0]["lines"][0].rsplit(" ", 1)[1])
    assert abs(stamp / 1e9 - datetime.now(timezone.utc).timestamp()) < 60


# ------------------------------------------------------------------------------------------------ gzip
def test_sink_write_requests_are_gzipped_only_when_asked(influx):
    with sink(influx) as s:
        s.write_results(results(50), T)
    with sink(influx, write_compression="gzip") as s:
        s.write_results(results(50), T)
        s.write_results_many([(results(50), T)])
    plain, zipped, zipped_many = writes(influx)
    assert plain["encoding"] is None and zipped["encoding"] == "gzip" and zipped_many["encoding"] == "gzip"
    assert zipped["lines"] == plain["lines"] == zipped_many["lines"]
    assert zipped["raw_bytes"] < plain["raw_bytes"] / 2
    with pytest.raises(ValueError):
        sink(influx, write_compression="zstd")


def test_connector_gzip_client_and_shared_key(influx):
    login = {"url": url(influx), "token": "t", "org": "o", "timeout_s": 5.0}
    client = InfluxdbActions.create_client(**login, write_compression="gzip")
    try:
        InfluxdbActions.write_lines(client, "b", [f"m,k=v x={i} {i}" for i in range(100)])
        assert InfluxdbActions.ping(client)
    finally:
        client.close()
    (request,) = writes(influx)
    assert request["encoding"] == "gzip" and len(request["lines"]) == 100
    assert shared_clients.key(**login) != shared_clients.key(**login, write_compression="gzip")
    assert shared_clients.key(**login) == shared_clients.key(**login, write_compression=None)
    with pytest.raises(ValueError):
        InfluxdbActions.create_client(**login, write_compression="brotli")


def _pulse(server, options):
    params = PulseParameters(
        model_id="m",
        model_version="1",
        data_id="d",
        reference_data_storage="none",
        target_name="y",
        storage_engine="influxdb",
        login={"url": url(server), "token": "t", "org": "o", "bucket_name": "b"},
        timestamp_column_name="_time",
    )
    return Pulse(params, options)


@pytest.mark.parametrize("mode", ["sync", "background"])
def test_pulse_write_compression(influx, mode):
    frame = pd.DataFrame({"f0": [float(i) for i in range(200)], "f1": [0.5] * 200})
    preds = pd.DataFrame({"y": [1.0] * 200})
    from pulsar_data_collection.models import DataWithPrediction

    data = DataWithPrediction(timestamp=T, predictions=preds, data_points=frame, features_names=["f0", "f1"])
    for compression in (None, "gzip"):
        pulse = _pulse(influx, CaptureOptions(mode=mode, write_compression=compression))
        pulse.capture_data(data)
        assert pulse.flush(10)
        pulse.close()
    plain, zipped = writes(influx)
    assert plain["encoding"] is None and zipped["encoding"] == "gzip"
    assert len(plain["lines"]) == len(zipped["lines"]) == 200
    assert zipped["raw_bytes"] < plain["raw_bytes"]


def test_capture_options_refuse_unknown_compression():
    with pytest.raises(ValueError):
        CaptureOptions(write_compression="zstd")
    assert CaptureOptions().write_compression is None


@pytest.mark.parametrize(
    "spool, given, expected", [(False, None, False), (True, None, True), (True, False, False), (False, True, True)]
)
def test_pulse_bound_in_flight_default_follows_spool_dir(influx, tmp_path, spool, given, expected):
    options = CaptureOptions(spool_dir=tmp_path if spool else None, bound_in_flight=given)
    pulse = _pulse(influx, options)
    try:
        channel = getattr(pulse._writer, "_channel", pulse._writer)
        assert channel.bound_in_flight is expected
    finally:
        pulse.close()
