"""Data control against a real InfluxDB 2: retention applied, deletion by system, dry run, export round trip, journal.

Runs when PULSAR_TEST_INFLUX_URL, PULSAR_TEST_INFLUX_TOKEN and PULSAR_TEST_INFLUX_ORG are set (e.g. the local validation
instance), skipped otherwise. The token must be allowed to create buckets: every test creates its own two buckets and
deletes them. Synthetic data only.
"""

import json
import os
import uuid

import numpy as np
import pandas as pd
import pytest
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS

from pulsar_data_collection.governance import DataController, DataSelector, verify_export
from pulsar_data_collection.governance.__main__ import main
from pulsar_data_collection.models import CaptureOptions, DataWithPrediction, PulseParameters
from pulsar_data_collection.pulse import Pulse
from pulsar_data_collection.results import ResultsSink

ENV = {k: os.environ.get(f"PULSAR_TEST_INFLUX_{k.upper()}") for k in ("url", "token", "org")}
pytestmark = pytest.mark.skipif(not all(ENV.values()), reason="PULSAR_TEST_INFLUX_* not set")
T0 = pd.Timestamp("2026-11-01T00:00:00Z")
N = 20


@pytest.fixture
def client():
    with InfluxDBClient(url=ENV["url"], token=ENV["token"], org=ENV["org"], timeout=30_000) as client:
        yield client


@pytest.fixture
def buckets(client):
    api = client.buckets_api()
    suffix = uuid.uuid4().hex[:8]
    created = {
        role: api.create_bucket(bucket_name=f"pulsar-gov-test-{role}-{suffix}", org=ENV["org"]) for role in ("raw", "results")
    }
    yield {role: bucket.name for role, bucket in created.items()}
    for bucket in created.values():
        api.delete_bucket(bucket)


@pytest.fixture
def policy(buckets, tmp_path):
    return {
        "influxdb": {"url": ENV["url"], "org": ENV["org"]},
        "buckets": {
            "raw": {"name": buckets["raw"], "retention": "30d", "content": "raw captures"},
            "results": {"name": buckets["results"], "retention": "infinite", "content": "analysis results"},
        },
        "spool_dirs": [str(tmp_path / "spool")],
        "outbox_dirs": [str(tmp_path / "outbox")],
        "journal": str(tmp_path / "audit" / "journal.jsonl"),
        "export_dir": str(tmp_path / "exports"),
    }


@pytest.fixture
def controller(policy):
    with DataController(policy, token=ENV["token"], actor="test-dpo") as controller:
        yield controller


def capture(bucket: str, system: str, model: str = "credit", start: pd.Timestamp = T0, rows: int = N, seed: int = 0):
    params = PulseParameters(
        model_id=model,
        model_version="1",
        data_id="prod",
        reference_data_storage="reference.csv",
        target_name="y",
        storage_engine="influxdb",
        timestamp_column_name="_time",
        additional_tags={"system_id": system},
        login={"url": ENV["url"], "token": ENV["token"], "org": ENV["org"], "bucket_name": bucket},
    )
    rng = np.random.default_rng(seed)
    data = pd.DataFrame({"income": rng.normal(50_000, 5_000, rows), "region": rng.choice(["north", "south"], rows)})
    data["age"] = rng.integers(18, 90, rows)
    predictions = pd.DataFrame({"score": rng.random(rows)})
    with Pulse(params, CaptureOptions(mode="sync")) as pulse:
        pulse.capture_data(
            DataWithPrediction(data_points=data, predictions=predictions, timestamp=start, features_names=list(data))
        )
    frame = pd.concat([data, predictions], axis=1)
    frame["_time"] = pd.DatetimeIndex(start + pd.to_timedelta(np.arange(rows) * 1000, unit="ns"))
    return frame


def write_results(bucket: str, system: str):
    with ResultsSink(ENV["url"], ENV["token"], ENV["org"], bucket) as sink:
        sink.write_results(
            [
                {
                    "metric_type": "drift",
                    "metric_name": "psi",
                    "feature_name": "income",
                    "metric_value": 0.1,
                    "drift_status": False,
                }
            ],
            timestamp=T0.to_pydatetime(),
            tags={"model_id": "credit", "model_version": "1", "system_id": system},
        )


def test_retention_is_applied_and_drift_reported(controller, buckets, client):
    rows = {r["role"]: r for r in controller.check_retention()}
    assert rows["raw"]["status"] == "drift" and rows["raw"]["actual"] == "infinite"
    assert rows["results"]["status"] == "ok"
    applied = {r["role"]: r for r in controller.apply_retention(reason="test")}
    assert applied["raw"]["status"] == "ok" and applied["raw"]["applied"] and not applied["results"]["applied"]
    bucket = client.buckets_api().find_bucket_by_name(buckets["raw"])
    assert bucket.retention_rules[0].every_seconds == 30 * 86400
    assert bucket.retention_rules[0].shard_group_duration_seconds == 86400
    assert all(r["status"] == "ok" for r in controller.check_retention())
    # a shorter retention than the bucket's shard group duration (7 days by default) is accepted too
    controller.policy.buckets["raw"].retention = "2h"
    assert controller.apply_retention()[0]["status"] == "ok"
    entries = controller.journal.entries()
    assert [e["action"] for e in entries] == ["retention.apply", "retention.apply"]
    assert entries[1]["details"]["previous_s"] == 30 * 86400 and entries[1]["details"]["retention_s"] == 7200
    assert controller.journal.verify()["ok"]
    assert ENV["token"] not in controller.journal.path.read_text()


def test_delete_by_system_removes_only_that_system(controller, buckets):
    capture(buckets["raw"], "SYS-A", seed=1)
    kept = capture(buckets["raw"], "SYS-B", seed=2)
    capture(buckets["raw"], "SYS-A", model="fraud", seed=3)
    write_results(buckets["results"], "SYS-A")
    write_results(buckets["results"], "SYS-B")
    selector = DataSelector(system_id="SYS-A")

    dry = controller.delete(selector, dry_run=True)
    assert dry["buckets"]["raw"]["rows"] == 2 * N and dry["buckets"]["raw"]["points"] == 2 * N * 4
    assert dry["buckets"]["raw"]["measurements"] == {"credit_1_input_data": N, "fraud_1_input_data": N}
    assert dry["buckets"]["results"]["rows"] == 1
    assert controller.count(selector)["raw"]["rows"] == 2 * N  # nothing changed
    assert controller.journal.entries() == []

    report = controller.delete(selector, reason="synthetic request")
    assert report["status"] == "ok"
    assert report["buckets"]["raw"]["remaining"] == 0 and report["buckets"]["results"]["remaining"] == 0
    everything = controller.count(DataSelector(all_series=True))
    assert everything["raw"]["rows"] == N and everything["raw"]["measurements"] == {
        "credit_1_input_data": {"points": 4 * N, "rows": N}
    }
    assert everything["results"]["rows"] == 1
    exported = controller.export(DataSelector(system_id="SYS-B"), roles=["raw"])
    path = os.path.join(controller.policy.export_dir, os.listdir(controller.policy.export_dir)[0], exported["files"][0]["path"])
    assert pd.read_csv(path)["income"].tolist() == pytest.approx(kept["income"].tolist())
    intent, entry, _ = controller.journal.entries()
    assert intent["action"] == "delete.intent" and entry["details"]["intent_seq"] == intent["seq"]
    assert entry["action"] == "delete" and entry["details"]["buckets"]["raw"]["rows"] == 2 * N
    assert entry["details"]["predicate"] == 'system_id="SYS-A"' and controller.journal.verify()["ok"]


def test_delete_of_a_period_is_inclusive(controller, buckets):
    capture(buckets["raw"], "SYS-A")
    start, stop = T0 + pd.Timedelta(5, "us"), T0 + pd.Timedelta(9, "us")  # rows 5 to 9
    selector = DataSelector(model_id="credit")
    assert controller.delete(selector, start, stop, roles=["raw"], dry_run=True)["buckets"]["raw"]["rows"] == 5
    report = controller.delete(selector, start, stop, roles=["raw"])
    assert report["buckets"]["raw"]["rows"] == 5 and report["buckets"]["raw"]["remaining"] == 0
    assert controller.count(selector)["raw"]["rows"] == N - 5
    assert controller.count(selector, T0, T0 + pd.Timedelta(4, "us"))["raw"]["rows"] == 5


def test_export_round_trip_with_checksums(controller, buckets, tmp_path):
    original = capture(buckets["raw"], "SYS-A", seed=7)
    capture(buckets["raw"], "SYS-B", seed=8)
    write_results(buckets["results"], "SYS-A")
    out = tmp_path / "export"
    manifest = controller.export(DataSelector(system_id="SYS-A"), out, reason="client copy")
    assert manifest["rows"] == N + 1 and manifest["selector"] == {"system_id": "SYS-A"}
    assert json.loads((out / "manifest.json").read_text())["files"] == manifest["files"]
    raw = next(f for f in manifest["files"] if f["role"] == "raw")
    assert raw["measurement"] == "credit_1_input_data" and raw["rows"] == N
    assert raw["tags"] == ["data_id", "model_id", "model_version", "system_id"]
    frame = pd.read_csv(out / raw["path"])
    assert list(frame.columns[:6]) == ["_time", "_measurement", "data_id", "model_id", "model_version", "system_id"]
    assert set(frame["system_id"]) == {"SYS-A"}
    assert (pd.to_datetime(frame["_time"], utc=True).astype("int64") == original["_time"].astype("int64")).all()  # exact ns
    assert frame["income"].tolist() == pytest.approx(original["income"].tolist(), rel=1e-15)
    assert frame["age"].tolist() == original["age"].tolist() and frame["region"].tolist() == original["region"].tolist()
    assert verify_export(out) == {"ok": True, "files": 2, "errors": []}
    with open(out / raw["path"], "a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    check = verify_export(out)
    assert not check["ok"] and "checksum" in check["errors"][0]
    (entry,) = controller.journal.entries()
    assert entry["action"] == "export" and entry["details"]["reason"] == "client copy"
    with pytest.raises(FileExistsError):
        controller.export(DataSelector(system_id="SYS-A"), out)


def test_export_keeps_sparse_integers_and_booleans(controller, buckets, client, tmp_path):
    big = 2**60 + 1  # not a float64
    lines = [
        f'events,system_id=SYS-A n={big}i,flag=true,note="a" {T0.value}',
        f"events,system_id=SYS-A x=0.5 {T0.value + 1}",  # the same series: n, flag and note are missing here
    ]
    client.write_api(write_options=SYNCHRONOUS).write(bucket=buckets["raw"], org=ENV["org"], record=lines)
    manifest = controller.export(DataSelector(system_id="SYS-A"), tmp_path / "e", roles=["raw"])
    (item,) = manifest["files"]
    assert item["columns"]["n"] == "Int64" and item["influx_types"]["n"] == "long"
    assert item["columns"]["flag"] == "boolean" and item["columns"]["note"] == "string"
    rows = (tmp_path / "e" / item["path"]).read_text(encoding="utf-8").splitlines()
    assert rows[0] == '"_time","_measurement","system_id","flag","n","note","x"'
    assert rows[1].endswith(f',"SYS-A",True,{big},"a",') and rows[2].endswith(',"SYS-A",,,,0.5')


def test_export_parquet(controller, buckets, tmp_path):
    pytest.importorskip("pyarrow")
    original = capture(buckets["raw"], "SYS-A")
    manifest = controller.export(DataSelector(system_id="SYS-A"), tmp_path / "p", roles=["raw"], format="parquet")
    frame = pd.read_parquet(tmp_path / "p" / manifest["files"][0]["path"])
    assert frame["_time"].astype("int64").tolist() == original["_time"].astype("int64").tolist()


def test_describe_and_cli(controller, policy, buckets, tmp_path, monkeypatch, capsys):
    description = controller.describe()
    roles = {b["role"]: b for b in description["influxdb"]["buckets"]}
    assert roles["raw"]["retention_status"] == "drift" and roles["results"]["retention_status"] == "ok"
    assert description["spool_dirs"][0]["exists"] is False
    assert ENV["token"] not in json.dumps(description)
    (tmp_path / "policy.json").write_text(json.dumps(policy))
    monkeypatch.setenv("PULSAR_INFLUX_TOKEN", ENV["token"])
    base = ["--policy", str(tmp_path / "policy.json"), "--actor", "cli-test"]
    assert main(base + ["retention", "check"]) == 1  # drift
    assert main(base + ["retention", "apply"]) == 0
    assert main(base + ["retention", "check"]) == 0
    capture(buckets["raw"], "SYS-A")
    capsys.readouterr()
    assert main(base + ["delete", "--system-id", "SYS-A", "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["buckets"]["raw"]["rows"] == N
    assert main(base + ["delete", "--system-id", "SYS-A", "--reason", "cli"]) == 0
    assert main(base + ["journal", "verify"]) == 0
    assert ENV["token"] not in capsys.readouterr().out
