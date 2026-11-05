"""Data control without InfluxDB: policy, selector, audit journal, spool and outbox purge, controller with a fake client"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from pulsar_data_collection.evidence import DATABASE_NAME, EvidenceOutbox
from pulsar_data_collection.governance import (
    AuditJournal,
    DataController,
    DataSelector,
    Period,
    load_policy,
    parse_duration,
    purge_outbox,
    purge_spool,
)
from pulsar_data_collection.governance.__main__ import main
from pulsar_data_collection.governance.policy import BucketPolicy, default_shard_group_s, format_duration
from pulsar_data_collection.spool import Spool

TOKEN = "tok-governance-synthetic-secret"  # synthetic


# ------------------------------------------------------------------ policy
def test_durations():
    assert parse_duration("90d") == 90 * 86400
    assert parse_duration("12h") == 43200
    assert parse_duration("2w") == 14 * 86400
    assert parse_duration("1y") == 365 * 86400
    assert parse_duration(7200) == 7200 and parse_duration("7200") == 7200
    assert parse_duration("infinite") == 0 and parse_duration(0) == 0
    assert parse_duration(None) is None
    for bad in ("90 days", "-1d", -5, True, "1.5d"):
        with pytest.raises(ValueError):
            parse_duration(bad)
    assert format_duration(90 * 86400) == "90d" and format_duration(0) == "infinite" and format_duration(3601) == "3601s"
    assert default_shard_group_s(3600) == 3600 and default_shard_group_s(30 * 86400) == 86400
    assert default_shard_group_s(0) == 7 * 86400 and default_shard_group_s(400 * 86400) == 7 * 86400


def test_bucket_policy_refuses_less_than_one_hour():
    with pytest.raises(ValidationError):
        BucketPolicy(name="b", retention="30m")
    assert BucketPolicy(name="b", retention="2d").shard_group_s == 86400
    assert BucketPolicy(name="b", retention="2d", shard_group_duration="1h").shard_group_s == 3600
    assert BucketPolicy(name="b").retention_s is None


def policy_dict(tmp_path, **extra) -> dict:
    return {
        "influxdb": {"url": "http://127.0.0.1:1", "org": "ministry"},
        "buckets": {"raw": {"name": "captures", "retention": "90d"}, "results": {"name": "results", "retention": "2y"}},
        "journal": str(tmp_path / "journal.jsonl"),
        **extra,
    }


def test_policy_refuses_a_token_in_the_file(tmp_path):
    data = policy_dict(tmp_path)
    data["influxdb"]["token"] = TOKEN
    with pytest.raises(ValidationError) as error:
        load_policy(data)
    assert TOKEN not in str(error.value)


def test_policy_files_json_and_yaml_with_relative_paths(tmp_path):
    data = policy_dict(tmp_path, spool_dirs=["spool"], outbox_dirs=["outbox"])
    data["journal"] = "audit/journal.jsonl"
    (tmp_path / "policy.json").write_text(json.dumps(data))
    policy = load_policy(tmp_path / "policy.json")
    assert policy.journal == tmp_path / "audit" / "journal.jsonl"
    assert policy.spool_dirs == [tmp_path / "spool"] and policy.outbox_dirs == [tmp_path / "outbox"]
    yaml = pytest.importorskip("yaml")
    (tmp_path / "policy.yaml").write_text(yaml.safe_dump(data))
    assert load_policy(tmp_path / "policy.yaml") == policy


# ------------------------------------------------------------------ selector and period
def test_selector_predicate_and_flux_filter():
    selector = DataSelector(system_id="SYS-1", model_id="credit", tags={"environment": "prod"})
    assert selector.predicate() == 'system_id="SYS-1" AND model_id="credit" AND environment="prod"'
    assert (
        selector.flux_filter() == '(r) => r["system_id"] == "SYS-1" and r["model_id"] == "credit" and r["environment"] == "prod"'
    )
    measured = DataSelector(model_id="m", measurement="m_1_input_data")
    assert measured.predicate() == '_measurement="m_1_input_data" AND model_id="m"'
    everything = DataSelector(all_series=True)
    assert everything.predicate() == "" and everything.flux_filter() == "(r) => true"


def test_selector_refuses_empty_ambiguous_and_injected_values():
    with pytest.raises(ValidationError):
        DataSelector()
    with pytest.raises(ValidationError):
        DataSelector(model_id="m", all_series=True)
    for bad in ('a" OR model_id="b', "a\\b", "a\nb", ""):
        with pytest.raises(ValidationError):
            DataSelector(model_id=bad)
    for key in ("bad-key", "_field", "1x", "model_id"):
        with pytest.raises(ValidationError):
            DataSelector(tags={key: "v"})


def test_selector_matching():
    selector = DataSelector(system_id="A", model_id="m")
    assert selector.matches({"system_id": "A", "model_id": "m", "data_id": "x"})
    assert not selector.matches({"system_id": "B", "model_id": "m"})
    assert not DataSelector(model_id="m", measurement="x").matches({"model_id": "m"}, "y")
    assert selector.evidence_applicable() and not DataSelector(model_id="m", data_id="d").evidence_applicable()
    assert selector.matches_evidence({"system_id": "A", "model_id": "m", "model_version": "1"})


def test_period_is_inclusive_and_covers_everything_by_default():
    period = Period("2026-11-01T00:00:00Z", "2026-11-02T00:00:00Z")
    assert period.flux_range() == "start: 2026-11-01T00:00:00Z, stop: 2026-11-02T00:00:00.000000001Z"
    assert Period().describe() == {"start": "1677-09-21T00:12:44Z", "stop": "2262-04-11T23:47:16Z", "bounded": False}
    assert Period("1969-12-31T23:59:59.5Z").describe()["start"] == "1969-12-31T23:59:59.500000000Z"
    assert Period("2026-11-01T02:00:00+02:00").start == pd.Timestamp("2026-11-01T00:00:00Z")
    assert Period("2026-11-01T00:00:00").start == pd.Timestamp("2026-11-01T00:00:00Z")  # naive: UTC
    with pytest.raises(ValueError):
        Period("2026-11-02", "2026-11-01")
    stamps = pd.Series([period.start.value - 1, period.start.value, period.stop.value, period.stop.value + 1])
    assert period.contains_ns(stamps).tolist() == [False, True, True, False]


# ------------------------------------------------------------------ audit journal
def test_journal_chain_verifies(tmp_path):
    journal = AuditJournal(tmp_path / "j" / "journal.jsonl")
    assert journal.verify()["ok"] and journal.head() is None
    first = journal.append("delete", "alice", {"predicate": 'system_id="A"', "points": 3})
    second = journal.append("retention.apply", "bob", {"bucket": "b", "retention_s": 3600})
    assert second["prev"] == first["hash"] and second["seq"] == 2
    result = journal.verify(expected_head=first["hash"])
    assert result == {"ok": True, "entries": 2, "head": second["hash"], "error": None, "line": None}


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines(keepends=True)


@pytest.fixture
def journal(tmp_path):
    journal = AuditJournal(tmp_path / "journal.jsonl")
    for i in range(4):
        journal.append("delete", "alice", {"points": i})
    return journal


def test_journal_detects_a_modified_entry(journal):
    lines = _lines(journal.path)
    lines[1] = lines[1].replace('"points":1', '"points":100')
    journal.path.write_text("".join(lines), encoding="utf-8")
    result = journal.verify()
    assert not result["ok"] and result["line"] == 2 and "modified" in result["error"]


def test_journal_detects_a_removed_or_reordered_entry(journal):
    lines = _lines(journal.path)
    journal.path.write_text("".join(lines[:1] + lines[2:]), encoding="utf-8")
    assert journal.verify()["line"] == 2
    journal.path.write_text("".join([lines[0], lines[2], lines[1], lines[3]]), encoding="utf-8")
    assert not journal.verify()["ok"]


def test_journal_detects_a_rehashed_entry_after_the_modified_one(journal):
    """Recomputing the hash of a modified entry still breaks the next entry's link"""
    lines = _lines(journal.path)
    entry = json.loads(lines[1])
    entry.pop("hash")
    entry["details"]["points"] = 100
    entry["hash"] = journal._digest(entry)
    lines[1] = json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n"
    journal.path.write_text("".join(lines), encoding="utf-8")
    result = journal.verify()
    assert not result["ok"] and result["line"] == 3


def test_journal_truncation_is_detected_against_a_recorded_head(journal):
    head = journal.head()
    journal.path.write_text("".join(_lines(journal.path)[:2]), encoding="utf-8")
    assert journal.verify()["ok"]  # a shorter chain is still a chain
    result = journal.verify(expected_head=head)
    assert not result["ok"] and "expected head" in result["error"]


def test_journal_refuses_secrets(tmp_path):
    journal = AuditJournal(tmp_path / "journal.jsonl")
    for details in ({"token": TOKEN}, {"login": {"Password": "x"}}, {"items": [{"api_key": "x"}]}):
        with pytest.raises(ValueError):
            journal.append("delete", "alice", details)
    assert not journal.path.exists() or TOKEN not in journal.path.read_text()


def test_keyed_journal(tmp_path, monkeypatch):
    monkeypatch.setenv("PULSAR_TEST_JOURNAL_KEY", "k1-synthetic")
    journal = AuditJournal(tmp_path / "journal.jsonl", key_env="PULSAR_TEST_JOURNAL_KEY")
    journal.append("delete", "alice", {"points": 1})
    assert journal.verify()["ok"]
    assert not AuditJournal(journal.path).verify()["ok"]  # plain SHA-256 cannot reproduce it
    monkeypatch.setenv("PULSAR_TEST_JOURNAL_KEY", "another")
    assert not journal.verify()["ok"]
    assert "k1-synthetic" not in journal.path.read_text()
    monkeypatch.delenv("PULSAR_TEST_JOURNAL_KEY")
    with pytest.raises(RuntimeError):
        journal.append("delete", "alice", {})


# ------------------------------------------------------------------ spool
T0 = pd.Timestamp("2026-11-01T00:00:00Z")


def meta(system: str, model: str = "m") -> dict:
    return {
        "bucket_name": "captures",
        "data_frame_measurement_name": f"{model}_1_input_data",
        "data_frame_timestamp_column": "_time",
        "tags": {"model_id": model, "model_version": "1", "data_id": "prod", "system_id": system},
    }


def frame(rows: int, start: pd.Timestamp = T0) -> pd.DataFrame:
    out = pd.DataFrame({"x": np.arange(rows, dtype="float64"), "label": ["a"] * rows})
    out["_time"] = pd.DatetimeIndex(start + pd.to_timedelta(np.arange(rows) * 1000, unit="ns"))
    return out


@pytest.mark.parametrize("compression", [None, "gzip"])
def test_spool_purge_deletes_only_the_selected_system(tmp_path, compression):
    spool = Spool(tmp_path, compression=compression)
    spool.save(frame(5), meta("A"))
    spool.save(frame(7), meta("B"))
    spool.save(frame(3), meta("A"), wal=True)
    selector = DataSelector(system_id="A")
    dry = purge_spool(tmp_path, selector, dry_run=True)
    assert dry["rows"] == 8 and dry["files_deleted"] == 2 and len(list(tmp_path.iterdir())) == 3
    report = purge_spool(tmp_path, selector)
    assert report["rows"] == 8 and report["files_deleted"] == 2 and report["unreadable"] == []
    left = Spool(tmp_path, compression=compression)
    assert len(left.files()) == 1 and left.load(left.files()[0])[1]["tags"]["system_id"] == "B"


def test_spool_purge_of_a_period_rewrites_a_batch_keeping_its_place(tmp_path):
    spool = Spool(tmp_path, compression="gzip")
    first = spool.save(frame(10), meta("A"))
    second = spool.save(frame(4, T0 + pd.Timedelta(1, "h")), meta("A"))
    period = Period(T0 + pd.Timedelta(5, "us"), T0 + pd.Timedelta(2, "h"))  # rows 5..9 of the first, all the second
    report = purge_spool(tmp_path, DataSelector(model_id="m"), period)
    assert report == {
        "directory": str(tmp_path),
        "exists": True,
        "files_deleted": 1,
        "files_rewritten": 1,
        "rows": 9,
        "unreadable": [],
    }
    files = Spool(tmp_path).files()
    assert len(files) == 1 and not second.exists() and not first.exists()
    assert files[0].name.split("_")[:2] == first.name.split("_")[:2] and files[0].name.endswith("_5r.json.gz")
    kept, kept_meta = Spool(tmp_path).load(files[0])
    assert kept["x"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0] and kept_meta == meta("A")
    assert Spool.rows(files[0]) == 5 and not list(tmp_path.glob("*.purge"))


def test_spool_purge_reports_unreadable_files_and_removes_leftovers(tmp_path):
    spool = Spool(tmp_path)
    spool.save(frame(2), meta("A"))
    (tmp_path / "1_abc_2r.json.bad").write_text("not json")
    (tmp_path / "2_abd_2r.json.purge").write_text("{}")
    (tmp_path / "notes.txt").write_text("not a batch")
    report = purge_spool(tmp_path, DataSelector(system_id="A"))
    assert report["files_deleted"] == 1 and report["unreadable"] == ["1_abc_2r.json.bad"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["1_abc_2r.json.bad", "notes.txt"]
    assert purge_spool(tmp_path / "missing", DataSelector(system_id="A"))["exists"] is False


# ------------------------------------------------------------------ outbox
def evidence(i: int, system: str, start: str = "2026-10-26T00:00:00Z", end: str = "2026-10-27T00:00:00Z") -> dict:
    return {
        "schema_version": "pulsar.evidence/1",
        "evidence_id": f"{system}-{i}",
        "system_id": system,
        "model_id": "credit_model",
        "model_version": "1.0",
        "window": {"start": start, "end": end},
    }


def test_outbox_purge_deletes_pending_and_quarantined_records_of_one_system(tmp_path):
    outbox = EvidenceOutbox("https://receiver.invalid/evidence", tmp_path, start=False)
    outbox.put_many([evidence(i, "A") for i in range(3)] + [evidence(i, "B") for i in range(2)])
    with outbox._lock:
        outbox._quarantine([outbox._batch()[0]], 400, "synthetic rejection")  # A-0 quarantined
    outbox.close(flush=False)
    selector = DataSelector(system_id="A")
    assert purge_outbox(tmp_path, selector, dry_run=True) | {"directory": None} == {
        "directory": None,
        "exists": True,
        "pending": 2,
        "quarantined": 1,
        "skipped": None,
    }
    report = purge_outbox(tmp_path, selector)
    assert report["pending"] == 2 and report["quarantined"] == 1
    reopened = EvidenceOutbox("https://receiver.invalid/evidence", tmp_path, start=False)
    try:
        assert reopened.stats()["pending"] == 2 and reopened.quarantined() == []
        assert [r[1] for r in reopened._batch()] == ["B-0", "B-1"]
        assert reopened.put(evidence(9, "A"))  # still usable
    finally:
        reopened.close(flush=False)


def test_outbox_purge_by_period_and_inapplicable_selectors(tmp_path):
    outbox = EvidenceOutbox("https://receiver.invalid/evidence", tmp_path, start=False)
    outbox.put_many([evidence(1, "A"), evidence(2, "A", "2026-11-01T00:00:00Z", "2026-11-02T00:00:00Z")])
    outbox.close(flush=False)
    skipped = purge_outbox(tmp_path, DataSelector(system_id="A", data_id="prod"))
    assert skipped["pending"] == 0 and "system_id" in skipped["skipped"]
    period = Period("2026-11-01T12:00:00Z", "2026-11-05T00:00:00Z")
    assert purge_outbox(tmp_path, DataSelector(system_id="A"), period)["pending"] == 1
    assert purge_outbox(tmp_path, DataSelector(system_id="A"))["pending"] == 1
    assert purge_outbox(tmp_path / "none", DataSelector(system_id="A")) == {
        "directory": str(tmp_path / "none"),
        "exists": False,
        "pending": 0,
        "quarantined": 0,
        "skipped": None,
    }
    assert not (tmp_path / "none" / DATABASE_NAME).exists()  # never creates a store


# ------------------------------------------------------------------ controller with a fake InfluxDB client
class FakeInflux:
    """Buckets, a scripted count per query and the delete calls"""

    def __init__(self, buckets: dict):
        self.buckets = {
            name: SimpleNamespace(name=name, retention_rules=[SimpleNamespace(type="expire", every_seconds=s)])
            for name, s in buckets.items()
        }
        self.points = {name: 6 for name in buckets}
        self.deleted = []
        self.updated = []

    def buckets_api(self):
        fake = self

        class Api:
            def find_bucket_by_name(self, name):
                return fake.buckets.get(name)

            def update_bucket(self, bucket):
                fake.updated.append((bucket.name, bucket.retention_rules[0].every_seconds))
                return bucket

        return Api()

    def query_api(self):
        fake = self

        class Api:
            def query(self, flux, org=None):
                name = flux.split('"')[1]
                value = fake.points[name] // (3 if "sum(" in flux else 1)
                record = SimpleNamespace(values={"_measurement": "m_1_input_data", "_value": value, "n": value})
                return [SimpleNamespace(records=[record])] if value else []

        return Api()

    def delete_api(self):
        fake = self

        class Api:
            def delete(self, start, stop, predicate, bucket, org):
                fake.deleted.append((start, stop, predicate, bucket, org))
                fake.points[bucket] = 0

        return Api()


def test_retention_drift_is_reported_and_applied_with_a_journal_entry(tmp_path):
    fake = FakeInflux({"captures": 0, "results": 2 * 365 * 86400})
    controller = DataController(policy_dict(tmp_path), client=fake, actor="dpo")
    check = {row["role"]: row for row in controller.check_retention()}
    assert check["raw"]["status"] == "drift" and check["raw"]["actual"] == "infinite"
    assert check["results"]["status"] == "ok"
    assert controller.apply_retention(dry_run=True)[0]["applied"] and fake.updated == []
    assert not controller.journal.entries()
    report = controller.apply_retention(reason="TICKET-1")
    assert fake.updated == [("captures", 90 * 86400)] and report[0]["status"] == "ok"
    (entry,) = controller.journal.entries()
    assert entry["action"] == "retention.apply" and entry["actor"] == "dpo"
    assert entry["details"] | {} == {
        "role": "raw",
        "bucket": "captures",
        "previous_s": 0,
        "retention_s": 90 * 86400,
        "shard_group_duration_s": 86400,
        "verified": True,
        "reason": "TICKET-1",
    }


def test_missing_bucket_is_reported_never_created(tmp_path):
    fake = FakeInflux({"captures": 0})
    rows = DataController(policy_dict(tmp_path), client=fake).apply_retention()
    assert {r["role"]: r["status"] for r in rows} == {"raw": "ok", "results": "missing"} and fake.updated == [
        ("captures", 90 * 86400)
    ]


def test_delete_counts_deletes_verifies_purges_and_journals(tmp_path):
    spool_dir, outbox_dir = tmp_path / "spool", tmp_path / "outbox"
    Spool(spool_dir).save(frame(4), meta("A"))
    outbox = EvidenceOutbox("https://receiver.invalid/evidence", outbox_dir, start=False)
    outbox.put(evidence(1, "A"))
    outbox.close(flush=False)
    fake = FakeInflux({"captures": 0, "results": 0})
    policy = policy_dict(tmp_path, spool_dirs=[str(spool_dir)], outbox_dirs=[str(outbox_dir)])
    controller = DataController(policy, client=fake, actor="dpo")
    selector = DataSelector(system_id="A")

    dry = controller.delete(selector, dry_run=True)
    assert dry["buckets"]["raw"]["points"] == 6 and dry["buckets"]["raw"]["rows"] == 2
    assert dry["spool"][0]["rows"] == 4 and dry["outbox"][0]["pending"] == 1
    assert fake.deleted == [] and not controller.journal.entries() and len(list(spool_dir.iterdir())) == 1

    report = controller.delete(selector, stop="2026-12-01T00:00:00Z", reason="request 42")
    assert report["status"] == "ok" and report["buckets"]["raw"]["remaining"] == 0
    assert fake.deleted[0] == ("1677-09-21T00:12:44Z", "2026-12-01T00:00:00Z", 'system_id="A"', "captures", "ministry")
    assert not list(spool_dir.iterdir()) and report["outbox"][0]["pending"] == 1
    (entry,) = controller.journal.entries()
    assert entry["action"] == "delete" and entry["details"]["predicate"] == 'system_id="A"'
    assert entry["details"]["spool"][0]["rows"] == 4 and entry["details"]["reason"] == "request 42"
    text = controller.journal.path.read_text()
    assert TOKEN not in text and '"label"' not in text and '"x"' not in text  # no token, no data value


def test_controller_needs_a_token_only_to_connect(tmp_path, monkeypatch):
    monkeypatch.delenv("PULSAR_INFLUX_TOKEN", raising=False)
    controller = DataController(policy_dict(tmp_path, central_service="https://user:pw@central.example/api?x=1"))
    description = controller.describe(check=False)
    assert description["leaves_the_client"]["destination"] == "https://central.example/api"
    assert [b["bucket"] for b in description["influxdb"]["buckets"]] == ["captures", "results"]
    with pytest.raises(ValueError, match="PULSAR_INFLUX_TOKEN"):
        controller.check_retention()
    monkeypatch.setenv("PULSAR_INFLUX_TOKEN", TOKEN)
    controller = DataController(policy_dict(tmp_path))
    assert TOKEN not in repr(controller) and TOKEN not in json.dumps(controller.describe(check=False))
    controller.close()


def test_cli_describe_offline_and_journal_verify(tmp_path, capsys):
    (tmp_path / "policy.json").write_text(json.dumps(policy_dict(tmp_path, spool_dirs=["spool"])))
    assert main(["--policy", str(tmp_path / "policy.json"), "describe", "--offline"]) == 0
    described = json.loads(capsys.readouterr().out)
    assert described["spool_dirs"] == [{"path": str(tmp_path / "spool")}]
    journal = AuditJournal(tmp_path / "journal.jsonl")
    journal.append("delete", "alice", {"points": 1})
    assert main(["--policy", str(tmp_path / "policy.json"), "journal", "verify"]) == 0
    capsys.readouterr()
    journal.path.write_text(journal.path.read_text().replace('"points":1', '"points":2'))
    assert main(["--policy", str(tmp_path / "policy.json"), "journal", "verify"]) == 1
    assert json.loads(capsys.readouterr().out)["line"] == 1


def test_cli_refuses_an_empty_or_injected_selection(tmp_path, monkeypatch, capsys):
    (tmp_path / "policy.json").write_text(json.dumps(policy_dict(tmp_path)))
    monkeypatch.setenv("PULSAR_INFLUX_TOKEN", TOKEN)  # never used: refused before any request
    base = ["--policy", str(tmp_path / "policy.json")]
    assert main(base + ["delete"]) == 2
    assert main(base + ["delete", "--system-id", 'A" OR system_id="B']) == 2
    assert "invalid selection" in capsys.readouterr().err
    assert not (tmp_path / "journal.jsonl").exists()
