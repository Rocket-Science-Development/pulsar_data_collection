"""Data control of one deployment, run by the client without the vendor: retention, deletion, export, residence.

Every retention change, deletion and export is written to the audit journal (see journal.py).
"""

import getpass
import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Iterable, Literal, Optional, Union

import pandas as pd
from influxdb_client import BucketRetentionRules, InfluxDBClient

from ..evidence import _safe_url
from .journal import AuditJournal
from .local import outbox_summary, purge_outbox, purge_spool, spool_summary
from .policy import GovernancePolicy, format_duration, load_policy
from .selection import DataSelector, Period, TimeLike

logger = logging.getLogger("pulsar_data_collection")

EXPORT_SCHEMA = "pulsar.export/1"
MANIFEST_NAME = "manifest.json"
_DROPPED_COLUMNS = ["result", "table", "_start", "_stop"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _version() -> str:
    """Version of the code that runs: the source tree's pyproject.toml when the package is imported from a checkout
    (an editable install keeps the metadata of the version it was installed at), else the installed metadata"""
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8") if pyproject.is_file() else ""
    if 'name = "pulsar-data-collection"' in text:
        for line in text.splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "version" and value.strip():
                return value.strip().strip("\"'")
    try:
        return metadata.version("pulsar-data-collection")
    except metadata.PackageNotFoundError:  # pragma: no cover - source checkout without install
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_name(measurement: str, used: set) -> str:
    base = re.sub(r"[^A-Za-z0-9_.-]", "_", measurement).strip(".") or "measurement"
    name, index = base, 1
    while name.lower() in used:
        index += 1
        name = f"{base}_{index}"
    used.add(name.lower())
    return name


def _csv(frame: pd.DataFrame, path: Path) -> None:
    """CSV with the times in RFC 3339 with nanoseconds (UTC), so that nothing is rounded"""
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[column]):
            values = out[column].dt.tz_convert("UTC") if out[column].dt.tz is not None else out[column]
            nanos = values.dt.as_unit("ns").astype("int64") % 1_000_000_000
            out[column] = values.dt.strftime("%Y-%m-%dT%H:%M:%S.") + nanos.astype(str).str.zfill(9) + "Z"
    out.to_csv(path, index=False)


def _string(value: str) -> str:
    """Flux string literal (bucket names come from the policy; tag values are already checked by DataSelector)"""
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


class DataController:
    """Retention, deletion, export and residence of the data of one deployment, as declared by a GovernancePolicy

    Parameters
    ----------
    policy : GovernancePolicy, dict or path
        The policy (a JSON or YAML file, see policy.py)
    token : str, optional
        InfluxDB token; default: the environment variable named by `policy.influxdb.token_env`. Never logged nor
        journaled.
    actor : str, optional
        Who acts, recorded in the journal (default: the operating system user)
    client : InfluxDBClient, optional
        Client to use instead of one built from the policy (tests)
    """

    def __init__(
        self,
        policy: Union[GovernancePolicy, dict, str, os.PathLike],
        token: Optional[str] = None,
        actor: Optional[str] = None,
        client: Optional[InfluxDBClient] = None,
    ):
        self.policy = load_policy(policy)
        self.journal = AuditJournal(self.policy.journal, key_env=self.policy.journal_key_env)
        self.actor = actor or getpass.getuser()
        self._token = token
        self._owns_client = client is None
        self._client_instance = client

    @property
    def _client(self) -> InfluxDBClient:
        """Client built at first use: describe(check=False) needs neither InfluxDB nor its token"""
        if self._client_instance is None:
            location = self.policy.influxdb
            token = self._token or os.environ.get(location.token_env)
            if not token:
                raise ValueError(f"InfluxDB token: pass token= or set the environment variable {location.token_env}")
            self._client_instance = InfluxDBClient(
                url=location.url, token=token, org=location.org, timeout=int(location.timeout_s * 1000)
            )
            self._token = None  # kept by the client only
        return self._client_instance

    def __repr__(self) -> str:
        return f"DataController(url={_safe_url(self.policy.influxdb.url)!r}, org={self.policy.influxdb.org!r})"

    def close(self) -> None:
        if self._owns_client and self._client_instance is not None:
            self._client_instance.close()
            self._client_instance = None

    def __enter__(self) -> "DataController":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ helpers
    @property
    def org(self) -> str:
        return self.policy.influxdb.org

    def _roles(self, roles: Optional[Iterable[str]]) -> list[str]:
        roles = list(self.policy.buckets) if roles is None else list(roles)
        for role in roles:
            self.policy.bucket(role)  # KeyError for an unknown role
        return roles

    def _query(self, flux: str) -> list:
        return self._client.query_api().query(flux, org=self.org)

    def _find_bucket(self, name: str):
        return self._client.buckets_api().find_bucket_by_name(name)

    @staticmethod
    def _actual_retention(bucket) -> int:
        """Retention of a bucket in seconds; 0 = infinite"""
        rules = [r for r in (bucket.retention_rules or []) if getattr(r, "type", "expire") == "expire"]
        return int(rules[0].every_seconds or 0) if rules else 0

    # ------------------------------------------------------------------ retention
    def check_retention(self) -> list[dict]:
        """Declared against actual retention of every bucket of the policy

        One dict per role: role, bucket, declared_s / declared, actual_s / actual, status "ok", "drift" (differs),
        "missing" (no such bucket) or "unmanaged" (no retention declared).
        """
        report = []
        for role, declared in self.policy.buckets.items():
            row = {"role": role, "bucket": declared.name, "declared_s": declared.retention_s}
            row["declared"] = format_duration(declared.retention_s)
            bucket = self._find_bucket(declared.name)
            if bucket is None:
                row.update(actual_s=None, actual=None, status="missing")
            else:
                actual = self._actual_retention(bucket)
                row.update(actual_s=actual, actual=format_duration(actual))
                if declared.retention_s is None:
                    row["status"] = "unmanaged"
                else:
                    row["status"] = "ok" if actual == declared.retention_s else "drift"
            report.append(row)
        return report

    def apply_retention(self, dry_run: bool = False, reason: Optional[str] = None) -> list[dict]:
        """Set the declared retention on every bucket in drift; each change is journaled (before and after)

        Returns the check_retention rows, with "applied" True for the buckets changed (or that would be, dry_run).
        A missing bucket is reported, never created. A journal that could not take the entry (key not set, last line
        not a complete entry) is found before any bucket is changed.
        """
        report = self.check_retention()
        for row in report:
            row["applied"] = False
            if row["status"] != "drift":
                continue
            declared = self.policy.bucket(row["role"])
            row["applied"] = True
            if dry_run:
                continue
            self.journal.check()
            bucket = self._find_bucket(declared.name)
            previous_s = row["actual_s"]
            retention_s, shard_s = declared.retention_s, declared.shard_group_s
            bucket.retention_rules = [
                BucketRetentionRules(type="expire", every_seconds=retention_s, shard_group_duration_seconds=shard_s)
            ]
            self._client.buckets_api().update_bucket(bucket)
            after = self._actual_retention(self._find_bucket(declared.name))
            row.update(status="ok" if after == retention_s else "drift", actual_s=after, actual=format_duration(after))
            self.journal.append(
                "retention.apply",
                self.actor,
                {
                    "role": row["role"],
                    "bucket": declared.name,
                    "previous_s": previous_s,
                    "retention_s": retention_s,
                    "shard_group_duration_s": shard_s,
                    "verified": after == retention_s,
                    "reason": reason,
                },
            )
            logger.warning("Retention of bucket %s set to %s", declared.name, format_duration(retention_s))
        return report

    # ------------------------------------------------------------------ counting
    def count(
        self, selector: DataSelector, start: TimeLike = None, stop: TimeLike = None, roles: Optional[Iterable[str]] = None
    ) -> dict:
        """Points (field values) and rows (distinct series and timestamps) selected, per role and measurement

        {"<role>": {"bucket", "points", "rows", "measurements": {"<measurement>": {"points", "rows"}}}}
        """
        period = Period(start, stop)
        return {role: self._count(self.policy.bucket(role).name, selector, period) for role in self._roles(roles)}

    def _count(self, bucket: str, selector: DataSelector, period: Period) -> dict:
        base = f"from(bucket: {_string(bucket)}) |> range({period.flux_range()}) |> filter(fn: {selector.flux_filter()})"
        points = base + ' |> group(columns: ["_measurement"]) |> count()'
        rows = base + (
            ' |> drop(columns: ["_field", "_value"]) |> group(columns: ["_time"], mode: "except") |> unique(column: "_time")'
            ' |> group(columns: ["_measurement"]) |> map(fn: (r) => ({r with n: 1})) |> sum(column: "n")'
        )
        measurements: dict = {}
        for flux, key, column in ((points, "points", "_value"), (rows, "rows", "n")):
            for table in self._query(flux):
                for record in table.records:
                    entry = measurements.setdefault(record.values.get("_measurement"), {"points": 0, "rows": 0})
                    entry[key] += int(record.values.get(column) or 0)
        return {
            "bucket": bucket,
            "points": sum(m["points"] for m in measurements.values()),
            "rows": sum(m["rows"] for m in measurements.values()),
            "measurements": measurements,
        }

    # ------------------------------------------------------------------ deletion
    def delete(
        self,
        selector: DataSelector,
        start: TimeLike = None,
        stop: TimeLike = None,
        roles: Optional[Iterable[str]] = None,
        dry_run: bool = False,
        local: bool = True,
        reason: Optional[str] = None,
    ) -> dict:
        """Delete the selected data in the buckets (InfluxDB delete API), then in the spool and outbox stores

        The period is inclusive; None ends: everything. dry_run counts what would be deleted and changes nothing (not
        journaled). Otherwise, once counted and before anything is deleted, a `delete.intent` entry is journaled (the
        predicate and the counts): a journal that cannot take it (key not set, last line not a complete entry,
        secret-looking selector keys) stops the deletion before any change. The deletion is then verified by counting
        again (`remaining` must be 0) and its outcome journaled as a `delete` entry (`intent_seq` points to the
        intent), whatever it is: status "ok", "incomplete" or "error". `local=False` leaves the spool and outbox
        stores untouched.
        """
        period = Period(start, stop)
        roles = self._roles(roles)
        report = {
            "dry_run": dry_run,
            "selector": selector.describe(),
            "predicate": selector.predicate(),
            "period": period.describe(),
            "buckets": {},
            "spool": [],
            "outbox": [],
            "status": "ok",
            "errors": [],
        }
        for role in roles:
            name = self.policy.bucket(role).name
            before = self._count(name, selector, period)
            entry = {"bucket": name, "points": before["points"], "rows": before["rows"]}
            entry["measurements"] = {m: c["rows"] for m, c in before["measurements"].items()}
            report["buckets"][role] = entry
        if dry_run:
            if local:
                report["spool"] = [purge_spool(d, selector, period, dry_run=True) for d in self.policy.spool_dirs]
                report["outbox"] = [purge_outbox(d, selector, period, dry_run=True) for d in self.policy.outbox_dirs]
            return report
        intent = self.journal.append(
            "delete.intent", self.actor, {**self._journal_view(report), "status": "started", "local": local, "reason": reason}
        )
        try:
            for role in roles:
                entry = report["buckets"][role]
                if entry["points"]:
                    self._client.delete_api().delete(
                        Period.rfc3339(period.start),
                        Period.rfc3339(period.stop),
                        selector.predicate(),
                        bucket=entry["bucket"],
                        org=self.org,
                    )
                entry["remaining"] = self._count(entry["bucket"], selector, period)["points"]
                if entry["remaining"]:
                    report["status"] = "incomplete"
            if local:
                report["spool"] = [purge_spool(d, selector, period) for d in self.policy.spool_dirs]
                report["outbox"] = [purge_outbox(d, selector, period) for d in self.policy.outbox_dirs]
        except Exception as error:  # noqa: BLE001 - journaled, then raised
            report["status"] = "error"
            report["errors"].append(f"{type(error).__name__}: {str(error)[:300]}")
            raise
        finally:
            details = {**self._journal_view(report), "intent_seq": intent["seq"], "reason": reason}
            self.journal.append("delete", self.actor, details)
        return report

    @staticmethod
    def _journal_view(report: dict) -> dict:
        """What the journal keeps of a deletion: no data value, only the predicate and the counts"""
        return {
            "selector": report["selector"],
            "predicate": report["predicate"],
            "period": report["period"],
            "buckets": report["buckets"],
            "spool": [
                {k: s[k] for k in ("directory", "files_deleted", "files_rewritten", "rows")}
                | {"unreadable": len(s["unreadable"])}
                for s in report["spool"]
            ],
            "outbox": [{k: o[k] for k in ("directory", "pending", "quarantined", "skipped")} for o in report["outbox"]],
            "status": report["status"],
            "errors": report["errors"],
        }

    # ------------------------------------------------------------------ export
    def export(
        self,
        selector: DataSelector,
        directory: Optional[os.PathLike] = None,
        start: TimeLike = None,
        stop: TimeLike = None,
        roles: Optional[Iterable[str]] = None,
        format: Literal["csv", "parquet"] = "csv",
        reason: Optional[str] = None,
    ) -> dict:
        """Export the selected data to open files: one file per role and measurement (rows with the timestamp `_time`,
        the tags and one column per field), plus `manifest.json` with the selector, the period, the columns and types,
        the row counts and the SHA-256 of every file. Journaled. Returns the manifest.

        `directory`: an empty or new directory (default: a new `pulsar-export-<time>` in policy.export_dir).
        Parquet needs pyarrow. A measurement is held in memory while it is written: export long periods in parts.
        """
        if format not in ("csv", "parquet"):
            raise ValueError("format must be 'csv' or 'parquet'")
        if format == "parquet":
            try:
                import pyarrow  # noqa: F401
            except ImportError as error:
                raise ImportError("Parquet export needs pyarrow (pip install pyarrow), or use format='csv'") from error
        if directory is None:
            if self.policy.export_dir is None:
                raise ValueError("no directory given and no export_dir in the policy")
            directory = Path(self.policy.export_dir) / f"pulsar-export-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"export directory {directory} is not empty")
        self.journal.check({"selector": selector.describe()})  # an export that cannot be journaled is not written
        directory.mkdir(parents=True, exist_ok=True)
        period = Period(start, stop)
        manifest = {
            "schema_version": EXPORT_SCHEMA,
            "created_at": _now(),
            "producer": "pulsar-data-collection",
            "producer_version": _version(),
            "source": {"influxdb": _safe_url(self.policy.influxdb.url), "org": self.org},
            "selector": selector.describe(),
            "period": period.describe(),
            "format": format,
            "files": [],
        }
        for role in self._roles(roles):
            bucket = self.policy.bucket(role).name
            used: set = set()
            for measurement, frame, tags in self._frames(bucket, selector, period):
                (directory / role).mkdir(exist_ok=True)
                path = directory / role / f"{_file_name(measurement, used)}.{format}"
                if format == "csv":
                    _csv(frame, path)
                else:
                    frame.to_parquet(path, index=False)
                manifest["files"].append(
                    {
                        "path": path.relative_to(directory).as_posix(),
                        "role": role,
                        "bucket": bucket,
                        "measurement": measurement,
                        "rows": len(frame),
                        "columns": {str(c): str(t) for c, t in frame.dtypes.items()},
                        "tags": tags,
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )
        manifest["rows"] = sum(f["rows"] for f in manifest["files"])
        (directory / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        entry = self.journal.append(
            "export",
            self.actor,
            {
                "directory": str(directory),
                "selector": manifest["selector"],
                "period": manifest["period"],
                "format": format,
                "files": [{k: f[k] for k in ("path", "rows", "sha256")} for f in manifest["files"]],
                "manifest_sha256": _sha256(directory / MANIFEST_NAME),
                "reason": reason,
            },
        )
        manifest["journal_seq"] = entry["seq"]
        return manifest

    def _frames(self, bucket: str, selector: DataSelector, period: Period):
        """(measurement, rows, tag columns) of the selection, one frame per measurement, sorted by time

        Columns: `_time` (UTC, nanoseconds), `_measurement`, the tags, then the fields.
        """
        flux = (
            f"from(bucket: {_string(bucket)}) |> range({period.flux_range()}) |> filter(fn: {selector.flux_filter()})"
            ' |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")'
            " |> map(fn: (r) => ({r with _time_ns: int(v: r._time)}))"  # the client parses times to microseconds
        )
        by_measurement: dict = {}
        for table in self._query(flux):  # one table per series: its group key gives the tags
            if not table.records:
                continue
            labels = [c.label for c in table.columns]
            tags = [c.label for c in table.columns if c.group and c.label not in ("_start", "_stop", "_measurement")]
            frame = pd.DataFrame([record.values for record in table.records], columns=labels)
            frame = frame.drop(columns=[c for c in _DROPPED_COLUMNS if c in frame])
            frame["_time"] = pd.to_datetime(frame.pop("_time_ns").astype("int64"), unit="ns", utc=True)
            entry = by_measurement.setdefault(str(frame["_measurement"].iloc[0]), {"frames": [], "tags": set()})
            entry["frames"].append(frame)
            entry["tags"].update(tags)
        for measurement in sorted(by_measurement):
            entry = by_measurement[measurement]
            frame = pd.concat(entry["frames"], ignore_index=True).dropna(axis=1, how="all")
            frame = frame.sort_values("_time", kind="stable").reset_index(drop=True)
            tags = sorted(t for t in entry["tags"] if t in frame)
            fields = sorted(c for c in frame.columns if c not in tags and c not in ("_time", "_measurement"))
            yield measurement, frame[["_time", "_measurement", *tags, *fields]], tags

    # ------------------------------------------------------------------ residence
    def describe(self, check: bool = True) -> dict:
        """Where each kind of data lives, as configured; with check=True, the actual retention and the local stores'
        sizes too (an unreachable InfluxDB is reported, not raised). Never includes the token."""
        location = self.policy.influxdb
        buckets = []
        actual: dict = {}
        if check:
            try:
                actual = {row["role"]: row for row in self.check_retention()}
            except Exception as error:  # noqa: BLE001 - describe must work with InfluxDB down
                actual = {"_error": f"{type(error).__name__}: {str(error)[:200]}"}
        for role, declared in self.policy.buckets.items():
            entry = {
                "role": role,
                "content": declared.content or None,
                "bucket": declared.name,
                "retention_declared": format_duration(declared.retention_s),
            }
            if role in actual:
                entry.update(retention_actual=actual[role]["actual"], retention_status=actual[role]["status"])
            buckets.append(entry)
        description = {
            "influxdb": {
                "location": "client InfluxDB (stays with the client)",
                "url": _safe_url(location.url),
                "org": location.org,
                "token_env": location.token_env,
                "buckets": buckets,
            },
            "spool_dirs": [
                spool_summary(d) if check else {"path": str(d)} for d in self.policy.spool_dirs
            ],  # rows waiting to be written, on the capture host
            "outbox_dirs": [outbox_summary(d) if check else {"path": str(d)} for d in self.policy.outbox_dirs],
            "journal": {"path": str(self.policy.journal), "keyed": self.policy.journal_key_env is not None},
            "export_dir": str(self.policy.export_dir) if self.policy.export_dir else None,
            "leaves_the_client": {
                "what": "evidence records only (aggregated results, schema pulsar.evidence/1), sent by EvidenceOutbox",
                "destination": _safe_url(self.policy.central_service) if self.policy.central_service else None,
            },
        }
        if "_error" in actual:
            description["influxdb"]["error"] = actual["_error"]
        if check:
            try:
                description["journal"].update(head=self.journal.head(), entries=len(self.journal.entries()))
            except (ValueError, RuntimeError) as error:  # a torn last line: reported, see journal verify
                description["journal"]["error"] = f"{type(error).__name__}: {str(error)[:300]}"
        return description


def verify_export(directory: os.PathLike) -> dict:
    """Check the files of an export against its manifest: {"ok", "files", "errors"}"""
    directory = Path(directory)
    manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    errors = []
    for item in manifest["files"]:
        path = directory / item["path"]
        if not path.exists():
            errors.append(f"{item['path']}: missing")
        elif _sha256(path) != item["sha256"]:
            errors.append(f"{item['path']}: checksum differs")
    return {"ok": not errors, "files": len(manifest["files"]), "errors": errors}
