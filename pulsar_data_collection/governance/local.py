"""Deletion in the local stores: spool / write-ahead directories of Pulse, and EvidenceOutbox SQLite stores.

Run them with the capture of the selected system stopped (Pulse closed): a spool directory belongs to one Pulse at a
time, and rows still in a running Pulse's memory would be written after the purge.

Spool invariants kept (see spool.py): a batch keeps its time prefix and id, so the replay order does not change; its
name gives its new row count; a partly selected batch is rewritten through a temporary `.purge` file renamed onto its
new name before the old file is deleted, so that a crash never leaves a lost or half-written batch. A crash before the
rename leaves the old batch and a `.purge` file, deleted by the next purge (purge to run again). A crash after the
rename and before the deletion leaves both batches under the same time prefix and id: the next purge, and the next
start of the spool (`spool.drop_superseded`), delete the one with more rows, so that its deleted rows are not
replayed.

Outbox: pending and quarantined records are deleted in one transaction. A running outbox copes with rows that
disappear (a send of a deleted record deletes or quarantines nothing), but the records it had already read for a send
in progress are still delivered: close the outbox (or let it drain) before purging to be sure none leaves.
"""

import gzip
import json
import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

from ..evidence import DATABASE_NAME
from ..spool import drop_superseded, frame_from_json, frame_to_json
from .selection import DataSelector, Period

logger = logging.getLogger("pulsar_data_collection")

# every file a spool directory may hold: batches, write-ahead copies, set-aside batches, temporary files
_BATCH = re.compile(
    r"^(?P<stamp>\d+)_(?P<id>[0-9a-f]+)_(?P<rows>\d+)r(?P<suffix>\.json(\.gz)?|\.wal(\.gz)?|\.tmp)(?P<bad>\.bad)?$"
)
_SELECT = {"outbox": "SELECT seq, body FROM outbox", "quarantine": "SELECT seq, body FROM quarantine"}
_ANY = re.compile(r"\.(json|wal)(\.gz)?(\.bad)?$|\.tmp$")


def _read(path: Path) -> tuple[pd.DataFrame, dict]:
    data = path.read_bytes()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return frame_from_json(data.decode("utf-8"))


def _times_ns(frame: pd.DataFrame, meta: dict) -> Optional[pd.Series]:
    """Row timestamps as int64 UTC nanoseconds (naive times are UTC); None when the batch has none"""
    for column in (meta.get("data_frame_timestamp_column"), "_time"):
        if column and column in frame and pd.api.types.is_datetime64_any_dtype(frame[column]):
            values = frame[column]
            if values.dt.tz is None:
                values = values.dt.tz_localize("UTC")
            return values.dt.tz_convert("UTC").dt.as_unit("ns").astype("int64")
    return None


def purge_spool(directory: os.PathLike, selector: DataSelector, period: Optional[Period] = None, dry_run: bool = False) -> dict:
    """Delete the selected rows of the batches of a spool directory

    Returns {"directory", "exists", "files_deleted", "files_rewritten", "rows", "unreadable"}: `rows` deleted (or that
    would be, with dry_run), `unreadable` files that could not be read nor classified, and selected batches that
    could not be purged (no timestamp column to apply a period to): kept, listed by name, and possibly replayed into
    InfluxDB by their Pulse (DataController.delete then reports "incomplete").
    """
    directory = Path(directory)
    report = {"directory": str(directory), "exists": directory.is_dir(), "files_deleted": 0, "files_rewritten": 0, "rows": 0}
    report["unreadable"] = []
    if not report["exists"]:
        return report
    period = period or Period()
    for leftover in sorted(directory.glob("*.purge")):
        if not dry_run:
            leftover.unlink(missing_ok=True)
    superseded = set(drop_superseded(directory, dry_run=dry_run))  # left by an interrupted rewrite
    for path in sorted(p for p in directory.iterdir() if p.is_file() and _ANY.search(p.name) and p.name not in superseded):
        try:
            frame, meta = _read(path)
        except Exception:  # noqa: BLE001 - incomplete .tmp or damaged batch: reported, never guessed
            report["unreadable"].append(path.name)
            continue
        if not selector.matches(meta.get("tags") or {}, meta.get("data_frame_measurement_name")):
            continue
        if period.bounded:
            stamps = _times_ns(frame, meta)
            if stamps is None:
                report["unreadable"].append(path.name)
                continue
            selected = period.contains_ns(stamps).to_numpy()
        else:
            selected = None
        count = len(frame) if selected is None else int(selected.sum())
        if count == 0:
            continue
        report["rows"] += count
        if selected is None or count == len(frame):
            report["files_deleted"] += 1
            if not dry_run:
                path.unlink(missing_ok=True)
            continue
        report["files_rewritten"] += 1
        if not dry_run:
            _rewrite(path, frame.loc[~selected].reset_index(drop=True), meta)
    if not dry_run and (report["files_deleted"] or report["files_rewritten"]):
        logger.warning(
            "Spool %s: %d rows deleted (%d files deleted, %d rewritten)",
            directory,
            report["rows"],
            report["files_deleted"],
            report["files_rewritten"],
        )
    return report


def _rewrite(path: Path, kept: pd.DataFrame, meta: dict) -> None:
    """Replace a batch by its kept rows, same time prefix and id, new row count; old file deleted last"""
    match = _BATCH.match(path.name)
    if match:
        name = f"{match['stamp']}_{match['id']}_{len(kept)}r{match['suffix']}{match['bad'] or ''}"
    else:  # name of an older version: same name, the row count is read from the content
        name = path.name
    data = frame_to_json(kept, meta).encode("utf-8")
    if ".gz" in path.name:
        data = gzip.compress(data, compresslevel=1)
    target = path.with_name(name)
    temporary = path.with_name(name + ".purge")
    temporary.write_bytes(data)
    temporary.replace(target)
    if target != path:
        path.unlink(missing_ok=True)


def _outbox_rows(db, table: str, selector: DataSelector, period: Period) -> list[int]:
    seqs = []
    for seq, body in db.execute(_SELECT[table]).fetchall():
        try:
            record = json.loads(body)
        except ValueError:
            continue
        if not selector.matches_evidence(record):
            continue
        if period.bounded:
            window = record.get("window") or {}
            if not (window.get("start") or window.get("end")):
                continue
            if not period.overlaps(window.get("start"), window.get("end")):
                continue
        seqs.append(seq)
    return seqs


def purge_outbox(
    directory: os.PathLike,
    selector: DataSelector,
    period: Optional[Period] = None,
    dry_run: bool = False,
    include_quarantine: bool = True,
) -> dict:
    """Delete the selected evidence records not yet sent (and quarantined ones) from an EvidenceOutbox store

    Records are selected on their `system_id`, `model_id` and `model_version`, and with a period on their `window`
    (intersecting the period). Selectors with other criteria (data_id, measurement, tags, all_series) do not apply to
    evidence records: the store is left untouched and `skipped` says why. Records already delivered are in the
    central service, out of reach of this function; so are those a running outbox had already read for a send in
    progress (they are delivered after the purge): close the outbox first.
    """
    path = Path(directory) / DATABASE_NAME
    report = {"directory": str(directory), "exists": path.exists(), "pending": 0, "quarantined": 0, "skipped": None}
    if not report["exists"]:
        return report
    if not selector.evidence_applicable():
        report["skipped"] = "evidence records carry system_id, model_id and model_version only"
        return report
    period = period or Period()
    # mode=rw: never creates a store; the path comes from the policy, not from data
    db = sqlite3.connect(f"{path.resolve().as_uri()}?mode=rw", uri=True, timeout=30, isolation_level=None)
    try:
        db.execute("BEGIN IMMEDIATE")
        try:
            pending = _outbox_rows(db, "outbox", selector, period)
            quarantined = _outbox_rows(db, "quarantine", selector, period) if include_quarantine else []
            if not dry_run:
                db.executemany("DELETE FROM outbox WHERE seq = ?", [(s,) for s in pending])
                db.executemany("DELETE FROM quarantine WHERE seq = ?", [(s,) for s in quarantined])
            db.execute("ROLLBACK" if dry_run else "COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise
    finally:
        db.close()
    report.update(pending=len(pending), quarantined=len(quarantined))
    if not dry_run and (pending or quarantined):
        logger.warning(
            "Evidence outbox %s: %d pending and %d quarantined records deleted", directory, len(pending), len(quarantined)
        )
    return report


def spool_summary(directory: os.PathLike) -> dict:
    """Files and bytes of a spool directory (for describe)"""
    directory = Path(directory)
    if not directory.is_dir():
        return {"path": str(directory), "exists": False}
    files = [p for p in directory.iterdir() if p.is_file() and _ANY.search(p.name)]
    return {"path": str(directory), "exists": True, "files": len(files), "bytes": sum(p.stat().st_size for p in files)}


def outbox_summary(directory: os.PathLike) -> dict:
    """Pending and quarantined records of an outbox store (for describe)"""
    path = Path(directory) / DATABASE_NAME
    if not path.exists():
        return {"path": str(path), "exists": False}
    db = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=30)
    try:
        pending = db.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
        quarantined = db.execute("SELECT COUNT(*) FROM quarantine").fetchone()[0]
    finally:
        db.close()
    return {"path": str(path), "exists": True, "pending": pending, "quarantined": quarantined}
