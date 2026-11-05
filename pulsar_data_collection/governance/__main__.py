"""Command line: python -m pulsar_data_collection.governance --policy policy.yaml <command> ...

Commands: describe, retention {check,apply}, count, delete, export, journal verify, export-verify. Output: JSON on
stdout. Exit code 0 when everything is as declared, 1 when a check fails (retention drift, broken journal, incomplete
deletion, altered export), 2 on usage errors.
"""

import argparse
import json
import sys
from typing import Optional

from .controller import DataController, verify_export
from .journal import AuditJournal
from .policy import load_policy
from .selection import DataSelector, Period


def _selector(args) -> DataSelector:
    tags = {}
    for item in args.tag or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--tag {item!r}: expected key=value")
        tags[key] = value
    return DataSelector(
        system_id=args.system_id,
        model_id=args.model_id,
        model_version=args.model_version,
        data_id=args.data_id,
        measurement=args.measurement,
        tags=tags,
        all_series=args.all_series,
    )


def _add_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--system-id")
    parser.add_argument("--model-id")
    parser.add_argument("--model-version")
    parser.add_argument("--data-id")
    parser.add_argument("--measurement")
    parser.add_argument("--tag", action="append", help="other tag, key=value (repeatable)")
    parser.add_argument("--all-series", action="store_true", help="every series of the period (no tag criterion)")
    parser.add_argument("--start", help="inclusive, ISO 8601 (default: the beginning)")
    parser.add_argument("--stop", help="inclusive, ISO 8601 (default: the end)")
    parser.add_argument("--role", action="append", help="bucket role of the policy (repeatable; default: all)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pulsar_data_collection.governance", description=__doc__.splitlines()[0])
    parser.add_argument("--policy", required=True, help="policy file (JSON, or YAML with PyYAML)")
    parser.add_argument("--actor", help="who acts, for the journal (default: the OS user)")
    commands = parser.add_subparsers(dest="command", required=True)

    describe = commands.add_parser("describe", help="where each kind of data lives")
    describe.add_argument("--offline", action="store_true", help="do not contact InfluxDB nor read the local stores")

    retention = commands.add_parser("retention", help="declared against actual retention")
    retention.add_argument("action", choices=["check", "apply"])
    retention.add_argument("--dry-run", action="store_true")
    retention.add_argument("--reason", help="why, for the journal (e.g. a ticket number)")

    count = commands.add_parser("count", help="count the selected data")
    _add_selection(count)

    delete = commands.add_parser("delete", help="delete the selected data (InfluxDB, spool, outbox)")
    _add_selection(delete)
    delete.add_argument("--dry-run", action="store_true", help="count what would be deleted, change nothing")
    delete.add_argument("--no-local", action="store_true", help="leave the spool and outbox stores untouched")
    delete.add_argument("--reason", help="why, for the journal (e.g. a ticket number)")

    export = commands.add_parser("export", help="export the selected data with a manifest of checksums")
    _add_selection(export)
    export.add_argument("--out", help="empty or new directory (default: a new directory in the policy's export_dir)")
    export.add_argument("--format", choices=["csv", "parquet"], default="csv")
    export.add_argument("--reason", help="why, for the journal")

    journal = commands.add_parser("journal", help="audit journal")
    journal.add_argument("action", choices=["verify"])
    journal.add_argument("--expected-head", help="a head recorded earlier, that the journal must still contain")

    check = commands.add_parser("export-verify", help="check an export against its manifest")
    check.add_argument("directory")
    return parser


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "journal":
        policy = load_policy(args.policy)
        result = AuditJournal(policy.journal, key_env=policy.journal_key_env).verify(expected_head=args.expected_head)
        return _print(result, result["ok"])
    if args.command == "export-verify":
        result = verify_export(args.directory)
        return _print(result, result["ok"])
    if args.command == "describe" and args.offline:
        with DataController(args.policy, actor=args.actor) as controller:  # configured locations only, no connection
            return _print(controller.describe(check=False), True)
    with DataController(args.policy, actor=args.actor) as controller:
        if args.command == "describe":
            return _print(controller.describe(), True)
        if args.command == "retention":
            if args.action == "check":
                rows = controller.check_retention()
            else:
                rows = controller.apply_retention(dry_run=args.dry_run, reason=args.reason)
            ok = all(r["status"] in ("ok", "unmanaged") for r in rows) or (args.action == "apply" and args.dry_run)
            return _print(rows, ok)
        try:
            selector = _selector(args)
        except ValueError as error:  # pydantic ValidationError included: no criterion, refused value
            print(f"invalid selection: {error}", file=sys.stderr)
            return 2
        try:
            Period(args.start, args.stop)
        except (ValueError, TypeError, OverflowError) as error:  # unreadable time, start after stop
            print(f"invalid period: {error}", file=sys.stderr)
            return 2
        if args.command == "count":
            return _print(controller.count(selector, args.start, args.stop, roles=args.role), True)
        if args.command == "delete":
            report = controller.delete(
                selector,
                args.start,
                args.stop,
                roles=args.role,
                dry_run=args.dry_run,
                local=not args.no_local,
                reason=args.reason,
            )
            return _print(report, report["status"] == "ok")
        manifest = controller.export(
            selector, args.out, args.start, args.stop, roles=args.role, format=args.format, reason=args.reason
        )
        return _print(manifest, True)


def _print(result, ok: bool) -> int:
    json.dump(result, sys.stdout, indent=2, ensure_ascii=False, default=str)
    sys.stdout.write("\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
