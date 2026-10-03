"""Prepare, admit, execute and report the original H1 pilot in bounded sessions."""
import argparse
import json
from pathlib import Path

from .pilot_control import load_policy, read_json, write_json, export_run, import_run, import_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sources = sub.add_parser("sources")
    sources.add_argument("--output", type=Path, required=True)
    orders = sub.add_parser("orders")
    orders.add_argument("--data-dir", type=Path, required=True)
    orders.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("preflight")
    freeze = sub.add_parser("freeze")
    for p in (check, freeze):
        p.add_argument("--data-dir", type=Path, required=True)
        p.add_argument("--orders", type=Path, required=True)
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
    check.add_argument("--prior-gpu-hours", type=float, required=True)
    check.add_argument("--session-remaining-minutes", type=float, required=True)
    check.add_argument("--tests", type=Path, required=True)
    freeze.add_argument("--preflight", type=Path, required=True)
    freeze.add_argument("--sources", type=Path, required=True)
    freeze.add_argument("--resume-validation", type=Path, required=True)
    proof = sub.add_parser("verify-resume")
    proof.add_argument("--preflight", type=Path, required=True)
    proof.add_argument("--data-dir", type=Path, required=True)
    proof.add_argument("--orders", type=Path, required=True)
    proof.add_argument("--output", type=Path, required=True)
    start = sub.add_parser("run")
    start.add_argument("--root", type=Path, required=True)
    start.add_argument("--data-dir", type=Path, required=True)
    start.add_argument("--orders", type=Path, required=True)
    start.add_argument("--session-remaining-minutes", type=float, required=True)
    start.add_argument("--external-gpu-hours", type=float, required=True)
    start.add_argument("--max-updates", type=int)
    start.add_argument("--recover-unclean", action="store_true")
    inspect = sub.add_parser("report")
    inspect.add_argument("--root", type=Path, required=True)
    export = sub.add_parser("export")
    export.add_argument("--root", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--reports-only", action="store_true")
    restore = sub.add_parser("import")
    restore.add_argument("--archive", type=Path, required=True)
    restore.add_argument("--output", type=Path, required=True)
    directory = sub.add_parser("import-directory")
    directory.add_argument("--source", type=Path, required=True)
    directory.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "sources":
        from .pilot_sources import capture
        result = capture(args.output)
    elif args.command == "orders":
        from .pilot_arrays import PilotArrays
        result = PilotArrays(args.data_dir).make_orders(args.output)
    elif args.command == "preflight":
        from .pilot_preflight import preflight
        result = preflight(args.data_dir, args.orders, args.output, load_policy(args.config),
            prior_gpu_hours=args.prior_gpu_hours, remaining_minutes=args.session_remaining_minutes, tests_root=args.tests)
    elif args.command == "freeze":
        import torch
        from .local_corpus import sha256
        from .pilot_arrays import PilotArrays
        from .pilot_runner import freeze_run
        evidence = read_json(args.sources / "source-evidence.json")
        if evidence.get("status") != "captured" or any(sha256(args.sources / name) != row["sha256"] for name, row in evidence["files"].items()):
            raise ValueError("Upstream source evidence mismatch")
        result = freeze_run(args.output, load_policy(args.config), PilotArrays(args.data_dir), args.orders,
            read_json(args.preflight / "validation.json"), read_json(args.preflight / "benchmark.json"), torch.device("cuda"),
            source_evidence=evidence, resume_validation=read_json(args.resume_validation))
    elif args.command == "verify-resume":
        from .pilot_preflight import verify_external_resume
        result = verify_external_resume(args.preflight, args.data_dir, args.orders, args.output)
    elif args.command == "run":
        from .pilot_runner import run
        if args.max_updates is not None and args.max_updates < 1:
            raise ValueError("max-updates must be positive")
        result = run(args.root, args.data_dir, args.orders, remaining_minutes=args.session_remaining_minutes,
            external_gpu_hours=args.external_gpu_hours, recover_unclean=args.recover_unclean, max_updates=args.max_updates)
    elif args.command == "report":
        from .pilot_report import report
        result = report(args.root)
    elif args.command == "export":
        result = export_run(args.root, args.output, reports_only=args.reports_only)
        write_json(args.output.with_suffix(".receipt.json"), result)
    elif args.command == "import-directory":
        import_directory(args.source, args.output)
        result = {"status": "restored_and_verified", "root": str(args.output)}
    else:
        import_run(args.archive, args.output)
        result = {"status": "restored_and_verified", "root": str(args.output)}
    # Keep notebook output compact; complete records are on disk.
    if isinstance(result, dict) and "decision" in result:
        print(json.dumps({"decision": result["decision"], "completed_seed_groups": result["completed_seed_groups"],
                          "missing_event_count": result["missing_event_count"]}, indent=2))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
