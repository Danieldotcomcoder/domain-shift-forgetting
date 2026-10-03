"""Two independent T4 workers with one elapsed-notebook budget and joint archives."""
import argparse
import math
from pathlib import Path
import sys
import time
import uuid

from .local_corpus import sha256
from .paired_control import CONDITIONS, exclusive_run, worker_environment, supervise, propagate_archive_indexes, assert_quiescent
from .pilot_control import (SessionClock, SessionLedger, load_policy, read_json, write_json, digest_json,
                            export_run, import_run, import_directory)


def policy_for_pair(path):
    policy = load_policy(path)
    if policy.get("execution", {}).get("mode") != "paired":
        raise ValueError("Use the paired policy; single-GPU preflights/freezes cannot be silently converted")
    return policy


def command(action, **kwargs):
    result = [sys.executable, "-m", "domain_shift_forgetting.paired", action]
    for key, value in kwargs.items():
        if value is not None:
            result += ["--" + key.replace("_", "-"), str(value)]
    return result


def jobs_for(action, *, root, policy, **kwargs):
    return {condition: {"command": command(action, condition=condition, root=root / "workers" / condition, **kwargs),
                        "env": worker_environment(policy["execution"]["workers"][condition]),
                        "status": root / "workers" / condition / "status.json"} for condition in CONDITIONS}


def phase_paths(root):
    phase = root / "coordinator" / uuid.uuid4().hex
    phase.mkdir(parents=True)
    return phase, phase / "STOP"


def validate_hours(hours):
    if not math.isfinite(hours) or hours < 0:
        raise ValueError("Cumulative external notebook hours must be finite and nonnegative")


def preflight(args):
    from .pilot_runner import source_identity
    started = time.monotonic()
    validate_hours(args.external_notebook_hours)
    policy = policy_for_pair(args.config)
    args.root.mkdir(parents=True, exist_ok=False)
    write_json(args.root / "paired-preflight.json", {"schema": 1, "policy": policy, "status": "running"})
    phase, stop_file = phase_paths(args.root)
    clock = SessionClock.start(args.remaining_minutes, 110, 10, now=started)
    jobs = jobs_for("_preflight", root=args.root, policy=policy, data=args.data, orders=args.orders,
                    config=args.config, tests=args.tests, shared=phase / "barriers", stop_file=stop_file,
                    remaining_minutes=max(0.1, (clock.deadline - time.monotonic()) / 60))
    try:
        with exclusive_run(args.root):
            supervise(jobs, phase, stop_file, deadline=clock.deadline - clock.reserve_seconds)
        validations = {c: read_json(args.root / "workers" / c / "validation.json") for c in CONDITIONS}
        benchmarks = {c: read_json(args.root / "workers" / c / "benchmark.json") for c in CONDITIONS}
        if any(v.get("status") != "passed" for v in validations.values()):
            raise ValueError("Both workers must pass validation")
        if len({v["arrays_sha256"] for v in validations.values()}) != 1:
            raise ValueError("Workers used different arrays")
        prior = args.external_notebook_hours + (time.monotonic() - started) / 3600
        totals = {c: benchmarks[c]["projection"]["total_worker_hours"] for c in CONDITIONS}
        # Barriers, imports, exports and differing stop points add overhead.
        projected = 1.15 * (prior + max(totals.values()) + 0.5)
        summary = {"schema": "paired-preflight-v1", "code": source_identity(), "policy": policy,
            "policy_sha256": digest_json(policy), "arrays_sha256": validations["RMS"]["arrays_sha256"],
            "runtimes": {c: validations[c]["runtime"] for c in CONDITIONS},
            "status": "passed" if projected <= policy["execution"]["notebook_hours_cap"] else "infeasible",
            "prior_notebook_hours": prior, "projected_worker_hours": totals,
            "projected_summed_worker_hours": sum(totals.values()),
            "projection_with_margin_notebook_hours": projected,
            "measured_worker_lifetimes": read_json(phase / "worker-lifetimes.json"),
            "restore": "Both externally persisted checkpoints must replay in a fresh session before seed 101"}
        write_json(args.root / "paired-preflight.json", summary)
        return summary
    except Exception as exc:
        write_json(args.root / "failure.json", {"phase": "paired preflight", "error": str(exc)})
        raise


def start(args):
    from .pilot_arrays import PilotArrays, load_orders
    from .pilot_runner import source_identity
    policy = policy_for_pair(args.config)
    validate_hours(args.external_notebook_hours)
    evidence = read_json(args.preflight / "paired-preflight.json")
    if evidence.get("status") != "passed" or evidence.get("code") != source_identity() or evidence.get("policy_sha256") != digest_json(policy):
        raise ValueError("Require passed paired preflight for exactly this code and policy")
    if args.external_notebook_hours < evidence["prior_notebook_hours"]:
        raise ValueError("Include all prior preflight/setup usage in cumulative external notebook hours")
    projection = evidence["projection_with_margin_notebook_hours"] + 1.15 * (args.external_notebook_hours - evidence["prior_notebook_hours"])
    if projection > policy["execution"]["notebook_hours_cap"]:
        raise ValueError("Updated cumulative usage no longer fits the notebook-hour budget")
    # Never reuse an existing primary root, including a single-GPU run.
    args.root.mkdir(parents=True, exist_ok=False)
    phase, stop_file = phase_paths(args.root)
    jobs = {c: {"command": command("_verify", root=args.preflight / "workers" / c,
                data=args.data, orders=args.orders, output=phase / f"{c}-restore.json"),
                "env": worker_environment(policy["execution"]["workers"][c])} for c in CONDITIONS}
    supervise(jobs, phase, stop_file, deadline=time.monotonic() + min(10, args.remaining_minutes - 10) * 60)
    proofs = {c: read_json(phase / f"{c}-restore.json") for c in CONDITIONS}
    for c in CONDITIONS:
        if (proofs[c].get("status") != "passed" or proofs[c]["runtime"] != evidence["runtimes"][c]
            or proofs[c]["code"] != source_identity() or proofs[c]["policy_sha256"] != digest_json(policy)
            or proofs[c]["arrays_sha256"] != evidence["arrays_sha256"]):
            raise ValueError("Fresh-session restore does not match paired preflight")
    arrays = PilotArrays(args.data)
    if arrays.identity != evidence["arrays_sha256"]:
        raise ValueError("Prepared data changed")
    orders = read_json(args.orders / "manifest.json")
    for seed in (101, 102, 103):
        load_orders(args.orders, arrays, seed)
    sources = read_json(args.sources / "source-evidence.json")
    if sources.get("status") != "captured" or any(sha256(args.sources / name) != row["sha256"] for name, row in sources["files"].items()):
        raise ValueError("Upstream evidence is missing or changed")
    frozen = {"schema": 1, "layout": "paired-v1", "admitted": True, "policy": policy, "code": source_identity(),
        "runtimes": evidence["runtimes"], "arrays": arrays.manifest, "arrays_sha256": arrays.identity,
        "orders": orders, "orders_sha256": sha256(args.orders / "manifest.json"), "upstream_sources": sources,
        "benchmark": {"prior_gpu_hours": args.external_notebook_hours, "budget_unit": "elapsed_notebook_hours",
                      "paired_preflight": evidence}, "external_resume": proofs,
        "run_order": [[seed, c] for seed in (101, 102, 103) for c in CONDITIONS],
        "prefix_gap_definition": "abs(Taper_prefix_web_CE - RMS_prefix_web_CE) / RMS_prefix_web_CE",
        "quick_subset": "first 512 of frozen 4096 full-development windows", "created_unix": time.time()}
    write_json(args.root / "freeze.json", frozen)
    for c in CONDITIONS:
        write_json(args.root / "workers" / c / "freeze.json", frozen)
    return {"status": "admitted", "external_resume": {c: proofs[c]["status"] for c in CONDITIONS}}


def run(args):
    from .pilot_runner import source_identity
    from .pilot_report import report
    began = time.monotonic()
    validate_hours(args.external_notebook_hours)
    frozen = read_json(args.root / "freeze.json")
    if frozen.get("layout") != "paired-v1" or frozen.get("admitted") is not True or frozen["code"] != source_identity():
        raise ValueError("Require unchanged admitted paired run")
    if (args.root / "failure.json").exists() or (args.root / "IMPORT-FAILED.json").exists():
        raise ValueError("Investigate failed run/import before any restart")
    if args.external_notebook_hours < frozen["benchmark"]["prior_gpu_hours"]:
        raise ValueError("Cumulative pre-run usage cannot decrease")
    policy = frozen["policy"]
    with exclusive_run(args.root):
        clock = SessionClock.start(args.remaining_minutes, policy["session_chunk_minutes"], policy["session_save_reserve_minutes"], now=began)
        ledger = SessionLedger(args.root / "notebook-ledger.json", reserved_seconds=clock.deadline - began,
            external_gpu_hours=args.external_notebook_hours, cap_hours=policy["execution"]["notebook_hours_cap"],
            recover_unclean=args.recover_unclean)
        ledger.started = began
        ledger.data["budget_unit"] = "elapsed_notebook_hours"
        ledger.row["summed_worker_seconds"] = 2 * ledger.reserved
        write_json(ledger.path, ledger.data)
        clock.deadline = min(clock.deadline, began + ledger.reserved)
        phase, stop_file = phase_paths(args.root)
        jobs = jobs_for("_run", root=args.root, policy=policy, data=args.data, orders=args.orders,
            stop_file=stop_file, remaining_minutes=max(.1, (clock.deadline - time.monotonic()) / 60),
            external_notebook_hours=args.external_notebook_hours, recover_unclean=int(args.recover_unclean))
        status, optimizer_seconds = "failed", 0.0
        try:
            supervise(jobs, phase, stop_file, deadline=clock.deadline - clock.reserve_seconds, stop_on_pause=True)
            states = {c: read_json(args.root / "workers" / c / "status.json") for c in CONDITIONS}
            status = "completed" if all(s["status"] == "completed" for s in states.values()) else "paused"
            lifetimes = read_json(phase / "worker-lifetimes.json")
            ledger.row["summed_worker_seconds"] = sum(v["elapsed_seconds"] for v in lifetimes.values())
            for c in CONDITIONS:
                child_ledger = read_json(args.root / "workers" / c / "session-ledger.json")
                optimizer_seconds += child_ledger["sessions"][-1]["optimizer_seconds"]
            write_json(args.root / "status.json", {"status": status, "workers": states,
                "next": "Persist full paired archive; next session uses MODE=resume"})
            return report(args.root)
        except Exception as exc:
            write_json(args.root / "failure.json", {"error": str(exc), "action": "Both workers stopped. Preserve evidence; do not replace seeds."})
            raise
        finally:
            if status == "failed":
                ledger.row["summed_worker_seconds"] = 2 * ledger.reserved
            ledger.finish(optimizer_seconds, status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "start", "run", "report", "import", "export", "_preflight", "_verify", "_run"))
    for name in ("root", "data", "orders", "config", "tests", "sources", "preflight", "output", "archive", "shared", "stop-file"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--remaining-minutes", type=float)
    parser.add_argument("--external-notebook-hours", type=float, default=0)
    parser.add_argument("--recover-unclean", type=int, choices=(0, 1), default=0)
    parser.add_argument("--reports-only", action="store_true")
    args = parser.parse_args()
    if args.action == "_preflight":
        from .paired_preflight import worker
        result = worker(args.data, args.orders, args.root, policy_for_pair(args.config), args.condition,
                        args.tests, args.shared, args.stop_file, args.remaining_minutes)
    elif args.action == "_verify":
        from .pilot_preflight import verify_external_resume
        result = verify_external_resume(args.root, args.data, args.orders, args.output)
    elif args.action == "_run":
        from .pilot_runner import run as worker_run
        result = worker_run(args.root, args.data, args.orders, remaining_minutes=args.remaining_minutes,
            external_gpu_hours=args.external_notebook_hours, recover_unclean=bool(args.recover_unclean),
            worker_condition=args.condition, stop_file=args.stop_file)
    elif args.action in ("preflight", "start", "run"):
        result = globals()[args.action](args)
    elif args.action == "report":
        from .pilot_report import report
        result = report(args.root)
    elif args.action == "import":
        if args.archive.is_dir():
            import_directory(args.archive, args.root)
        else:
            import_run(args.archive, args.root)
        propagate_archive_indexes(args.root)
        result = {"status": "paired archive verified and restored"}
    else:
        with exclusive_run(args.root):
            if not args.reports_only:
                assert_quiescent(args.root)
            result = export_run(args.root, args.output, reports_only=args.reports_only)
    import json
    if result is not None:
        if "decision" in result:
            result = {"decision": result["decision"], "completed_seed_groups": result["completed_seed_groups"]}
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
