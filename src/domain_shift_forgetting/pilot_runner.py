"""Original paired experiment, executed only from a provenance-bound admission freeze."""
from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import time

import numpy as np
import torch
from torch.nn import functional as F

from .local_corpus import sha256
from .local_training import save, restore
from .models.transformer import Transformer, copy_canonical_initialization
from .pilot_arrays import PilotArrays, load_orders, evaluate
from .pilot_control import SessionClock, SessionLedger, digest_json, read_json, write_json
from .pilot_diagnostics import diagnostics
from .pilot_report import event_path, report
from .protocol import (Condition, CALIBRATION_END, PREFIX_END, CONTINUATION_UPDATES, FULL_PREFIX,
                       FULL_CONTINUATION, QUICK_CONTINUATION, DIAGNOSTIC_CONTINUATION, learning_rate, gate)
from .training.step import make_optimizer


def source_identity():
    root = Path(__file__).parent
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob("*.py"))}


def runtime_identity(device):
    import platform
    import subprocess
    driver = None
    if device.type == "cuda":
        driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).splitlines()[0].strip()
    return {"torch": str(torch.__version__), "numpy": np.__version__, "python": platform.python_version(),
            "cuda": torch.version.cuda, "driver": driver, "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "visible_cuda_devices": torch.cuda.device_count() if device.type == "cuda" else 0,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def make_state(seed, condition, policy, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    canonical = Transformer(Condition.RMS)
    model = canonical if condition == "RMS" else Transformer(Condition(condition))
    if model is not canonical:
        copy_canonical_initialization(canonical, model)
        del canonical
    model.to(device)
    base = make_optimizer(model)
    groups = [{"params": g["params"], "weight_decay": g["weight_decay"]} for g in base.param_groups]
    optimizer = torch.optim.AdamW(groups, lr=0.0006, betas=(0.9, 0.95), eps=1e-8,
                                 fused=device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=policy["precision"] == "fp16", init_scale=128)
    sampler = torch.Generator().manual_seed(seed + 1000)  # Saved even though orders are precomputed.
    return model, optimizer, scaler, sampler


def ordered_update(model, optimizer, scaler, stream, indices, policy, device):
    if len(indices) != 32 or policy["microbatch"] * policy["accumulation"] != 32:
        raise ValueError("Exactly 16,384 labels required per completed update")
    u = int(model.completed_updates) + 1
    for group in optimizer.param_groups:
        group["lr"] = learning_rate(u)
    model.train()
    for attempt in range(6):
        optimizer.zero_grad(set_to_none=True)
        for name, buffer in model.named_buffers():
            if "pending_" in name:
                buffer.zero_()
        ce_total = torch.zeros((), device=device)
        for start in range(0, 32, policy["microbatch"]):
            rows = torch.from_numpy(stream.take(indices[start:start + policy["microbatch"]])).to(device)
            with torch.autocast(device.type, dtype=torch.float16 if policy["precision"] == "fp16" else torch.bfloat16,
                                enabled=policy["precision"] != "fp32"):
                logits, _ = model(rows[:, :-1], update=u, collect=u <= CALIBRATION_END)
                ce = F.cross_entropy(logits.float().reshape(-1, 50257), rows[:, 1:].reshape(-1))
            if not bool(torch.isfinite(ce)):
                raise FloatingPointError("Nonfinite forward CE; never retry as a seed replacement")
            scaler.scale(ce / policy["accumulation"]).backward()
            ce_total += ce.detach() / policy["accumulation"]
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
        if not bool(torch.isfinite(norm)):
            if not scaler.is_enabled():
                raise FloatingPointError("Nonfinite gradients")
            scaler.update(new_scale=scaler.get_scale() / 2)
            continue  # Identical fixed indices; no cursor or calibration clock advanced.
        scaler.step(optimizer)
        scaler.update()
        if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise FloatingPointError("Nonfinite weights after optimizer step")
        model.finish_update(u)
        return {"ce": float(ce_total), "gradient_norm": float(norm), "clipped": float(norm) > 1,
                "overflow_retries": attempt, "lr": learning_rate(u), "gate": gate(u), "global_update": u}
    raise FloatingPointError("FP16 overflow persisted; preserve failure and last committed state")


def freeze_run(output, policy, arrays, order_root, validation, benchmark, device, *, source_evidence, resume_validation):
    """All empirical admission evidence is produced on the intended host first."""
    if output.exists():
        raise FileExistsError("Freeze into a new run directory")
    code, runtime = source_identity(), runtime_identity(device)
    for name, evidence in (("validation", validation), ("benchmark", benchmark), ("external resume", resume_validation)):
        if evidence.get("status") != "passed" or evidence.get("code") != code or evidence.get("runtime") != runtime:
            raise ValueError(f"Missing/stale/wrong-host {name} evidence")
        if evidence.get("arrays_sha256") != arrays.identity or evidence.get("policy_sha256") != digest_json(policy):
            raise ValueError(f"Mismatched {name} data/configuration")
    if benchmark["projection_with_margin_hours"] > policy["active_gpu_hours_cap"]:
        raise ValueError("Measured allocation does not fit resource cap; do not shorten scientific controls")
    orders = read_json(order_root / "manifest.json")
    if orders["array_manifest_sha256"] != arrays.identity:
        raise ValueError("Order manifest mismatch")
    for seed in (101, 102, 103):
        load_orders(order_root, arrays, seed)
    output.mkdir(parents=True)
    frozen = {"schema": 1, "admitted": True, "policy": policy, "arrays": arrays.manifest,
              "arrays_sha256": arrays.identity, "orders": orders, "orders_sha256": sha256(order_root / "manifest.json"),
              "code": code, "runtime": runtime, "validation": validation, "benchmark": benchmark,
              "external_resume": resume_validation, "upstream_sources": source_evidence,
              "prefix_gap_definition": "abs(Taper_prefix_web_CE - RMS_prefix_web_CE) / RMS_prefix_web_CE",
              "quick_subset": "first 512 of the frozen 4096 full-development windows",
              "run_order": [[seed, condition] for seed in (101, 102, 103) for condition in ("RMS", "Taper-minus")],
              "created_unix": time.time()}
    write_json(output / "freeze.json", frozen)
    return frozen


def run(root: Path, data_root: Path, order_root: Path, *, remaining_minutes: float,
        external_gpu_hours: float, recover_unclean=False, max_updates=None, worker_condition=None, stop_file=None):
    invocation_started = time.monotonic()
    if (root / "IMPORT-FAILED.json").exists() or (root / "failure.json").exists():
        raise ValueError("Failed import/numerical attempt requires investigation; do not hide it by resuming")
    frozen = read_json(root / "freeze.json")
    if frozen.get("admitted") is not True:
        raise ValueError("Missing validated admission freeze")
    policy = frozen["policy"]
    device = torch.device("cuda")
    paired = frozen.get("layout") == "paired-v1"
    if paired != (worker_condition in ("RMS", "Taper-minus")):
        raise ValueError("Paired roots require exactly one assigned condition worker")
    expected_runtime = frozen["runtimes"][worker_condition] if paired else frozen["runtime"]
    if frozen["code"] != source_identity() or expected_runtime != runtime_identity(device):
        raise ValueError("Code/runtime/hardware changed; retain old stratum and do not resume silently")
    torch.set_num_threads(2 if paired else 4)
    torch.use_deterministic_algorithms(True)
    arrays = PilotArrays(data_root)
    if arrays.identity != frozen["arrays_sha256"] or sha256(order_root / "manifest.json") != frozen["orders_sha256"]:
        raise ValueError("Frozen data/order identity changed")
    # Account for validation/setup measured before freeze in the same overall cap.
    minimum_external = frozen["benchmark"]["prior_gpu_hours"]
    if external_gpu_hours < minimum_external:
        raise ValueError("External GPU usage must include all pre-freeze validation/benchmark activity")
    clock = SessionClock.start(remaining_minutes, policy["session_chunk_minutes"], policy["session_save_reserve_minutes"], now=invocation_started)
    ledger = SessionLedger(root / "session-ledger.json", reserved_seconds=clock.deadline - invocation_started,
                           external_gpu_hours=external_gpu_hours, cap_hours=policy["active_gpu_hours_cap"],
                           recover_unclean=recover_unclean)
    ledger.started = invocation_started
    clock.deadline = min(clock.deadline, invocation_started + ledger.reserved)
    optimizer_seconds, invocation_updates = 0.0, 0
    freeze_hash = digest_json(frozen)
    stop_requested = [False]
    prior_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        prior_handlers[sig] = signal.signal(sig, lambda *_: stop_requested.__setitem__(0, True))
    status = "paused"
    def summarize():
        return read_json(root / "status.json") if paired else report(root)

    try:
        for seed, condition in frozen["run_order"]:
            if paired and condition != worker_condition:
                continue
            run_dir = root / "runs" / f"S{seed}-{condition}"
            run_dir.mkdir(parents=True, exist_ok=True)
            identity = {"freeze_sha256": freeze_hash, "seed": seed, "condition": condition}
            orders = load_orders(order_root, arrays, seed)
            model, optimizer, scaler, sampler = make_state(seed, condition, policy, device)
            progress = {"stage": "prefix", "step": 0, "optimizer_seconds": 0.0, "updates": 0, "clipped": 0,
                        "overflow_retries": 0, "parent_sha256": None}
            if (run_dir / "latest.pt.json").exists():
                progress = restore(run_dir / "latest.pt", model, optimizer, scaler, sampler, identity)
            def completed_receipt():
                write_json(run_dir / "completion.json", {"freeze_sha256": freeze_hash, "seed": seed,
                    "condition": condition, "latest_sha256": read_json(run_dir / "latest.pt.json")["sha256"],
                    "prefix_sha256": read_json(run_dir / "switch.pt.json")["sha256"]})
            if progress["stage"] == "complete":
                completed_receipt()
                del model, optimizer, scaler
                gc.collect()
                torch.cuda.empty_cache()
                continue
            last_save = time.monotonic()

            def checkpoint(name="latest.pt"):
                nonlocal last_save
                save(run_dir / name, model, optimizer, scaler, sampler, progress, identity)
                ledger.record_optimizer(optimizer_seconds)
                last_save = time.monotonic()

            while progress["stage"] != "complete":
                stage, step = progress["stage"], progress["step"]
                limit = PREFIX_END if stage == "prefix" else CONTINUATION_UPDATES
                full = step in (FULL_PREFIX if stage == "prefix" else FULL_CONTINUATION)
                quick = step == PREFIX_END if stage == "prefix" else step in QUICK_CONTINUATION
                diag = step == PREFIX_END if stage == "prefix" else step in DIAGNOSTIC_CONTINUATION
                if (clock.must_stop(30) or stop_requested[0] or (stop_file is not None and stop_file.exists()) or shutil.disk_usage(root).free < 4 * 1024**3
                    or ledger.previous_optimizer_seconds + optimizer_seconds >= policy["optimizer_hours_cap"] * 3600
                    or (max_updates is not None and invocation_updates >= max_updates)):
                    checkpoint()
                    write_json(root / "status.json", {"status": "paused", "run": run_dir.name, "stage": stage, "step": step,
                        "reason": "deadline/signal/storage/update/budget boundary", "next": "export, persist and resume same state"})
                    return summarize()
                if full or quick or diag:
                    # Commit this exact event state before any expensive evaluation.
                    checkpoint()
                    checkpoint_sha = read_json(run_dir / "latest.pt.json")["sha256"]
                    if full:
                        weight_path = run_dir / f"weights-{stage}-{step}.pt"
                        archive_index = read_json(root / "archived-weights.json") if (root / "archived-weights.json").exists() else {}
                        if not weight_path.exists() and weight_path.relative_to(root).as_posix() not in archive_index:
                            temp = weight_path.with_suffix(".tmp")
                            torch.save({"model": model.state_dict(), "identity": identity, "global_update": int(model.completed_updates)}, temp)
                            temp.replace(weight_path)
                    for role in (["full"] if full else []) + (["quick"] if quick else []):
                        for domain in ("web", "python"):
                            path = event_path(root, seed, condition, stage, step, role, domain)
                            if not path.exists():
                                measured = evaluate(model, arrays, domain, role=role, rare=orders["rare"],
                                    microbatch=policy["microbatch"], precision=policy["precision"], clock=clock)
                                write_json(path, {"freeze_sha256": freeze_hash, "seed": seed, "condition": condition,
                                    "stage": stage, "step": step, "global_update": int(model.completed_updates),
                                    "attempt_id": ledger.row["id"], "checkpoint_sha256": checkpoint_sha,
                                    "parent_sha256": progress["parent_sha256"], "measurement": measured})
                    diag_path = root / "diagnostics" / run_dir.name / f"{stage}-{step}.json"
                    if diag and not diag_path.exists():
                        measured = diagnostics(model, arrays, orders["rare"], microbatch=policy["microbatch"],
                                               precision=policy["precision"], clock=clock)
                        write_json(diag_path, {"freeze_sha256": freeze_hash, "seed": seed, "condition": condition,
                                             "stage": stage, "step": step, "measurement": measured})
                if step == limit:
                    if stage == "prefix":
                        if not (run_dir / "switch.pt.json").exists():
                            checkpoint("switch.pt")
                        parent_sha = read_json(run_dir / "switch.pt.json")["sha256"]
                        durable = read_json(root / "durable-prefixes.json") if (root / "durable-prefixes.json").exists() else {"prefixes": {}}
                        if durable["prefixes"].get(str(run_dir.relative_to(root))) != parent_sha:
                            checkpoint()
                            write_json(root / "status.json", {"status": "awaiting_durable_prefix", "run": run_dir.name,
                                "next": "Export run; save outside session; import the verified archive in the next session."})
                            return summarize()
                    else:
                        if not (run_dir / f"{stage}-final.pt.json").exists():
                            checkpoint(f"{stage}-final.pt")
                    if stage == "python":
                        progress["stage"] = "complete"
                        checkpoint()
                        completed_receipt()
                        break
                    cumulative = {key: progress[key] for key in ("optimizer_seconds", "updates", "clipped", "overflow_retries")}
                    restore(run_dir / "switch.pt", model, optimizer, scaler, sampler, identity)
                    progress = {**progress, **cumulative, "stage": "web" if stage == "prefix" else "python", "step": 0,
                                "parent_sha256": read_json(run_dir / "switch.pt.json")["sha256"]}
                    checkpoint()
                    continue
                offset = (PREFIX_END + step) * 32 if stage == "web" else step * 32
                domain = "python" if stage == "python" else "web"
                indices = orders[domain][offset:offset + 32].astype(np.int64)
                torch.cuda.synchronize()
                tick = time.monotonic()
                metrics = ordered_update(model, optimizer, scaler, arrays.streams[f"{domain}_train"], indices, policy, device)
                torch.cuda.synchronize()
                elapsed = time.monotonic() - tick
                optimizer_seconds += elapsed
                progress["optimizer_seconds"] += elapsed
                progress["step"] += 1
                progress["updates"] += 1
                progress["clipped"] += int(metrics["clipped"])
                progress["overflow_retries"] += metrics["overflow_retries"]
                invocation_updates += 1
                with (run_dir / "training.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"attempt_id": ledger.row["id"], "stage": stage, "step": progress["step"], **metrics, "seconds": elapsed}) + "\n")
                if time.monotonic() - last_save >= policy["checkpoint_interval_seconds"]:
                    checkpoint()
                if progress["step"] % 50 == 0:
                    current = {"status": "running", "run": run_dir.name, "stage": stage, "step": progress["step"],
                               "tokens_per_second": 16384 / elapsed, "train_ce": metrics["ce"],
                               "clipping_fraction": progress["clipped"] / progress["updates"],
                               "overflow_retries": progress["overflow_retries"]}
                    write_json(root / "status.json", current)
                    print(json.dumps(current), flush=True)
            del model, optimizer, scaler
            gc.collect()
            torch.cuda.empty_cache()
            summarize()
        status = "completed"
        write_json(root / "status.json", {"status": "completed"})
        return summarize()
    except TimeoutError:
        # An event checkpoint was committed before beginning evaluation.
        write_json(root / "status.json", {"status": "paused", "reason": "evaluation/session deadline"})
        return summarize()
    except Exception as exc:
        status = "failed"
        write_json(root / "failure.json", {"type": type(exc).__name__, "message": str(exc),
                   "action": "Investigate; no seed replacement or threshold change. Retain all evidence."})
        raise
    finally:
        ledger.finish(optimizer_seconds, status)
        for sig, handler in prior_handlers.items():
            signal.signal(sig, handler)
