"""Real-array correctness and phase timing before scientific seed 101.

Seed 997 is a disposable engineering fixture; these observations are not primary
results. A calibrated state is used for phase timing, with explicitly repositioned
clocks. Only real scientific trajectories can establish long-run stability.
"""
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from .kaggle import snapshot, compare
from .local_training import save, restore
from .local_corpus import sha256
from .pilot_arrays import PilotArrays, load_orders, evaluate
from .pilot_control import SessionClock, digest_json, read_json, write_json
from .pilot_diagnostics import diagnostics
from .pilot_runner import make_state, ordered_update, runtime_identity, source_identity
from .protocol import CALIBRATION_END, GATE_END, FULL_PREFIX, FULL_CONTINUATION, QUICK_CONTINUATION, DIAGNOSTIC_CONTINUATION


def preflight(data_root: Path, orders_root: Path, output: Path, policy: dict,
              *, prior_gpu_hours: float, remaining_minutes: float, tests_root: Path):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA hardware required for admission measurements")
    if policy["precision"] == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError("Original BF16 stratum requires native BF16-capable hardware")
    if prior_gpu_hours < 0 or not np.isfinite(prior_gpu_hours):
        raise ValueError("Supply all previously used stage GPU hours")
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    clock = SessionClock.start(remaining_minutes, 110, 10)
    started = time.monotonic()
    arrays = PilotArrays(data_root)
    orders = load_orders(orders_root, arrays, 101)
    base = {"code": source_identity(), "runtime": runtime_identity(device), "arrays_sha256": arrays.identity,
            "policy_sha256": digest_json(policy)}
    (output / "pip-freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    checks = ["test_protocol.py", "test_norms.py", "test_model.py", "test_checkpoint_resume.py",
              "test_analysis.py", "test_statistics_budget.py", "test_data.py", "test_pilot.py"]
    if not all((tests_root / name).exists() for name in checks):
        raise ValueError("Bundle missing acceptance tests")
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", *[str(tests_root / name) for name in checks],
        "-o", f"cache_dir={output / 'pytest-cache'}",
        "--basetemp", str(output / "test-tmp"), "--junitxml", str(output / "tests.xml")], capture_output=True, text=True)
    (output / "tests.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        write_json(output / "validation.json", base | {"status": "failed", "reason": "Acceptance tests failed; inspect tests.log"})
        raise RuntimeError("Acceptance tests failed")
    timing, precision_gaps, eval_times, peaks = {}, {}, {}, {}
    replay = False
    try:
        for condition in ("RMS", "Taper-minus"):
            model, optimizer, scaler, sampler = make_state(997, condition, policy, device)
            cursor = 0

            def advance(count, label):
                nonlocal cursor
                elapsed = 0.0
                for i in range(count):
                    if clock.must_stop(60):
                        raise TimeoutError("Preflight reached the safe session/2-hour limit")
                    indices = orders["web"][cursor:cursor + 32].astype(np.int64)
                    torch.cuda.synchronize()
                    tick = time.monotonic()
                    ordered_update(model, optimizer, scaler, arrays.streams["web_train"], indices, policy, device)
                    torch.cuda.synchronize()
                    elapsed += time.monotonic() - tick
                    cursor += 32
                    if (i + 1) % 50 == 0:
                        print(f"Preflight {condition}/{label}: {i + 1}/{count}", flush=True)
                return elapsed / count

            if condition == "RMS":
                advance(50, "warmup")
                timing["rms"] = advance(200, "timed")
                # Ten uninterrupted updates versus a five-update checkpoint and
                # reconstruction into NEW model/optimizer/scaler objects.
                advance(5, "resume-first-half")
                state_path = output / "resume.pt"
                state_id = base | {"condition": condition}
                save(state_path, model, optimizer, scaler, sampler, {"cursor": cursor}, state_id)
                advance(5, "resume-second-half")
                expected = snapshot({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                     "scaler": scaler.state_dict(), "sampler": sampler.get_state(),
                                     "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()})
                torch.save(expected, output / "resume-expected.pt")
                write_json(output / "resume-proof.json", {"identity": state_id, "policy": policy,
                    "expected_sha256": sha256(output / "resume-expected.pt"), "seed": 997, "steps": 5})
                del model, optimizer, scaler
                model, optimizer, scaler, sampler = make_state(997, condition, policy, device)
                restored = restore(state_path, model, optimizer, scaler, sampler, state_id)
                cursor = restored["cursor"]
                advance(5, "replayed-second-half")
                compare({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                         "sampler": sampler.get_state(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, expected)
                replay = True
                del expected
            else:
                advance(50, "calibration-warmup")
                timing["calibration"] = advance(CALIBRATION_END - 50, "genuine-calibration")
                # This is phase performance admission, not a primary trajectory.
                model.completed_updates.fill_(3052)
                advance(50, "intermediate-warmup")
                timing["intermediate"] = advance(200, "intermediate-timed")
                model.completed_updates.fill_(GATE_END)
                advance(50, "zero-warmup")
                timing["zero"] = advance(200, "zero-timed")
            for role in ("full", "quick"):
                tick = time.monotonic()
                for domain in ("web", "python"):
                    selected = evaluate(model, arrays, domain, role=role, rare=orders["rare"], microbatch=policy["microbatch"], precision=policy["precision"], clock=clock)
                    if role == "quick":
                        reference = evaluate(model, arrays, domain, role=role, rare=orders["rare"], microbatch=policy["microbatch"], precision="fp32", clock=clock)
                        gap = abs(selected["ce"] - reference["ce"])
                        precision_gaps[f"{condition}/{domain}"] = gap
                        if gap > policy["numerical_max_ce_gap_nats"]:
                            raise FloatingPointError("Selected precision exceeds the predeclared CE parity tolerance")
                torch.cuda.synchronize()
                # Includes FP32 reference for quick; conservative admission estimate.
                eval_times[f"{condition}/{role}"] = time.monotonic() - tick
            tick = time.monotonic()
            diagnostics(model, arrays, orders["rare"], microbatch=policy["microbatch"], precision=policy["precision"], clock=clock)
            eval_times[f"{condition}/diagnostics"] = time.monotonic() - tick
            tick = time.monotonic()
            save(output / f"{condition}-io.pt", model, optimizer, scaler, sampler, {"cursor": cursor}, base)
            eval_times[f"{condition}/checkpoint"] = time.monotonic() - tick
            peaks[condition] = torch.cuda.max_memory_reserved() / 2**20
            del model, optimizer, scaler
            gc.collect()
            torch.cuda.empty_cache()
        prior = prior_gpu_hours + (time.monotonic() - started) / 3600
        training = 3 * ((9156 + 2 * 6104) * timing["rms"] + 763 * timing["calibration"] +
                         (6104 - 763) * timing["intermediate"] + (9156 - 6104 + 2 * 6104) * timing["zero"])
        full_events = len(FULL_PREFIX) + 2 * len(FULL_CONTINUATION)
        quick_events = 1 + 2 * len(QUICK_CONTINUATION)
        diag_events = 1 + 2 * len(DIAGNOSTIC_CONTINUATION)
        # Event state is fully committed before every scheduled evaluation.
        checkpoint_events = len(set(FULL_PREFIX) | {9156}) + 2 * len(set(FULL_CONTINUATION) | set(QUICK_CONTINUATION) | set(DIAGNOSTIC_CONTINUATION))
        overhead = 3 * sum(full_events * eval_times[f"{c}/full"] + quick_events * eval_times[f"{c}/quick"] +
            diag_events * eval_times[f"{c}/diagnostics"] + (checkpoint_events + 50) * eval_times[f"{c}/checkpoint"]
            for c in ("RMS", "Taper-minus"))
        # A measured full checkpoint is larger than a weight snapshot; use its
        # write time as a conservative allowance for each full-dev weight file.
        overhead += 3 * full_events * sum(eval_times[f"{c}/checkpoint"] for c in ("RMS", "Taper-minus"))
        overhead += 1800  # Explicit reserve for imports, exports, process startup.
        projected = 1.15 * (prior * 3600 + training + overhead) / 3600
        validation = base | {"status": "passed", "fresh_objects_resume_passed": replay, "fp32_ce_gaps": precision_gaps,
                            "acceptance_tests": checks, "cross_session_archive_restore": "must be verified before first branch"}
        benchmark = base | {"status": "passed" if training <= policy["optimizer_hours_cap"] * 3600 and projected <= policy["active_gpu_hours_cap"] else "infeasible",
            "seconds_per_update": timing, "evaluation_and_io_seconds": eval_times, "peak_reserved_mib": peaks,
            "prior_gpu_hours": prior, "projected_optimizer_hours": training / 3600,
            "projected_overhead_hours": overhead / 3600, "projection_with_margin_hours": projected,
            "phase_clock_note": "Taper calibrated genuinely to 763; clocks repositioned for timing. Not a long-run stability result."}
        write_json(output / "validation.json", validation)
        write_json(output / "benchmark.json", benchmark)
        return {"validation": validation, "benchmark": benchmark}
    except Exception as exc:
        write_json(output / "validation.json", base | {"status": "failed", "reason": str(exc), "type": type(exc).__name__})
        raise


def verify_external_resume(preflight_root: Path, data_root: Path, orders_root: Path, output: Path) -> dict:
    """Run in a fresh process against the persisted preflight state before seed 101."""
    if Path("/kaggle/working").exists() and not preflight_root.resolve().is_relative_to(Path("/kaggle/input").resolve()):
        receipt = read_json(preflight_root / "durable-prefixes.json")
        source = Path(receipt["archive"])
        if not source.resolve().is_relative_to(Path("/kaggle/input").resolve()) or sha256(source / "archive-receipt.json" if receipt.get("archive_is_directory") else source) != receipt["archive_sha256"]:
            raise ValueError("Reattach saved preflight outputs under /kaggle/input to prove external recovery")
    proof = read_json(preflight_root / "resume-proof.json")
    base = read_json(preflight_root / "validation.json")
    device = torch.device("cuda")
    torch.set_num_threads(2 if proof["policy"].get("execution", {}).get("mode") == "paired" else 4)
    torch.use_deterministic_algorithms(True)
    if base["code"] != source_identity() or base["runtime"] != runtime_identity(device):
        raise ValueError("External proof came from different code/runtime")
    arrays = PilotArrays(data_root)
    if arrays.identity != base["arrays_sha256"]:
        raise ValueError("External proof data mismatch")
    if sha256(preflight_root / "resume-expected.pt") != proof["expected_sha256"]:
        raise ValueError("Expected replay state checksum mismatch")
    orders = load_orders(orders_root, arrays, 101)
    model, optimizer, scaler, sampler = make_state(proof["seed"], proof["identity"].get("condition", "RMS"), proof["policy"], device)
    progress = restore(preflight_root / "resume.pt", model, optimizer, scaler, sampler, proof["identity"])
    cursor = progress["cursor"]
    for _ in range(proof["steps"]):
        ordered_update(model, optimizer, scaler, arrays.streams["web_train"], orders["web"][cursor:cursor + 32].astype(np.int64), proof["policy"], device)
        cursor += 32
    expected = torch.load(preflight_root / "resume-expected.pt", map_location="cpu", weights_only=True)
    compare({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
             "sampler": sampler.get_state(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, expected)
    result = {k: base[k] for k in ("code", "runtime", "arrays_sha256", "policy_sha256")}
    result.update(status="passed", external_checkpoint_replay=True, source_proof_sha256=sha256(preflight_root / "resume-proof.json"))
    write_json(output, result)
    return result
