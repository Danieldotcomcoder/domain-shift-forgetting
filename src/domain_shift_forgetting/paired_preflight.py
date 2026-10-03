"""Real RMS/Taper concurrent phase timings and per-device resume proofs."""
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from .kaggle import snapshot, compare
from .local_corpus import sha256
from .local_training import save, restore
from .paired_control import barrier
from .pilot_arrays import PilotArrays, load_orders, evaluate
from .pilot_control import SessionClock, digest_json, write_json
from .pilot_diagnostics import diagnostics
from .pilot_runner import make_state, ordered_update, runtime_identity, source_identity
from .protocol import FULL_PREFIX, FULL_CONTINUATION, QUICK_CONTINUATION, DIAGNOSTIC_CONTINUATION


def projected_work(condition, timing, evaluation):
    training = 3 * ((9156 + 2 * 6104) * max(timing.values()) if condition == "RMS" else
                    763 * timing["calibration"] + 5341 * timing["intermediate"] + 15260 * timing["zero"])
    full = len(FULL_PREFIX) + 2 * len(FULL_CONTINUATION)
    quick = 1 + 2 * len(QUICK_CONTINUATION)
    diagnostics_count = 1 + 2 * len(DIAGNOSTIC_CONTINUATION)
    checkpoints = len(set(FULL_PREFIX) | {9156}) + 2 * len(set(FULL_CONTINUATION) | set(QUICK_CONTINUATION) | set(DIAGNOSTIC_CONTINUATION))
    overhead = 3 * (full * evaluation["full"] + quick * evaluation["quick"] + diagnostics_count * evaluation["diagnostics"]
                    + (checkpoints + 50 + full) * evaluation["checkpoint"])
    return {"optimizer_hours": training / 3600, "overhead_hours": overhead / 3600,
            "total_worker_hours": (training + overhead) / 3600}


def worker(data_root, order_root, output, policy, condition, tests_root, shared, stop_file, remaining_minutes):
    output.mkdir(parents=True, exist_ok=False)
    clock = SessionClock.start(remaining_minutes, 110, 10)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Each preflight worker must see exactly one CUDA GPU")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    arrays = PilotArrays(data_root)
    orders = load_orders(order_root, arrays, 101)
    base = {"code": source_identity(), "runtime": runtime_identity(device), "arrays_sha256": arrays.identity,
            "policy_sha256": digest_json(policy)}
    (output / "pip-freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    checks = ["test_protocol.py", "test_norms.py", "test_model.py", "test_checkpoint_resume.py", "test_analysis.py",
              "test_statistics_budget.py", "test_data.py", "test_pilot.py", "test_paired.py"]
    try:
        result = subprocess.run([sys.executable, "-m", "pytest", "-q", *[str(tests_root / name) for name in checks],
            "-o", f"cache_dir={output / 'pytest-cache'}", "--basetemp", str(output / "test-tmp"),
            "--junitxml", str(output / "tests.xml")], capture_output=True, text=True)
        (output / "tests.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            print(result.stdout[-12000:], flush=True)
            raise RuntimeError("Acceptance tests failed; inspect this worker's tests.log")
        model, optimizer, scaler, sampler = make_state(997, condition, policy, device)
        cursor, timing, evaluation, gaps = 0, {}, {}, {}
        torch.cuda.reset_peak_memory_stats()

        def sync(label):
            barrier(shared, label, condition, clock, stop_file)

        def advance(count, label):
            nonlocal cursor
            elapsed = 0.0
            for i in range(count):
                if stop_file.exists() or clock.must_stop(30):
                    raise TimeoutError("Paired preflight interrupted at an update boundary")
                indices = orders["web"][cursor:cursor + 32].astype(np.int64)
                torch.cuda.synchronize()
                tick = time.monotonic()
                ordered_update(model, optimizer, scaler, arrays.streams["web_train"], indices, policy, device)
                torch.cuda.synchronize()
                elapsed += time.monotonic() - tick
                cursor += 32
                if (i + 1) % 50 == 0:
                    print(f"{condition}/{label}: {i + 1}/{count}", flush=True)
            return elapsed / count

        for phase in ("calibration", "intermediate", "zero"):
            if condition == "Taper-minus" and phase != "calibration":
                model.completed_updates.fill_(3052 if phase == "intermediate" else 6104)
            sync(phase + "-warmup")
            advance(50, phase + "-warmup")
            sync(phase + "-timed")
            timing[phase] = advance(713 if phase == "calibration" else 200, phase)
            sync(phase + "-done")

        # Both conditions produce a real checkpoint+expected continuation proof.
        state_id = base | {"condition": condition}
        sync("replay")
        advance(5, "replay-first-half")
        save(output / "resume.pt", model, optimizer, scaler, sampler, {"cursor": cursor}, state_id)
        advance(5, "replay-second-half")
        expected = snapshot({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                             "sampler": sampler.get_state(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()})
        torch.save(expected, output / "resume-expected.pt")
        write_json(output / "resume-proof.json", {"identity": state_id, "policy": policy, "seed": 997, "steps": 5,
                   "expected_sha256": sha256(output / "resume-expected.pt")})
        del model, optimizer, scaler
        model, optimizer, scaler, sampler = make_state(997, condition, policy, device)
        cursor = restore(output / "resume.pt", model, optimizer, scaler, sampler, state_id)["cursor"]
        advance(5, "replay-restored")
        compare({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                 "sampler": sampler.get_state(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, expected)
        del expected
        for role in ("full", "quick"):
            sync("evaluation-" + role)
            tick = time.monotonic()
            selected = {domain: evaluate(model, arrays, domain, role=role, rare=orders["rare"],
                microbatch=policy["microbatch"], precision=policy["precision"], clock=clock) for domain in ("web", "python")}
            torch.cuda.synchronize()
            evaluation[role] = time.monotonic() - tick
            if role == "quick":
                sync("fp32-reference")
                for domain in ("web", "python"):
                    reference = evaluate(model, arrays, domain, role="quick", rare=orders["rare"],
                        microbatch=policy["microbatch"], precision="fp32", clock=clock)
                    gaps[domain] = abs(reference["ce"] - selected[domain]["ce"])
                    if gaps[domain] > policy["numerical_max_ce_gap_nats"]:
                        raise FloatingPointError("FP16 / FP32 CE parity gate failed")
        sync("diagnostics")
        tick = time.monotonic()
        diagnostic = diagnostics(model, arrays, orders["rare"], microbatch=policy["microbatch"], precision=policy["precision"], clock=clock)
        evaluation["diagnostics"] = time.monotonic() - tick
        write_json(output / "diagnostics.json", diagnostic)
        sync("checkpoint-io")
        tick = time.monotonic()
        save(output / "io.pt", model, optimizer, scaler, sampler, {"cursor": cursor}, state_id)
        evaluation["checkpoint"] = time.monotonic() - tick
        benchmark = base | {"status": "measured", "condition": condition, "seconds_per_update": timing,
            "evaluation_and_io_seconds": evaluation, "projection": projected_work(condition, timing, evaluation),
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "note": "Matched RMS/Taper phases run concurrently. Taper clocks repositioned after genuine calibration. FP32 reference time excluded from recurring evaluation projection."}
        write_json(output / "validation.json", base | {"status": "passed", "fresh_objects_resume_passed": True,
            "condition": condition, "fp32_ce_gaps": gaps, "acceptance_tests": checks})
        write_json(output / "benchmark.json", benchmark)
    except Exception as exc:
        write_json(output / "validation.json", base | {"status": "failed", "reason": str(exc)})
        raise
