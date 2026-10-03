"""Bounded single-GPU portability benchmark; never launches the scientific pilot."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .local_corpus import TokenStream, json_write
from .local_training import code_identity, create_model, evaluate, restore, save, settings, update
from .models.norms import TaperNorm
from .protocol import Condition, GATE_END


def snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [snapshot(item) for item in value]
    return value


def compare(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-6)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            compare(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            compare(a, b)
    else:
        assert actual == expected


def run(output: Path, *, case="rms", precision="fp16", updates=10, batch_size=2,
        context=512, effective_sequences=32, device="cuda"):
    if case not in ("rms", "taper-zero") or precision not in ("fp16", "fp32"):
        raise ValueError("Invalid benchmark case/precision")
    if updates < 1 or updates > 100 or batch_size < 1 or effective_sequences % batch_size:
        raise ValueError("Use 1..100 updates and a batch size dividing effective sequences")
    if effective_sequences < 1:
        raise ValueError("Effective sequences must be positive")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU accelerator first")
    config = settings(overrides=dict(precision=precision, context=context, batch_size=batch_size,
                                    accumulation=effective_sequences // batch_size, eval_windows=4))
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    # Fixed seeds and identical tokens for every case and precision.
    path = output / "synthetic.bin"
    np.random.default_rng(101).integers(0, 50257, size=context * 64 + 1, dtype=np.uint16).tofile(path)
    stream = TokenStream(path, context)
    identity = {"config": config, "case": case, "code": code_identity(),
                "benchmark_code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "torch": str(torch.__version__), "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                "purpose": "synthetic portability test, not H1 evidence"}
    json_write(output / "manifest.json", identity)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    started = time.perf_counter()
    try:
        model, optimizer, scaler = create_model(config, device,
            Condition.RMS if case == "rms" else Condition.TAPER_MINUS)
        if case == "taper-zero":
            # Deliberate fixture: c=1 is NOT a trained/calibrated Taper checkpoint.
            # Exercise the actual zero-gate operator without faking scientific evidence.
            model.completed_updates.fill_(GATE_END)
            for module in model.modules():
                if isinstance(module, TaperNorm):
                    module.calibrated.fill_(True)
        generator = torch.Generator().manual_seed(102)
        # Compare CE under both precisions at the identical initial state.
        fp32_ce = evaluate(model, stream, config | {"precision": "fp32"}, device)
        selected_ce = evaluate(model, stream, config, device)
        for _ in range(2):
            update(model, optimizer, scaler, stream, config, generator, device)
        sync()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        tick = time.perf_counter()
        retries = 0
        losses = []
        for _ in range(updates):
            ce, _, retry = update(model, optimizer, scaler, stream, config, generator, device)
            losses.append(ce)
            retries += retry
        sync()
        training_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        end_fp32_ce = evaluate(model, stream, config | {"precision": "fp32"}, device)
        end_selected_ce = evaluate(model, stream, config, device)
        sync()
        evaluation_seconds = time.perf_counter() - tick
        checkpoint = output / "latest.pt"
        tick = time.perf_counter()
        save(checkpoint, model, optimizer, scaler, generator,
             {"completed_updates": int(model.completed_updates)}, identity)
        checkpoint_seconds = time.perf_counter() - tick
        update(model, optimizer, scaler, stream, config, generator, device)
        expected = snapshot({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                             "scaler": scaler.state_dict(), "sampler": generator.get_state()})
        tick = time.perf_counter()
        restore(checkpoint, model, optimizer, scaler, generator, identity)
        sync()
        restore_seconds = time.perf_counter() - tick
        update(model, optimizer, scaler, stream, config, generator, device)
        compare({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict(), "sampler": generator.get_state()}, expected)
        result = {"status": "passed", "identity": identity, "resume_replay_passed": True,
                  "updates": updates, "tokens_per_update": context * effective_sequences,
                  "training_seconds": training_seconds,
                  "training_tokens_per_second": updates * context * effective_sequences / training_seconds,
                  "evaluation_seconds": evaluation_seconds, "checkpoint_seconds": checkpoint_seconds,
                  "restore_seconds": restore_seconds, "overflow_retries": retries,
                  "initial_ce_fp32": fp32_ce, "initial_ce_selected": selected_ce,
                  "final_ce_fp32": end_fp32_ce, "final_ce_selected": end_selected_ce,
                  "final_precision_ce_absolute_difference": abs(end_selected_ce - end_fp32_ce),
                  "training_ce": losses,
                  "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None,
                  "elapsed_seconds": time.perf_counter() - started,
                  "note": "Synthetic c=1 zero-gate fixture; no calibration or trained-trajectory stability claim. "
                          "Timing includes separate eval/checkpoint measurements; no full-pilot feasibility claim."}
        if not all(math.isfinite(x) for x in losses + [fp32_ce, selected_ce, end_fp32_ce, end_selected_ce]):
            raise FloatingPointError("Nonfinite benchmark result")
    except (Exception, KeyboardInterrupt) as exc:
        json_write(output / "benchmark.json", {"status": "failed", "identity": identity, "error": repr(exc)})
        raise
    json_write(output / "benchmark.json", result)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=["rms", "taper-zero"], default="rms")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    run(args.output, case=args.case, precision=args.precision, updates=args.updates, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
