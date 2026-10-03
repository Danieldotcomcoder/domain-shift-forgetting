"""Sustained, resumable RMS training on prepared GPT-2 local corpora."""

import argparse
import csv
import gc
import json
import math
import os
import random
import shutil
import uuid
from pathlib import Path
import subprocess
import threading
import time

import numpy as np
import torch
from torch.nn import functional as F

from domain_shift_forgetting.local_corpus import TokenStream, json_write, load_corpus, sha256
from domain_shift_forgetting.models.transformer import Transformer
from domain_shift_forgetting.protocol import CALIBRATION_END, Condition
from domain_shift_forgetting.training.step import make_optimizer


DEFAULT = dict(context=512, batch_size=4, accumulation=4, precision="fp16", lr=0.0006,
               warmup=200, prefix_steps=2000, continuation_steps=1000, eval_every=200,
               eval_windows=128, checkpoint_every=100, seed=17)


def settings(path: Path | None = None, overrides: dict | None = None) -> dict:
    config = DEFAULT | (json.loads(path.read_text()) if path else {}) | (overrides or {})
    if set(config) != set(DEFAULT):
        raise ValueError(f"Unknown config keys: {set(config) - set(DEFAULT)}")
    for key in ("context", "batch_size", "accumulation", "prefix_steps", "continuation_steps", "eval_every", "eval_windows", "checkpoint_every"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(config["warmup"]) is not int or config["warmup"] < 0 or type(config["seed"]) is not int:
        raise ValueError("Invalid warmup or seed")
    if config["context"] > 512 or config["precision"] not in ("fp16", "fp32"):
        raise ValueError("Context <=512 and precision fp16/fp32 required")
    if not math.isfinite(config["lr"]) or config["lr"] <= 0:
        raise ValueError("Learning rate must be finite and positive")
    return config


def create_model(config: dict, device: torch.device, condition: Condition = Condition.RMS):
    if config["precision"] == "fp16" and device.type != "cuda":
        raise ValueError("FP16 training requires CUDA; choose fp32 for CPU checks")
    torch.manual_seed(config["seed"])
    model = Transformer(condition).to(device)
    # Preserve gain/matrix weight-decay groups. CUDA fused Adam reduces launch overhead.
    base = make_optimizer(model)
    groups = [{"params": group["params"], "weight_decay": group["weight_decay"]} for group in base.param_groups]
    optimizer = torch.optim.AdamW(groups, lr=config["lr"], betas=(0.9, 0.95), eps=1e-8,
                                 fused=device.type == "cuda")
    del base
    scaler = torch.amp.GradScaler("cuda", enabled=config["precision"] == "fp16", init_scale=128.0)
    return model, optimizer, scaler


def to_device(rows: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(rows)
    return tensor.pin_memory().to(device, non_blocking=True) if device.type == "cuda" else tensor


def update(model, optimizer, scaler, stream: TokenStream, config: dict, generator: torch.Generator,
           device: torch.device) -> tuple[float, float, int]:
    model.train()
    sampler_state = generator.get_state()
    # Retry the SAME data when FP16 scaling overflows. No skipped logical updates.
    for attempt in range(6):
        generator.set_state(sampler_state)
        # Overflow retries must not double-count calibration observations.
        for name, buffer in model.named_buffers():
            if "pending_" in name:
                buffer.zero_()
        optimizer.zero_grad(set_to_none=True)
        loss_total = torch.zeros((), device=device)
        for _ in range(config["accumulation"]):
            indices = torch.randint(stream.windows, (config["batch_size"],), generator=generator).numpy()
            rows = to_device(stream.take(indices), device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                next_update = int(model.completed_updates) + 1
                logits, _ = model(rows[:, :-1], update=next_update,
                                  collect=next_update <= CALIBRATION_END)
                ce = F.cross_entropy(logits.reshape(-1, 50257), rows[:, 1:].reshape(-1))
            scaler.scale(ce / config["accumulation"]).backward()
            loss_total += ce.detach() / config["accumulation"]
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
        if not bool(torch.isfinite(loss_total)):
            raise FloatingPointError("Nonfinite training CE; restore the latest completed checkpoint")
        if not bool(torch.isfinite(norm)):
            if not scaler.is_enabled():
                raise FloatingPointError("Nonfinite FP32 gradients")
            scaler.update(new_scale=scaler.get_scale() / 2)
            continue
        scaler.step(optimizer)
        scaler.update()
        model.finish_update(int(model.completed_updates) + 1)
        return float(loss_total), float(norm), attempt
    raise FloatingPointError("FP16 overflow persisted after six attempts")


@torch.no_grad()
def evaluate(model, stream: TokenStream, config: dict, device: torch.device) -> float:
    model.eval()
    indices = np.linspace(0, stream.windows - 1, min(config["eval_windows"], stream.windows), dtype=np.int64)
    total = torch.zeros((), dtype=torch.float64, device=device)
    count = 0
    for start in range(0, len(indices), config["batch_size"]):
        rows = to_device(stream.take(indices[start:start + config["batch_size"]]), device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=config["precision"] == "fp16"):
            logits, _ = model(rows[:, :-1])
            ce = F.cross_entropy(logits.reshape(-1, 50257), rows[:, 1:].reshape(-1), reduction="sum")
        total += ce.double()
        count += rows[:, 1:].numel()
    value = float(total) / count
    if not math.isfinite(value):
        raise FloatingPointError("Nonfinite development CE")
    return value


def save(path: Path, model, optimizer, scaler, generator, progress: dict, identity: dict) -> None:
    for name, buffer in model.named_buffers():
        if "pending_" in name and bool(torch.any(buffer != 0)):
            raise ValueError("Cannot save an unfinished calibration update")
    np_state = np.random.get_state()
    payload = {"schema": 2, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
               "python_rng": random.getstate(),
               "numpy_rng": (np_state[0], np_state[1].tolist(), *np_state[2:]),
               "model_training": model.training,
               "sampler_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
               "cuda_rng": torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [],
               "progress": progress, "identity": identity}
    # Immutable, checksummed generations plus an atomic small pointer. Readers
    # see either the old or the new generation even if publication is interrupted.
    # Keep the conventional .pt file for inspection; restore uses the pointer.
    generation = path.with_name(f".{path.stem}-{uuid.uuid4().hex}.pt")
    pointer = path.with_suffix(path.suffix + ".json")
    old = json.loads(pointer.read_text()) if pointer.exists() else None
    with generation.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    record = {"file": generation.name, "sha256": sha256(generation), "previous": old and
              {"file": old["file"], "sha256": old["sha256"]}}
    temporary = pointer.with_name(f".{pointer.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(record, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(pointer)
    alias = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        os.link(generation, alias)
    except OSError:
        shutil.copyfile(generation, alias)
    alias.replace(path)
    if os.name != "nt":
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    if old and old.get("previous"):
        obsolete = path.parent / old["previous"]["file"]
        if obsolete.parent == path.parent and obsolete.name.startswith(f".{path.stem}-"):
            obsolete.unlink(missing_ok=True)


def restore(path: Path, model, optimizer, scaler, generator, identity: dict) -> dict:
    pointer = path.with_suffix(path.suffix + ".json")
    if not pointer.exists():
        raise ValueError("Checkpoint integrity pointer missing; legacy checkpoints require the original runner")
    record = json.loads(pointer.read_text())
    if Path(record["file"]).name != record["file"]:
        raise ValueError("Invalid checkpoint generation path")
    generation = path.parent / record["file"]
    if sha256(generation) != record["sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    payload = torch.load(generation, map_location="cpu", weights_only=True)
    if payload.get("schema") != 2:
        raise ValueError("Checkpoint schema mismatch")
    if payload["identity"] != identity:
        raise ValueError("Checkpoint/config/data/code identity mismatch")
    if payload["cuda_rng"] and (not torch.cuda.is_available() or len(payload["cuda_rng"]) != torch.cuda.device_count()):
        raise ValueError("CUDA RNG topology mismatch")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scaler.load_state_dict(payload["scaler"])
    generator.set_state(payload["sampler_rng"])
    torch.set_rng_state(payload["torch_rng"])
    random.setstate(payload["python_rng"])
    state = payload["numpy_rng"]
    np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
    model.train(payload["model_training"])
    optimizer.zero_grad(set_to_none=True)
    if payload["cuda_rng"]:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return payload["progress"]


def code_identity() -> dict:
    root = Path(__file__).parent
    return {name: sha256(root / name) for name in (
        "local_training.py", "local_corpus.py", "models/transformer.py", "models/norms.py", "training/step.py", "protocol.py")}


def learning_rate(config: dict, global_step: int) -> float:
    if config["warmup"] and global_step <= config["warmup"]:
        return config["lr"] * global_step / config["warmup"]
    total = config["prefix_steps"] + config["continuation_steps"]
    phase = min(1.0, max(0.0, (global_step - config["warmup"]) / max(1, total - config["warmup"])))
    return config["lr"] * (0.1 + 0.9 * (1 + math.cos(math.pi * phase)) / 2)


def train(config: dict, data_root: Path, output: Path, device: torch.device,
          resume: bool = False, max_updates: int | None = None) -> dict:
    config = settings(overrides=config)
    if max_updates is not None and max_updates < 1:
        raise ValueError("max-updates must be positive")
    torch.set_num_threads(4)
    streams, corpus_hash = load_corpus(data_root, config["context"])
    identity = {"config": config, "corpus_sha256": corpus_hash, "code": code_identity(),
                "torch": str(torch.__version__), "device": str(device)}
    if resume:
        original = json.loads((output / "manifest.json").read_text())
        if original["identity"] != identity:
            raise ValueError("Cannot resume with changed settings, data, code, runtime, or device")
        if (output / "summary.json").exists():
            raise ValueError("This run is already complete")
    else:
        output.mkdir(parents=True, exist_ok=False)
        json_write(output / "manifest.json", {"purpose": "exploratory RMS baseline; not H1 evidence", "identity": identity,
                   "data_root": str(data_root.resolve()), "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
                   "tokenizer": "gpt2", "parameters": 17718784, "effective_tokens_per_update":
                   config["context"] * config["batch_size"] * config["accumulation"]})
    try:
        model, optimizer, scaler = create_model(config, device)
    except Exception as exc:
        json_write(output / "status.json", {"status": "failed", "error": repr(exc)})
        raise
    generator = torch.Generator().manual_seed(config["seed"] + 1)
    progress = {"stage": "prefix", "step": 0, "history": [], "endpoints": {}, "elapsed_seconds": 0.0,
                "overflow_retries": 0}
    if resume:
        progress = restore(output / "latest.pt", model, optimizer, scaler, generator, identity)
    prior_elapsed, started = progress["elapsed_seconds"], time.perf_counter()
    invocation_updates = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def status(state: str, **extra) -> None:
        json_write(output / "status.json", {"status": state, "stage": progress["stage"], "step": progress["step"], **extra})

    def checkpoint(name: str = "latest.pt") -> None:
        progress["elapsed_seconds"] = prior_elapsed + time.perf_counter() - started
        save(output / name, model, optimizer, scaler, generator, progress, identity)

    def record(ce: float | None = None, norm: float | None = None, throughput: float | None = None) -> list[float]:
        values = [evaluate(model, streams[f"{domain}_dev"], config, device) for domain in ("web", "python")]
        row = {"stage": progress["stage"], "step": progress["step"], "global_step": int(model.completed_updates),
               "train_ce": ce, "gradient_norm": norm, "lr": optimizer.param_groups[0]["lr"],
               "web_dev_ce": values[0], "python_dev_ce": values[1], "tokens_per_second": throughput,
               "elapsed_seconds": prior_elapsed + time.perf_counter() - started,
               "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
               "overflow_retries": progress["overflow_retries"]}
        progress["history"].append(row)
        temporary = output / "metrics.tmp"
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(progress["history"])
        temporary.replace(output / "metrics.csv")
        print(f"{progress['stage']} {progress['step']}: prose CE {values[0]:.4f}, Python CE {values[1]:.4f}", flush=True)
        return values

    try:
        status("running")
        if not progress["history"]:
            progress["initial_ce"] = record()
            checkpoint()
        while True:
            stage = progress["stage"]
            steps = config["prefix_steps"] if stage == "prefix" else config["continuation_steps"]
            stream = streams["python_train" if stage == "python" else "web_train"]
            while progress["step"] < steps:
                lr = learning_rate(config, int(model.completed_updates) + 1)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                tick = time.perf_counter()
                ce, norm, retries = update(model, optimizer, scaler, stream, config, generator, device)
                progress["step"] += 1
                progress["overflow_retries"] += retries
                invocation_updates += 1
                throughput = config["context"] * config["batch_size"] * config["accumulation"] / (time.perf_counter() - tick)
                if progress["step"] % config["eval_every"] == 0 or progress["step"] == steps:
                    progress["endpoints"][stage] = record(ce, norm, throughput)
                    checkpoint()
                elif progress["step"] % config["checkpoint_every"] == 0:
                    checkpoint()
                if progress["step"] % 10 == 0:
                    status("running", train_ce=ce, tokens_per_second=throughput)
                if max_updates is not None and invocation_updates >= max_updates and not (stage == "python" and progress["step"] == steps):
                    checkpoint()
                    status("paused", reason="max-updates reached")
                    return progress
            if stage == "prefix":
                checkpoint("switch.pt")
            else:
                checkpoint(f"{stage}-final.pt")
            if stage == "python":
                break
            # Restore the same prefix model, moments, scale, RNG and LR clock for each branch.
            restore(output / "switch.pt", model, optimizer, scaler, generator, identity)
            progress["stage"] = "web" if stage == "prefix" else "python"
            progress["step"] = 0
            record()
            checkpoint()
        endpoint = progress["endpoints"]
        summary = {"status": "completed", "purpose": "exploratory RMS baseline; no Taper comparison",
                   "endpoints_prose_python_ce": endpoint,
                   "web_deterioration_after_python": endpoint["python"][0] - endpoint["prefix"][0],
                   "excess_web_ce_after_switch": endpoint["python"][0] - endpoint["web"][0],
                   "elapsed_seconds": prior_elapsed + time.perf_counter() - started}
        json_write(output / "summary.json", summary)
        status("completed")
        return progress
    except (Exception, KeyboardInterrupt) as exc:
        # Never serialize a possibly half-finished optimizer update. Resume latest.pt.
        status("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=repr(exc),
               resume_from="latest.pt")
        raise


class Telemetry:
    def __init__(self):
        self.samples: list[dict] = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self) -> None:
        while not self.stop.is_set():
            try:
                result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,memory.used",
                                         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                values = [float(x.strip()) for x in result.stdout.splitlines()[0].split(",")]
                self.samples.append(dict(utilization=values[0], temperature=values[1], used_mib=values[2]))
            except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
                pass
            self.stop.wait(1)


def benchmark(output: Path, hours: float = 2, seconds: float = 12) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the RTX benchmark")
    if not math.isfinite(hours) or hours <= 0 or not math.isfinite(seconds) or seconds < 1:
        raise ValueError("Positive hours and at least one measurement second required")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    # Synthetic tokens test compute/memory only; they are never used as training corpora.
    path = output / "hardware-only.bin"
    np.random.default_rng(17).integers(0, 50257, size=262145, dtype=np.uint16).tofile(path)
    stream = TokenStream(path, 512)
    free, total = torch.cuda.mem_get_info()
    cap = min(total * 0.85, free - 768 * 2**20)
    results = []
    for size in (1, 2, 4, 8, 16):
        config = settings(overrides={"batch_size": size, "accumulation": 16 // size})
        if results and results[-1].get("peak_reserved_bytes", 0) * 2 > cap:
            break
        model = optimizer = scaler = None
        monitor = Telemetry()
        try:
            gc.collect()
            torch.cuda.empty_cache()
            model, optimizer, scaler = create_model(config, device)
            generator = torch.Generator().manual_seed(18)
            for _ in range(2):
                update(model, optimizer, scaler, stream, config, generator, device)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            monitor.thread.start()
            started, updates = time.perf_counter(), 0
            while updates < 3 or time.perf_counter() - started < seconds:
                update(model, optimizer, scaler, stream, config, generator, device)
                updates += 1
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            reserved = torch.cuda.max_memory_reserved()
            row = {"batch_size": size, "accumulation": 16 // size, "updates": updates,
                   "seconds_per_update": elapsed / updates, "tokens_per_second": updates * 8192 / elapsed,
                   "peak_reserved_bytes": reserved, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "within_memory_budget": reserved <= cap, "telemetry": monitor.samples}
            results.append(row)
            print(f"batch {size}: {row['tokens_per_second']:,.0f} tokens/s, {reserved / 2**20:.0f} MiB reserved", flush=True)
        except torch.OutOfMemoryError:
            results.append({"batch_size": size, "out_of_memory": True})
            break
        finally:
            monitor.stop.set()
            if monitor.thread.is_alive():
                monitor.thread.join(timeout=6)
            del model, optimizer, scaler
            gc.collect()
            torch.cuda.empty_cache()
    eligible = [row for row in results if row.get("within_memory_budget")]
    if not eligible:
        json_write(output / "benchmark.json", {"candidates": results, "status": "no_feasible_candidate"})
        raise RuntimeError("No candidate fits with headroom; free GPU memory and retry")
    best = max(eligible, key=lambda row: row["tokens_per_second"])
    # 15% allowance for evaluation/checkpoints; thermal drift can still exceed this estimate.
    unit = max(1, int(hours * 3600 / (best["seconds_per_update"] * 1.15 * 4)))
    config = settings(overrides={"batch_size": best["batch_size"], "accumulation": best["accumulation"],
                                "prefix_steps": unit * 2, "continuation_steps": unit})
    json_write(output / "recommended.json", config)
    result = {"status": "completed", "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
              "purpose": "synthetic hardware benchmark only", "memory_budget_bytes": cap,
              "candidates": results, "selected_batch_size": best["batch_size"], "target_hours": hours,
              "estimated_training_hours": unit * 4 * best["seconds_per_update"] / 3600,
              "planned_training_tokens": unit * 4 * 8192,
              "note": "Short-run estimate, not a deadline or thermal soak; suggested config does not launch training."}
    json_write(output / "benchmark.json", result)
    print(f"Recommended config: {output / 'recommended.json'}; training was NOT launched.")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--hours", type=float, default=2)
    bench.add_argument("--seconds", type=float, default=12)
    start = sub.add_parser("train")
    start.add_argument("--config", type=Path)
    start.add_argument("--data-dir", type=Path, default=Path("data/local-large"))
    start.add_argument("--output", type=Path, required=True)
    start.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    start.add_argument("--max-updates", type=int)
    resume = sub.add_parser("resume")
    resume.add_argument("--output", type=Path, required=True)
    resume.add_argument("--max-updates", type=int)
    resume.add_argument("--data-dir", type=Path, help="Relocated corpus; content hashes must still match")
    args = p.parse_args()
    if args.command == "benchmark":
        benchmark(args.output, args.hours, args.seconds)
    elif args.command == "train":
        train(settings(args.config), args.data_dir, args.output, torch.device(args.device), max_updates=args.max_updates)
    else:
        manifest = json.loads((args.output / "manifest.json").read_text())
        train(manifest["identity"]["config"], args.data_dir or Path(manifest["data_root"]), args.output,
              torch.device(manifest["identity"]["device"]), resume=True, max_updates=args.max_updates)


if __name__ == "__main__":
    main()
