"""Small, explicitly non-protocol language-model training lab.

Run with python -m domain_shift_forgetting.local_lab. No research data are read.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import torch
from torch import nn
from torch.nn import functional as F

from domain_shift_forgetting.models.transformer import Block, Transformer
from domain_shift_forgetting.models.norms import RMSNorm
from domain_shift_forgetting.protocol import Condition
from domain_shift_forgetting.training.step import make_optimizer


class TinyLM(nn.Module):
    """Two existing research blocks; 257 byte/EOS tokens and tied output."""

    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(257, 256)
        self.position_embedding = nn.Embedding(512, 256)
        self.blocks = nn.ModuleList([Block(Condition.RMS) for _ in range(2)])
        self.final_norm = RMSNorm(256)
        self.apply(Transformer._initialize)

    def forward(self, tokens):
        x = self.token_embedding(tokens) + self.position_embedding(
            torch.arange(tokens.shape[1], device=tokens.device))
        for block in self.blocks:
            x = block(x, update=1, collect=False)
        return F.linear(self.final_norm(x), self.token_embedding.weight), None


def documents(domain, split):
    """Original synthetic fixtures, disjoint documents but shared templates."""
    offset = 0 if split == "train" else 10000
    count = 128 if split == "train" else 24
    subjects = ["garden", "river", "library", "village", "forest", "station"]
    result = []
    for i in range(offset, offset + count):
        subject = subjects[i % len(subjects)]
        if domain == "web":
            text = (f"Field report {i}: We visited the {subject} in the morning. "
                    f"The team recorded {i + 3} observations and compared the notes. "
                    "A careful explanation helps readers understand what changed. "
                    "We will return tomorrow to check the measurements again.\n")
        else:
            text = (f"def measure_{i}(values):\n"
                    f"    # Record observations for the {subject}.\n"
                    f"    total = {i + 3}\n"
                    "    for value in values:\n"
                    "        if value > 0:\n"
                    "            total += value\n"
                    "    return total / max(len(values), 1)\n")
        result.append(text.encode("utf-8"))
    return result


def load_data(root, length):
    data, provenance, seen = {}, {}, {}
    for domain in ("web", "python"):
        for split in ("train", "dev"):
            key = f"{domain}_{split}"
            if root is None:
                docs = documents(domain, split)
            else:
                folder = root / domain / split
                paths = sorted(folder.glob("*.txt"))
                if not paths:
                    raise ValueError(f"Expected UTF-8 .txt documents in {folder}")
                docs = [p.read_bytes() for p in paths]
            hashes, blocks = [], []
            for doc in docs:
                doc.decode("utf-8")  # Validate without altering bytes.
                digest = hashlib.sha256(doc).hexdigest()
                if digest in seen:
                    raise ValueError(f"Duplicate document in {key} and {seen[digest]}")
                seen[digest] = key
                hashes.append(digest)
                tokens = list(doc) + [256]
                # Windows stay inside a document; no padding or dev/train overlap.
                for start in range(0, len(tokens) - length, length):
                    blocks.append(tokens[start:start + length + 1])
            if not blocks:
                raise ValueError(f"{key}: no documents long enough for context {length}")
            data[key] = torch.tensor(blocks, dtype=torch.long)
            provenance[key] = {"document_sha256": hashes, "windows": len(blocks),
                               "evaluated_or_available_labels": len(blocks) * length}
    return data, provenance


def batch(data, size, generator, device):
    rows = data[torch.randint(len(data), (size,), generator=generator)].to(device)
    return rows[:, :-1], rows[:, 1:]


def update(model, optimizer, data, args, generator, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for _ in range(args.accumulation):
        inputs, labels = batch(data, args.batch_size, generator, device)
        logits, _ = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite training loss")
        (loss / args.accumulation).backward()
        total += float(loss.detach()) / args.accumulation
    norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True))
    optimizer.step()
    return total, norm


@torch.no_grad()
def evaluate(model, data, size, device):
    model.eval()
    loss_sum, count = 0.0, 0
    for rows in data.split(size):
        rows = rows.to(device)
        logits, _ = model(rows[:, :-1])
        labels = rows[:, 1:]
        loss_sum += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                         labels.reshape(-1), reduction="sum"))
        count += labels.numel()
    ce = loss_sum / count
    if not math.isfinite(ce):
        raise FloatingPointError("Nonfinite development CE")
    return ce


def save_checkpoint(path, model, optimizer, generator, step):
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "sampler_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
             "step": step}
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def restore_checkpoint(path, model, optimizer, generator):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    generator.set_state(state["sampler_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["step"]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["tiny", "pilot-rms"], default="tiny")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--prefix-steps", type=int, default=100)
    p.add_argument("--continuation-steps", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--context", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accumulation", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.0006)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--output", type=Path, default=Path("artifacts/local-lab") / time.strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--smoke", action="store_true", help="Two updates per stage; pipeline check only")
    return p


def run(args):
    if args.smoke:
        args.prefix_steps = args.continuation_steps = 2
        args.eval_every = 1
    for name in ("prefix_steps", "continuation_steps", "eval_every", "context", "batch_size", "accumulation"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.context > 512 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("Context must be <=512 and learning rate finite and positive")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: install the CUDA wheel or use --device cpu")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed + 1)
    data, provenance = load_data(args.data_dir, args.context)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        model = (TinyLM() if args.model == "tiny" else Transformer(Condition.RMS)).to(device)
        optimizer = make_optimizer(model)
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        manifest = {"purpose": "learning-only; not protocol-v3 evidence", "arguments": vars(args),
                    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "torch": torch.__version__, "python": platform.python_version(),
                    "cuda_runtime": torch.version.cuda, "device": str(device),
                    "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "precision": "float32", "tokenizer": "UTF-8 bytes 0..255; EOS 256",
                    "data_kind": "custom" if args.data_dir else "synthetic shared-template fixtures",
                    "data": provenance}
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["stage", "step", "global_step", "train_ce", "grad_norm", "clipped",
                      "train_tokens_per_second", "web_dev_ce", "python_dev_ce", "web_dev_ppl",
                      "python_dev_ppl", "elapsed_seconds", "peak_allocated_mib", "peak_reserved_mib"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()

            def record(stage, step, global_step, loss=None, norm=None, throughput=None):
                web = evaluate(model, data["web_dev"], args.batch_size, device)
                python = evaluate(model, data["python_dev"], args.batch_size, device)
                writer.writerow(dict(stage=stage, step=step, global_step=global_step,
                    train_ce=loss, grad_norm=norm, clipped=None if norm is None else norm > 1,
                    train_tokens_per_second=throughput, web_dev_ce=web, python_dev_ce=python,
                    web_dev_ppl=math.exp(web), python_dev_ppl=math.exp(python),
                    elapsed_seconds=time.perf_counter() - started,
                    peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
                    peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20 if device.type == "cuda" else None))
                handle.flush()
                print(f"{stage:8s} {step:4d}: web CE={web:.4f}, Python CE={python:.4f}", flush=True)
                return web, python

            initial = record("initial", 0, 0)
            endpoints = {}
            for stage, source, steps in [("prefix", "web", args.prefix_steps),
                                         ("web", "web", args.continuation_steps),
                                         ("python", "python", args.continuation_steps)]:
                offset = 0
                if stage != "prefix":
                    offset = restore_checkpoint(args.output / "switch.pt", model, optimizer, generator)
                    record(stage, 0, offset)
                for step in range(1, steps + 1):
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    tick = time.perf_counter()
                    loss, norm = update(model, optimizer, data[f"{source}_train"], args, generator, device)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    throughput = args.batch_size * args.context * args.accumulation / (time.perf_counter() - tick)
                    if step % args.eval_every == 0 or step == steps:
                        endpoints[stage] = record(stage, step, offset + step, loss, norm, throughput)
                save_checkpoint(args.output / ("switch.pt" if stage == "prefix" else f"{stage}-final.pt"),
                                model, optimizer, generator, offset + steps)
            summary = {"status": "completed", "purpose": "learning-only; no H1/Taper inference",
                       "initial_ce": initial, "endpoints_web_python_ce": endpoints,
                       "web_deterioration_after_python": endpoints["python"][0] - endpoints["prefix"][0],
                       "web_deterioration_after_web": endpoints["web"][0] - endpoints["prefix"][0],
                       "excess_web_ce_after_switch": endpoints["python"][0] - endpoints["web"][0],
                       "elapsed_seconds": time.perf_counter() - started}
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except (Exception, KeyboardInterrupt) as exc:
        (args.output / "failure.json").write_text(json.dumps({"status": "failed", "error": repr(exc)}), encoding="utf-8")
        raise
    print(f"Saved learning run to {args.output.resolve()}")


def main():
    args = parser().parse_args()
    try:
        run(args)
    except torch.OutOfMemoryError:
        raise SystemExit("Out of memory. Retry in a NEW output folder with --batch-size 1 --context 32; use --model tiny.")


if __name__ == "__main__":
    main()
