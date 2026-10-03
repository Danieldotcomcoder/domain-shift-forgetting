#!/usr/bin/env python3
"""H1 (protocol v3) single-notebook runner for Kaggle T4 x2.

H1: does internal TaperNorm (Taper-minus) increase persistent held-out web
deterioration after a web -> Python switch, relative to continuing web training,
compared with RMSNorm?

One file does everything: verifies the prepared C4/Stack arrays, trains
RMS on GPU 0 and Taper-minus on GPU 1 (seeds 101, 102, 103; prefix -> web
branch -> Python branch), evaluates on the frozen schedule, checkpoints, and
applies the frozen decision rules. All state lives in ``<work>/h1state``.
On Kaggle that is the notebook's own output, which the next run of the same
notebook receives as input and resumes from automatically.

The model, TaperNorm, optimizer, update step and the analysis/decision rules are
transcribed from ``src/domain_shift_forgetting`` (tested code). Only the
orchestration is new. Scientific amendments are recorded in CONFIG below.

Commands:
  python h1_run.py main            # orchestrator (Kaggle notebook cell)
  python h1_run.py worker ...      # internal: one GPU, one condition
  python h1_run.py report --state DIR
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback
from typing import Iterable, Mapping, Sequence

import numpy as np

RUNNER_VERSION = "h1-single-notebook-1"

# =============================================================================
# Protocol constants (protocol v3, unchanged). ``use_mini_protocol`` shrinks them
# ONLY for local plumbing tests; it is never used on Kaggle.
# =============================================================================
SEEDS = (101, 102, 103)
CONDITIONS = ("RMS", "Taper-minus")
CONTEXT = 512
VOCAB = 50257
EOS = 50256
SEQUENCES_PER_UPDATE = 32
TOKENS_PER_UPDATE = 16_384
WARMUP_END = 305
CALIBRATION_END = 763
GATE_END = 6_104
PREFIX_END = 9_156
CONTINUATION_UPDATES = 6_104
TRAJECTORY_END = 15_260
FULL_PREFIX = (763, 3052, 6104, 6409, 6714, 7019, 7324, 7629, 7934, 8239, 8544, 8849, 9156)
QUICK_CONTINUATION = tuple(sorted({0, 1, 2, 5, 10, 20, 1525, 3050, 6104}
                                  | set(range(50, 1001, 50)) | set(range(1100, 6101, 100))))
FULL_CONTINUATION = tuple(sorted({0, 1, 10, 100, 1525, 3050, 6104} | set(range(305, 6101, 305))))
DIAGNOSTIC_CONTINUATION = (0, 10, 100, 1525, 6104)
PERSISTENCE_POINTS = (1525, 3050, 6104)  # D persistence points == matching targets
D_MID_POINT = 3050
TRANSIENT_MAX_UPDATE = 1000
MATCH_MAX_BRACKET = 100
FULL_WINDOWS = 4096   # 2,097,152 labels
QUICK_WINDOWS = 512   # first 512 full-dev windows = 262,144 labels
DIAG_WINDOWS = 256    # last 256 training windows per domain = 131,072 labels
WORKER_RESERVE_SECONDS = 360  # workers stop this long before the session work deadline
EVAL_RESERVE_SECONDS = 240    # never start an evaluation burst with less time than this
MINI = False

DECISION_THRESHOLDS = {"prefix_gap": 0.02, "adaptation": 0.05, "proceed_d": 0.03, "proceed_d3050": 0.015,
                       "q_fraction": 0.5, "opposite_d": -0.03, "small_d": 0.015, "transient_d": 0.05}


def use_mini_protocol() -> None:
    """Tiny schedule for local end-to-end plumbing tests only (not science)."""
    global WARMUP_END, CALIBRATION_END, GATE_END, PREFIX_END, CONTINUATION_UPDATES, TRAJECTORY_END
    global FULL_PREFIX, QUICK_CONTINUATION, FULL_CONTINUATION, DIAGNOSTIC_CONTINUATION, PERSISTENCE_POINTS
    global D_MID_POINT, TRANSIENT_MAX_UPDATE, FULL_WINDOWS, QUICK_WINDOWS, DIAG_WINDOWS, MINI
    global WORKER_RESERVE_SECONDS, EVAL_RESERVE_SECONDS
    WORKER_RESERVE_SECONDS, EVAL_RESERVE_SECONDS = 3, 1
    WARMUP_END, CALIBRATION_END, GATE_END, PREFIX_END = 3, 6, 14, 20
    CONTINUATION_UPDATES = 12
    TRAJECTORY_END = PREFIX_END + CONTINUATION_UPDATES
    FULL_PREFIX = (6, 10, 14, 16, 17, 18, 19, 20)
    QUICK_CONTINUATION = (0, 1, 2, 3, 4, 5, 6, 8, 10, 12)
    FULL_CONTINUATION = (0, 1, 3, 6, 9, 12)
    DIAGNOSTIC_CONTINUATION = (0, 3, 12)
    PERSISTENCE_POINTS = (3, 6, 12)
    D_MID_POINT = 6
    TRANSIENT_MAX_UPDATE = 5
    FULL_WINDOWS, QUICK_WINDOWS, DIAG_WINDOWS = 64, 16, 8
    MINI = True


def learning_rate(update: int) -> float:
    if not 1 <= update <= TRAJECTORY_END:
        raise ValueError("Optimizer update out of range")
    if update <= WARMUP_END:
        return 0.0006 * update / WARMUP_END
    return 0.00006 + 0.5 * (0.0006 - 0.00006) * (
        1 + math.cos(math.pi * (update - WARMUP_END) / (TRAJECTORY_END - WARMUP_END)))


def gate(update: int) -> float:
    if not 1 <= update <= TRAJECTORY_END:
        raise ValueError("Optimizer update out of range")
    if update <= CALIBRATION_END:
        return 1.0
    if update >= GATE_END:
        return 0.0
    return 0.5 * (1 + math.cos(math.pi * (update - CALIBRATION_END) / (GATE_END - CALIBRATION_END)))


def stage_limit(stage: str) -> int:
    return PREFIX_END if stage == "prefix" else CONTINUATION_UPDATES


def events_at(stage: str, step: int) -> tuple[list[tuple[str, str]], bool]:
    """(role, domain) evaluations and whether diagnostics run at this stage step."""
    if stage == "prefix":
        roles = (["full"] if step in FULL_PREFIX else []) + (["quick"] if step == PREFIX_END else [])
        diag = step == PREFIX_END
    else:
        roles = (["full"] if step in FULL_CONTINUATION else []) + (["quick"] if step in QUICK_CONTINUATION else [])
        diag = step in DIAGNOSTIC_CONTINUATION
    return [(role, domain) for role in roles for domain in ("web", "python")], diag


def config_document(microbatch: int, accumulation: int, eval_microbatch: int) -> dict:
    return {
        "protocol": "v3-kaggle-fp16-paired-single-notebook-1",
        "runner": RUNNER_VERSION,
        "mini_protocol": MINI,
        "hypothesis": "H1 only: Taper-minus vs RMS, primary allocation (no Taper-plus).",
        "amendments": [
            "2026-09-28 (user): Tesla T4 FP16 autocast + GradScaler, FP32 master weights/moments, FP32 norm/EMA "
            "reductions and FP32 loss; replaces BF16. Not pooled with any BF16 result.",
            "2026-09-30 (user): free Kaggle GPU time across sessions/weekly resets replaces the paid 24h cap; "
            "fixed token budgets, seeds and decision rules unchanged.",
            "2026-10-01 (user): RMS and Taper-minus run as independent workers on two T4s; no shared state.",
            "2026-10-03 (pre-run freeze): microbatch 8 x accumulation 4 (= protocol v3 default; effective batch "
            "32 x 512 unchanged), eval microbatch 16; single-notebook orchestration with self-output resume. "
            "Frozen before any primary-run update.",
            "Weight snapshots: only the full switch state and branch-final weights are kept (storage); every "
            "full-dev point keeps per-document/per-class sufficient statistics as specified.",
        ],
        "seeds": list(SEEDS), "conditions": list(CONDITIONS),
        "model": {"layers": 6, "width": 256, "heads": 4, "mlp": 1024, "context": CONTEXT, "vocab": VOCAB,
                  "tied_embeddings": True, "final_norm": "RMSNorm"},
        "optimizer": {"name": "AdamW", "lr_peak": 0.0006, "lr_final": 0.00006, "betas": [0.9, 0.95], "eps": 1e-8,
                      "matrix_weight_decay": 0.1, "gain_weight_decay": 0.0, "clip": 1.0},
        "precision": "fp16", "microbatch": microbatch, "accumulation": accumulation,
        "eval_microbatch": eval_microbatch,
        "timeline": {"warmup_end": WARMUP_END, "calibration_end": CALIBRATION_END, "gate_end": GATE_END,
                     "prefix_end": PREFIX_END, "continuation_updates": CONTINUATION_UPDATES,
                     "trajectory_end": TRAJECTORY_END},
        "schedule": {"full_prefix": list(FULL_PREFIX), "quick_continuation": list(QUICK_CONTINUATION),
                     "full_continuation": list(FULL_CONTINUATION),
                     "diagnostic_continuation": list(DIAGNOSTIC_CONTINUATION),
                     "persistence_points": list(PERSISTENCE_POINTS), "full_windows": FULL_WINDOWS,
                     "quick_windows": QUICK_WINDOWS, "diag_windows": DIAG_WINDOWS},
        "decision_thresholds": DECISION_THRESHOLDS,
    }


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# =============================================================================
# Data: prepared, hash-verified online arrays and per-seed paired orders.
# =============================================================================
CLASS_CODES = {"W": 0, "A": 1, "P": 2, "X": 3}


class DevSet:
    def __init__(self, online: Path, manifest: dict, domain: str):
        row = manifest["splits"][f"{domain}_dev"]
        self.tokens = np.fromfile(online / row["file"], dtype="<u2")
        owners = np.fromfile(online / row["owners"], dtype="<u4")
        docs = read_json(online / row["documents"])["documents"]
        if len(self.tokens) != FULL_WINDOWS * CONTEXT + 1 and not MINI:
            raise ValueError(f"{domain} dev quota mismatch")
        if len(owners) != len(self.tokens):
            raise ValueError("Owner array misaligned")
        self.doc_ids = [doc["id"] for doc in docs]
        if len(set(self.doc_ids)) != len(self.doc_ids):
            raise ValueError("Duplicate development document ids")
        n = FULL_WINDOWS * CONTEXT
        self.labels = self.tokens[1:n + 1].astype(np.int64)
        self.owners = owners[1:n + 1].astype(np.int64)
        if self.owners.max() >= len(docs):
            raise ValueError("Invalid owners")

    def take(self, indices: np.ndarray) -> np.ndarray:
        offsets = indices[:, None] * CONTEXT + np.arange(CONTEXT + 1)
        return self.tokens[offsets].astype(np.int64)


class TrainStream:
    def __init__(self, path: Path):
        self.tokens = np.fromfile(path, dtype="<u2")  # in RAM: no random-access I/O stalls
        self.windows = (len(self.tokens) - 1) // CONTEXT

    def take(self, indices: np.ndarray) -> np.ndarray:
        offsets = np.asarray(indices, dtype=np.int64)[:, None] * CONTEXT + np.arange(CONTEXT + 1)
        return self.tokens[offsets].astype(np.int64)


def locate_data(inputs: Path) -> tuple[Path, Path]:
    candidates = []
    for path in inputs.rglob("manifest.json"):
        if path.parent.name != "online":
            continue
        try:
            if read_json(path).get("schema") == "pilot-arrays-v1":
                candidates.append(path.parent)
        except Exception:
            continue
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected exactly one prepared 'online' corpus, found {candidates}")
    online = candidates[0]
    orders = online.parent / "orders"
    if not (orders / "manifest.json").exists():
        raise FileNotFoundError("Prepared 'orders' directory missing next to 'online'")
    return online, orders


def verify_data(online: Path, orders: Path) -> dict:
    """Hash every file training/evaluation reads. Reserved test is never referenced."""
    manifest = read_json(online / "manifest.json")
    if manifest.get("schema") != "pilot-arrays-v1":
        raise ValueError("Not the original-protocol arrays")
    if set(manifest["splits"]) != {"web_train", "python_train", "web_dev", "python_dev"}:
        raise ValueError("Online corpus must contain exactly the four train/dev splits")
    if manifest.get("reserved_test_sealed", {}).get("online_path_included") is not False and not MINI:
        raise ValueError("Reserved test isolation not established")
    checked = {}
    for key, row in manifest["splits"].items():
        fields = [("file", "sha256")]
        if key.endswith("dev"):
            fields += [("owners", "owners_sha256"), ("documents", "documents_sha256")]
        for name_field, digest_field in fields:
            path = online / row[name_field]
            actual = sha256_file(path)
            if actual != row[digest_field]:
                raise ValueError(f"Hash mismatch: {path.name}")
            checked[path.name] = actual
    if sha256_file(online / "classes.json") != manifest["classes_sha256"]:
        raise ValueError("Token class table hash mismatch")
    order_manifest = read_json(orders / "manifest.json")
    online_identity = sha256_file(online / "manifest.json")
    if order_manifest["array_manifest_sha256"] != online_identity:
        raise ValueError("Orders were generated for different arrays")
    for seed in SEEDS:
        for key, row in order_manifest["seeds"][str(seed)].items():
            if sha256_file(orders / row["file"]) != row["sha256"]:
                raise ValueError(f"Order hash mismatch: {row['file']}")
    if not MINI:
        for key, minimum in (("web_train", 260_000_000), ("python_train", 110_000_000)):
            if manifest["splits"][key]["tokens"] < minimum:
                raise ValueError(f"Training quota not met: {key}")
    return {"online_manifest_sha256": online_identity,
            "orders_manifest_sha256": sha256_file(orders / "manifest.json"),
            "sources": {k: {"repo": v.get("repo"), "revision": v.get("revision")}
                        for k, v in manifest.get("sources", {}).items()},
            "files_sha256": checked}


class Data:
    def __init__(self, online: Path, orders: Path):
        self.online, self.orders_root = online, orders
        self.manifest = read_json(online / "manifest.json")
        classes = read_json(online / "classes.json")["classes"]
        if len(classes) != VOCAB or not set(classes) <= set("WAPX"):
            raise ValueError("Invalid class table")
        self.class_letters = np.array(classes)
        self.class_codes = np.array([CLASS_CODES[c] for c in classes], dtype=np.int64)
        self.train = {domain: TrainStream(online / self.manifest["splits"][f"{domain}_train"]["file"])
                      for domain in ("web", "python")}
        self.dev = {domain: DevSet(online, self.manifest, domain) for domain in ("web", "python")}
        self.order_manifest = read_json(orders / "manifest.json")

    def seed_orders(self, seed: int) -> dict:
        rows = self.order_manifest["seeds"][str(seed)]
        result = {key: np.load(self.orders_root / row["file"], allow_pickle=False) for key, row in rows.items()}
        if len(result["web"]) < (PREFIX_END + CONTINUATION_UPDATES) * SEQUENCES_PER_UPDATE:
            raise ValueError("Web order too short")
        if len(result["python"]) < CONTINUATION_UPDATES * SEQUENCES_PER_UPDATE:
            raise ValueError("Python order too short")
        result["rare"] = np.asarray(result["rare"], dtype=bool)
        return result


# =============================================================================
# Model (transcribed from src/domain_shift_forgetting/models + training/step.py)
# =============================================================================
import torch  # noqa: E402
from torch import Tensor, nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402


class Condition(str, Enum):
    RMS = "RMS"
    TAPER_MINUS = "Taper-minus"
    TAPER_PLUS = "Taper-plus"


class RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(width))

    def forward(self, h: Tensor, *, update: int = 1, collect: bool = False) -> Tensor:
        with torch.autocast(device_type=h.device.type, enabled=False):
            x = h.float()
            y = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6)
            return (y * self.gamma.float()).to(h.dtype)


class TaperNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(width))
        self.gamma_tilde = nn.Parameter(torch.ones(width))
        for name in ("numerator_ema", "denominator_ema", "pending_numerator", "pending_denominator"):
            self.register_buffer(name, torch.zeros((), dtype=torch.float32))
        for name in ("ema_updates", "pending_positions"):
            self.register_buffer(name, torch.zeros((), dtype=torch.int64))
        self.register_buffer("c", torch.ones((), dtype=torch.float32))
        self.register_buffer("calibrated", torch.tensor(False))

    def forward(self, h: Tensor, *, update: int, collect: bool = False) -> Tensor:
        g = gate(update)
        if update > CALIBRATION_END and not bool(self.calibrated):
            raise RuntimeError("Taper requires completed calibration before the first post-calibration update")
        with torch.autocast(device_type=h.device.type, enabled=False):
            x = h.float()
            if g == 0.0:
                # No internal norm calculation and no graph connection to gamma.
                return (self.c * x * self.gamma_tilde.float()).to(h.dtype)
            mean_square = x.square().mean(dim=-1, keepdim=True) + 1e-6
            r = torch.sqrt(mean_square)
            if collect:
                if update > CALIBRATION_END or bool(self.calibrated):
                    raise RuntimeError("Calibration collection outside warmup")
                with torch.no_grad():
                    energy = (x.detach() * self.gamma.detach().float()).square().sum(dim=-1)
                    self.pending_numerator.add_((energy / r.detach().squeeze(-1)).sum())
                    self.pending_denominator.add_(energy.sum())
                    self.pending_positions.add_(energy.numel())
            normalized = (x * torch.rsqrt(mean_square)) * self.gamma.float()
            if g == 1.0:
                return normalized.to(h.dtype)
            return (g * normalized + (1 - g) * self.c * x * self.gamma_tilde.float()).to(h.dtype)

    @torch.no_grad()
    def finish_update(self, update: int) -> None:
        """Invoke once AFTER optimizer.step; collected means came from pre-step forwards."""
        if update > CALIBRATION_END:
            if int(self.pending_positions):
                raise RuntimeError("Unexpected calibration samples after freeze")
            return
        if int(self.ema_updates) != update - 1 or int(self.pending_positions) <= 0:
            raise RuntimeError("Missing samples or duplicate/out-of-order EMA update")
        self.numerator_ema.mul_(0.99).add_(self.pending_numerator / self.pending_positions, alpha=0.01)
        self.denominator_ema.mul_(0.99).add_(self.pending_denominator / self.pending_positions, alpha=0.01)
        self.ema_updates.add_(1)
        self.pending_numerator.zero_()
        self.pending_denominator.zero_()
        self.pending_positions.zero_()
        if update == CALIBRATION_END:
            correction = 1 - 0.99 ** int(self.ema_updates)
            self.c.copy_((self.numerator_ema / correction) / (self.denominator_ema / correction + 1e-12))
            self.gamma_tilde.copy_(self.gamma)
            self.calibrated.fill_(True)


class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(256, 3 * 256, bias=False)
        self.projection = nn.Linear(256, 256, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(batch, length, 4, 64).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        return self.projection(y.transpose(1, 2).contiguous().view(batch, length, 256))


class Block(nn.Module):
    def __init__(self, condition: Condition) -> None:
        super().__init__()
        norm = RMSNorm if condition == Condition.RMS else TaperNorm
        self.attention_norm = norm(256)
        self.mlp_norm = norm(256)
        self.attention = Attention()
        self.mlp_in = nn.Linear(256, 1024, bias=False)
        self.mlp_out = nn.Linear(1024, 256, bias=False)

    def forward(self, x: Tensor, *, update: int, collect: bool) -> Tensor:
        x = x + self.attention(self.attention_norm(x, update=update, collect=collect))
        z = self.mlp_norm(x, update=update, collect=collect)
        return x + self.mlp_out(F.gelu(self.mlp_in(z), approximate="none"))


class Transformer(nn.Module):
    def __init__(self, condition: Condition) -> None:
        super().__init__()
        self.condition = Condition(condition)
        self.token_embedding = nn.Embedding(VOCAB, 256)
        self.position_embedding = nn.Embedding(CONTEXT, 256)
        self.blocks = nn.ModuleList(Block(self.condition) for _ in range(6))
        self.final_norm = RMSNorm(256)
        self.register_buffer("completed_updates", torch.zeros((), dtype=torch.int64))
        self.register_buffer("target_ema", torch.zeros((), dtype=torch.float32))
        self.register_buffer("target_updates", torch.zeros((), dtype=torch.int64))
        self.register_buffer("pending_rms", torch.zeros((), dtype=torch.float32))
        self.register_buffer("pending_positions", torch.zeros((), dtype=torch.int64))
        self.register_buffer("s_target", torch.zeros((), dtype=torch.float32))
        self.register_buffer("target_frozen", torch.tensor(False))
        self.apply(self._initialize)
        for block in self.blocks:
            nn.init.normal_(block.attention.projection.weight, std=0.02 / math.sqrt(12))
            nn.init.normal_(block.mlp_out.weight, std=0.02 / math.sqrt(12))

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: Tensor, *, update: int | None = None,
                collect: bool = False, auxiliary: bool = False) -> tuple[Tensor, Tensor | None]:
        if tokens.ndim != 2 or not 1 <= tokens.shape[1] <= CONTEXT:
            raise ValueError("Expected batch by sequence tokens")
        completed = int(self.completed_updates)
        update = max(1, completed) if update is None else update
        if collect and (not self.training or update != completed + 1 or update > CALIBRATION_END):
            raise ValueError("Calibration requires the next training update in warmup")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x, update=update, collect=collect)
        logits = F.linear(self.final_norm(x), self.token_embedding.weight)
        return logits, None

    @torch.no_grad()
    def finish_update(self, update: int) -> None:
        if update != int(self.completed_updates) + 1:
            raise ValueError("Updates must finish sequentially")
        for module in self.modules():
            if isinstance(module, TaperNorm):
                module.finish_update(update)
        self.completed_updates.fill_(update)


@torch.no_grad()
def copy_canonical_initialization(canonical: Transformer, destination: Transformer) -> None:
    """Explicitly pair all shared tensors; call before constructing optimizers."""
    if canonical.condition != Condition.RMS or int(canonical.completed_updates) != 0 \
            or int(destination.completed_updates) != 0:
        raise ValueError("Require fresh RMS canonical and destination models")
    shared = dict(canonical.named_parameters())
    copied = set()
    for name, parameter in destination.named_parameters():
        if name.endswith("gamma_tilde"):
            parameter.fill_(1.0)
            continue
        if name not in shared or parameter.shape != shared[name].shape:
            raise ValueError(f"Canonical tensor mismatch: {name}")
        parameter.copy_(shared[name])
        copied.add(name)
    if copied != set(shared):
        raise ValueError("Not every canonical parameter was copied")


def make_state(seed: int, condition: str, device: torch.device):
    """Paired canonical initialization (pilot_runner.make_state)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    canonical = Transformer(Condition.RMS)
    model = canonical if condition == "RMS" else Transformer(Condition(condition))
    if model is not canonical:
        copy_canonical_initialization(canonical, model)
        del canonical
    model.to(device)
    matrices, gains = [], []
    for parameter in model.parameters():
        if parameter.dtype != torch.float32:
            raise ValueError("Protocol requires FP32 master parameters")
        (matrices if parameter.ndim >= 2 else gains).append(parameter)
    optimizer = torch.optim.AdamW([{"params": matrices, "weight_decay": 0.1},
                                   {"params": gains, "weight_decay": 0.0}],
                                  lr=0.0006, betas=(0.9, 0.95), eps=1e-8, fused=device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda", init_scale=128)
    return model, optimizer, scaler


def ordered_update(model, optimizer, scaler, stream: TrainStream, indices: np.ndarray,
                   microbatch: int, device: torch.device) -> dict:
    """One effective update of exactly 32 x 512 labels (pilot_runner.ordered_update)."""
    accumulation = SEQUENCES_PER_UPDATE // microbatch
    if len(indices) != SEQUENCES_PER_UPDATE or microbatch * accumulation != SEQUENCES_PER_UPDATE:
        raise ValueError("Exactly 16,384 labels required per completed update")
    u = int(model.completed_updates) + 1
    for group in optimizer.param_groups:
        group["lr"] = learning_rate(u)
    model.train()
    fp16 = scaler.is_enabled()
    batches = [torch.from_numpy(stream.take(indices[s:s + microbatch])).to(device, non_blocking=True)
               for s in range(0, SEQUENCES_PER_UPDATE, microbatch)]
    for attempt in range(6):
        optimizer.zero_grad(set_to_none=True)
        for name, buffer in model.named_buffers():
            if "pending_" in name:
                buffer.zero_()
        ce_total = torch.zeros((), device=device)
        for rows in batches:
            with torch.autocast(device.type, dtype=torch.float16, enabled=fp16):
                logits, _ = model(rows[:, :-1], update=u, collect=u <= CALIBRATION_END)
                ce = F.cross_entropy(logits.float().reshape(-1, VOCAB), rows[:, 1:].reshape(-1))
            if not bool(torch.isfinite(ce)):
                raise FloatingPointError(f"Nonfinite forward CE at global update {u}")
            scaler.scale(ce / accumulation).backward()
            ce_total += ce.detach() / accumulation
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
        if not bool(torch.isfinite(norm)):
            if not fp16:
                raise FloatingPointError("Nonfinite gradients")
            scaler.update(new_scale=scaler.get_scale() / 2)
            continue  # identical fixed data; no cursor or calibration clock advanced
        scaler.step(optimizer)
        scaler.update()
        if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise FloatingPointError(f"Nonfinite weights after optimizer step at update {u}")
        model.finish_update(u)
        return {"u": u, "ce": float(ce_total), "gn": float(norm), "clip": float(norm) > 1.0,
                "retry": attempt, "scale": float(scaler.get_scale()) if fp16 else None}
    raise FloatingPointError("FP16 overflow persisted after six retries")


# =============================================================================
# Evaluation: exact token-weighted sufficient statistics (vectorized).
# =============================================================================
@torch.no_grad()
def evaluate(model, data: Data, domain: str, role: str, rare: np.ndarray, eval_microbatch: int,
             device: torch.device) -> dict:
    dev = data.dev[domain]
    windows = FULL_WINDOWS if role == "full" else QUICK_WINDOWS
    n = windows * CONTEXT
    model.eval()
    losses = torch.empty(n, dtype=torch.float32, device=device)
    fp16 = device.type == "cuda"
    for start in range(0, windows, eval_microbatch):
        stop = min(windows, start + eval_microbatch)
        rows = torch.from_numpy(dev.take(np.arange(start, stop))).to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=fp16):
            logits, _ = model(rows[:, :-1], collect=False, auxiliary=False)
            ce = F.cross_entropy(logits.float().reshape(-1, VOCAB), rows[:, 1:].reshape(-1), reduction="none")
        losses[start * CONTEXT:stop * CONTEXT] = ce
    values = losses.double().cpu().numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise FloatingPointError(f"Nonfinite evaluation CE ({domain}/{role})")
    labels = dev.labels[:n]
    classes = data.class_codes[labels]
    class_sums = np.bincount(classes, weights=values, minlength=4)
    class_counts = np.bincount(classes, minlength=4)
    rare_mask = rare[labels]
    total = math.fsum(values)
    record = {"domain": domain, "role": role,
              "array_sha256": data.manifest["splits"][f"{domain}_dev"]["sha256"],
              "total": [total, n],
              "classes": {letter: [float(class_sums[code]), int(class_counts[code])]
                          for letter, code in CLASS_CODES.items()},
              "rare": [float(values[rare_mask].sum()), int(rare_mask.sum())]}
    record["ce"] = total / n
    apx = sum(record["classes"][c][1] for c in "APX")
    ap = sum(record["classes"][c][1] for c in "AP")
    record["non_w_ce"] = sum(record["classes"][c][0] for c in "APX") / apx if apx else None
    record["ap_ce"] = sum(record["classes"][c][0] for c in "AP") / ap if ap else None
    if role == "full":
        owners = dev.owners[:n]
        record["doc_sums"] = np.bincount(owners, weights=values, minlength=len(dev.doc_ids)).tolist()
        record["doc_counts"] = np.bincount(owners, minlength=len(dev.doc_ids)).tolist()
    return record


def sensitivity_mask(inputs: Tensor, *, eos: int = EOS) -> Tensor:
    mask = torch.ones_like(inputs, dtype=torch.bool)
    mask[:, :16] = False
    mask[:, 1:] &= inputs[:, :-1] != eos
    return mask


@torch.no_grad()
def diagnostics(model, data: Data, rare: np.ndarray, microbatch: int, device: torch.device) -> dict:
    """Forward-only probes (pilot_diagnostics.diagnostics). Descriptive, not a decision input."""
    model.eval()
    norm_samples, ratios, attention = {}, {}, {}
    current = {"domain": None, "mask": None, "attention": False}
    handles = []
    fp16 = device.type == "cuda"

    def collect(name, tensor):
        squared = tensor.detach().float().square().sum(-1)
        for view, values in (("all", squared.flatten()), ("position_mask", squared[current["mask"]])):
            norm_samples.setdefault((current["domain"], name, view), []).append(values.cpu())

    for index, block in enumerate(model.blocks):
        for branch, norm in (("attention", block.attention_norm), ("mlp", block.mlp_norm)):
            def hook(module, inputs, output, name=f"{index}.{branch}"):
                collect(name + ".h", inputs[0])
                collect(name + ".z", output)
            handles.append(norm.register_forward_hook(hook))
        residual, mlp_residual = {}, {}

        def pre_attention(module, inputs, stash=residual):
            stash["h"] = inputs[0].detach().float().square().mean().sqrt()

        def post_attention(module, inputs, output, stash=residual, name=f"{index}.attention"):
            denominator = float(stash["h"])
            numerator = float(output.detach().float().square().mean().sqrt())
            ratios.setdefault((current["domain"], name), []).append(numerator / denominator if denominator else None)

        handles.append(block.attention_norm.register_forward_pre_hook(pre_attention))
        handles.append(block.attention.register_forward_hook(post_attention))
        handles.append(block.mlp_norm.register_forward_pre_hook(lambda module, inputs, stash=mlp_residual:
                       stash.update(h=inputs[0].detach().float().square().mean().sqrt())))

        def post_mlp(module, inputs, output, stash=mlp_residual, name=f"{index}.mlp"):
            denominator = float(stash["h"])
            numerator = float(output.detach().float().square().mean().sqrt())
            ratios.setdefault((current["domain"], name), []).append(numerator / denominator if denominator else None)
        handles.append(block.mlp_out.register_forward_hook(post_mlp))

        def entropy(module, inputs, output, name=f"{index}.attention"):
            if not current["attention"]:
                return
            batch, length, _ = output.shape
            with torch.autocast(device.type, enabled=False):
                q, k, _ = output.float().chunk(3, dim=-1)
                q, k = (x.view(batch, length, 4, 64).transpose(1, 2) for x in (q, k))
                scores = q @ k.transpose(-2, -1) / 8
                mask = torch.ones(length, length, dtype=torch.bool, device=device).triu(1)
                scores.masked_fill_(mask, -torch.inf)
                logp = scores.log_softmax(-1)
                values = (-(logp.exp() * logp.masked_fill(mask, 0)).sum(-1).mean((0, 2))).cpu().tolist()
            attention.setdefault((current["domain"], name), []).append(values)
        handles.append(block.attention.qkv.register_forward_hook(entropy))
    try:
        for domain in ("web", "python"):
            current["domain"] = domain
            stream = data.train[domain]
            indices = np.arange(stream.windows - DIAG_WINDOWS, stream.windows)
            for start in range(0, len(indices), microbatch):
                rows = torch.from_numpy(stream.take(indices[start:start + microbatch])).to(device)
                current["mask"] = sensitivity_mask(rows[:, :-1])
                current["attention"] = False
                with torch.autocast(device.type, dtype=torch.float16, enabled=fp16):
                    model(rows[:, :-1], collect=False)
            current["attention"] = True
            rows = torch.from_numpy(stream.take(indices[:2])).to(device)
            current["mask"] = sensitivity_mask(rows[:, :-1])
            with torch.autocast(device.type, dtype=torch.float16, enabled=fp16):
                model(rows[:, :-1], collect=False)
        summaries = {}
        for (domain, site, view), chunks in norm_samples.items():
            values = torch.cat(chunks[:-1])  # exclude the entropy probe's contribution
            finite = bool(torch.isfinite(values).all())
            norms = values.sqrt()
            p50, p99 = (torch.quantile(norms, torch.tensor([0.5, 0.99])).tolist() if finite else (None, None))
            summaries.setdefault(site, {}).setdefault(view, {})[domain] = {
                "positions": len(values), "finite": finite, "mean_squared_norm": float(values.mean()),
                "norm_p50": p50, "norm_p99": p99, "p99_over_p50": (p99 / p50) if p50 else None}
        for site in summaries.values():
            for view in site.values():
                web, code = view["web"]["mean_squared_norm"], view["python"]["mean_squared_norm"]
                view["log_k"] = 0.5 * math.log(max(web, 1e-12) / max(code, 1e-12)) \
                    if math.isfinite(web) and math.isfinite(code) else None
                view["near_zero_energy"] = min(web, code) <= 1e-12
        gains = {name: {"min": float(p.min()), "max": float(p.max()), "rms": float(p.float().square().mean().sqrt())}
                 for name, p in model.named_parameters() if p.ndim == 1}
        c = {name: float(b) for name, b in model.named_buffers() if name.endswith(".c")}
        embeddings = model.token_embedding.weight.float().norm(dim=-1).cpu().numpy()
        groups = [(k, data.class_letters == k) for k in "WAPX"] + [("R", np.asarray(rare))]
        embedding_groups = {key: {"count": int(mask.sum()),
                                  "mean_norm": float(embeddings[mask].mean()) if mask.any() else None}
                            for key, mask in groups}
        return {"probe": "last training windows of each stream; may overlap trained documents",
                "forward_labels_per_domain": DIAG_WINDOWS * CONTEXT, "sites": summaries,
                "branch_residual_rms_ratios": {f"{a}/{b}": float(np.mean([v for v in x[:-1] if v is not None]))
                                               if any(v is not None for v in x[:-1]) else None
                                               for (a, b), x in ratios.items()},
                "attention_entropy_nats_per_head": {f"{a}/{b}": np.mean(x, axis=0).tolist()
                                                    for (a, b), x in attention.items()},
                "gains": gains, "calibration_c": c, "embedding_norms": embedding_groups}
    finally:
        for handle in handles:
            handle.remove()


# =============================================================================
# Persistent per-run state
# =============================================================================
class EventLog:
    """Append-only JSONL of evaluation/diagnostic records; a torn last line is ignored."""

    def __init__(self, path: Path):
        self.path = path
        self.records: dict[tuple, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.records[self.key(record)] = record

    @staticmethod
    def key(record: dict) -> tuple:
        return record["stage"], record["step"], record["role"], record["domain"]

    def has(self, stage, step, role, domain) -> bool:
        return (stage, step, role, domain) in self.records

    def get(self, stage, step, role, domain):
        return self.records.get((stage, step, role, domain))

    def add(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.records[self.key(record)] = record


def save_checkpoint(path: Path, model, optimizer, scaler, progress: dict, identity: dict) -> str:
    for name, buffer in model.named_buffers():
        if "pending_" in name and bool(torch.any(buffer != 0)):
            raise ValueError("Cannot checkpoint inside an unfinished update")
    payload = {"schema": "h1-ckpt-1", "identity": identity, "progress": progress,
               "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
               "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                       "torch_cpu": torch.get_rng_state(),
                       "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    digest = sha256_file(path)
    write_json(path.with_name(path.name + ".json"), {"sha256": digest, "progress": progress,
                                                     "global_update": int(model.completed_updates)})
    return digest


def load_checkpoint(path: Path, model, optimizer, scaler, identity: dict) -> dict:
    sidecar = path.with_name(path.name + ".json")
    if sidecar.exists() and sha256_file(path) != read_json(sidecar)["sha256"]:
        raise ValueError(f"Checkpoint checksum mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["schema"] != "h1-ckpt-1" or payload["identity"] != identity:
        raise ValueError(f"Checkpoint identity mismatch: {path}")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    optimizer.zero_grad(set_to_none=True)
    scaler.load_state_dict(payload["scaler"])
    random.setstate(payload["rng"]["python"])
    np.random.set_state(payload["rng"]["numpy"])
    torch.set_rng_state(payload["rng"]["torch_cpu"])
    if payload["rng"]["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(payload["rng"]["torch_cuda"])
    return payload["progress"]


def save_weights(path: Path, model, identity: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save({"identity": identity, "global_update": int(model.completed_updates),
                "model": model.state_dict()}, temporary)
    os.replace(temporary, path)


# =============================================================================
# Worker: one condition on one GPU, seeds in protocol order.
# =============================================================================
class Stop(Exception):
    pass


def run_job(seed: int, condition: str, state: Path, data: Data, cfg: dict, deadline: float,
            stop_flag: list, device: torch.device) -> str:
    run_dir = state / "runs" / f"S{seed}-{condition}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "completion.json").exists():
        return "complete"
    if (run_dir / "failure.json").exists():
        raise RuntimeError(f"S{seed}-{condition} has a recorded failure; investigate before continuing")
    identity = {"config_sha256": cfg["config_sha256"], "seed": seed, "condition": condition}
    orders = data.seed_orders(seed)
    model, optimizer, scaler = make_state(seed, condition, device)
    events = EventLog(run_dir / "events.jsonl")
    progress = {"stage": "prefix", "step": 0, "optimizer_seconds": 0.0, "updates": 0, "clipped": 0,
                "overflow_retries": 0, "parent_sha256": None}
    if (run_dir / "latest.pt").exists():
        progress = load_checkpoint(run_dir / "latest.pt", model, optimizer, scaler, identity)
        log(f"S{seed}-{condition}: resumed at {progress['stage']} step {progress['step']} "
            f"(global update {int(model.completed_updates)})")
    else:
        log(f"S{seed}-{condition}: fresh paired initialization")
    microbatch, eval_microbatch = cfg["microbatch"], cfg["eval_microbatch"]
    last_save = time.time()
    window_t, window_n = time.time(), 0

    def checkpoint(name="latest.pt") -> str:
        nonlocal last_save
        digest = save_checkpoint(run_dir / name, model, optimizer, scaler, progress, identity)
        last_save = time.time()
        return digest

    train_log = (run_dir / "train.jsonl").open("a", encoding="utf-8")
    try:
        while True:
            stage, step = progress["stage"], progress["step"]
            if stop_flag[0] or time.time() > deadline:
                checkpoint()
                raise Stop()
            evals, diag = events_at(stage, step)
            pending = [(role, domain) for role, domain in evals if not events.has(stage, step, role, domain)]
            diag_pending = diag and not events.has(stage, step, "diag", "both")
            if pending or diag_pending:
                if time.time() > deadline - EVAL_RESERVE_SECONDS:  # never start an eval burst without reserve
                    checkpoint()
                    raise Stop()
                for role, domain in pending:
                    tick = time.time()
                    record = evaluate(model, data, domain, role, orders["rare"], eval_microbatch, device)
                    record.update(stage=stage, step=step, global_update=int(model.completed_updates),
                                  parent_sha256=progress["parent_sha256"], seconds=time.time() - tick)
                    events.add(record)
                if diag_pending:
                    tick = time.time()
                    try:
                        measured = diagnostics(model, data, orders["rare"], eval_microbatch, device)
                        status = "ok"
                    except Exception as exc:  # descriptive only; never blocks the primary endpoint
                        measured, status = {"error": f"{type(exc).__name__}: {exc}"}, "error"
                    events.add({"stage": stage, "step": step, "role": "diag", "domain": "both",
                                "global_update": int(model.completed_updates), "status": status,
                                "measurement": measured, "seconds": time.time() - tick})
                if step in (FULL_PREFIX if stage == "prefix" else FULL_CONTINUATION):
                    web_full = events.get(stage, step, "full", "web")
                    code_full = events.get(stage, step, "full", "python")
                    log(f"S{seed}-{condition} {stage}@{step}: full-dev web CE {web_full['ce']:.4f}, "
                        f"code non-W CE {code_full['non_w_ce']:.4f}")
            if step == stage_limit(stage):
                if stage == "prefix":
                    if not (run_dir / "switch.pt").exists():
                        checkpoint("switch.pt")
                    next_stage = "web"
                elif stage == "web":
                    save_weights(run_dir / "web-final.pt", model, identity)
                    next_stage = "python"
                else:
                    save_weights(run_dir / "python-final.pt", model, identity)
                    write_json(run_dir / "completion.json", {
                        "seed": seed, "condition": condition, "config_sha256": cfg["config_sha256"],
                        "switch_sha256": read_json(run_dir / "switch.pt.json")["sha256"],
                        "updates": progress["updates"], "optimizer_seconds": progress["optimizer_seconds"],
                        "clipped": progress["clipped"], "overflow_retries": progress["overflow_retries"],
                        "completed_unix": time.time()})
                    for name in ("latest.pt", "latest.pt.json"):
                        (run_dir / name).unlink(missing_ok=True)
                    log(f"S{seed}-{condition}: COMPLETE")
                    return "complete"
                counters = {k: progress[k] for k in ("optimizer_seconds", "updates", "clipped", "overflow_retries")}
                switch_sha = read_json(run_dir / "switch.pt.json")["sha256"]
                load_checkpoint(run_dir / "switch.pt", model, optimizer, scaler, identity)
                progress = {**counters, "stage": next_stage, "step": 0, "parent_sha256": switch_sha}
                checkpoint()
                log(f"S{seed}-{condition}: branch '{next_stage}' starts from switch {switch_sha[:12]}")
                continue
            if stage == "prefix":
                domain, offset = "web", step * SEQUENCES_PER_UPDATE
            elif stage == "web":
                domain, offset = "web", (PREFIX_END + step) * SEQUENCES_PER_UPDATE
            else:
                domain, offset = "python", step * SEQUENCES_PER_UPDATE
            indices = np.asarray(orders[domain][offset:offset + SEQUENCES_PER_UPDATE], dtype=np.int64)
            tick = time.time()
            metrics = ordered_update(model, optimizer, scaler, data.train[domain], indices, microbatch, device)
            elapsed = time.time() - tick
            progress["step"] += 1
            progress["updates"] += 1
            progress["optimizer_seconds"] += elapsed
            progress["clipped"] += int(metrics["clip"])
            progress["overflow_retries"] += metrics["retry"]
            train_log.write(json.dumps({"stage": stage, "step": progress["step"], **metrics,
                                        "sec": round(elapsed, 4)}, separators=(",", ":")) + "\n")
            window_n += 1
            if window_n >= 250:
                train_log.flush()
                rate = window_n * TOKENS_PER_UPDATE / (time.time() - window_t)
                log(f"S{seed}-{condition} {stage} {progress['step']}/{stage_limit(stage)} "
                    f"u={metrics['u']} ce={metrics['ce']:.4f} gn={metrics['gn']:.3f} "
                    f"scale={metrics['scale']} {rate:,.0f} tok/s (incl. eval)")
                window_t, window_n = time.time(), 0
            if time.time() - last_save >= cfg["checkpoint_seconds"]:
                checkpoint()
    finally:
        train_log.close()


def worker_main(args) -> int:
    state = Path(args.state)
    cfg = read_json(state / "config.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    torch.set_num_threads(2)
    stop_flag = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop_flag.__setitem__(0, True))
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    log(f"worker {args.conditions} on {gpu} (torch {torch.__version__}); deadline in "
        f"{(args.deadline - time.time()) / 3600:.2f} h")
    data = Data(Path(args.online), Path(args.orders))
    jobs = [(seed, condition) for seed in SEEDS for condition in args.conditions.split(",")]
    for seed, condition in jobs:
        try:
            status = run_job(seed, condition, state, data, cfg, args.deadline, stop_flag, device)
        except Stop:
            log(f"worker {args.conditions}: paused at deadline/stop; state saved")
            return 0
        except FloatingPointError as exc:
            write_json(state / "runs" / f"S{seed}-{condition}" / "failure.json", {
                "type": "numerical", "message": str(exc), "traceback": traceback.format_exc(),
                "action": "Numerical failure: invalidates a clean H1 interpretation. Investigate; never replace "
                          "the seed or change thresholds."})
            log(f"NUMERICAL FAILURE S{seed}-{condition}: {exc}")
            return 3
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if status != "complete":
            return 0
    log(f"worker {args.conditions}: all assigned runs complete")
    return 0


# =============================================================================
# Analysis (transcribed from src/domain_shift_forgetting/analysis + evaluation/statistics)
# =============================================================================
@dataclass
class LossSum:
    ce_sum: float = 0.0
    count: int = 0

    @property
    def mean(self) -> float | None:
        return self.ce_sum / self.count if self.count else None


@dataclass
class EvaluationSums:
    total: LossSum = field(default_factory=LossSum)
    documents: dict = field(default_factory=dict)
    classes: dict = field(default_factory=lambda: {c: LossSum() for c in "WAPX"})
    rare: LossSum = field(default_factory=LossSum)

    def code_ce(self, *, alphanumeric_punctuation_only: bool = False) -> float | None:
        included = "AP" if alphanumeric_punctuation_only else "APX"
        count = sum(self.classes[c].count for c in included)
        return sum(self.classes[c].ce_sum for c in included) / count if count else None

    def check_reconstruction(self, atol: float = 1e-6) -> None:
        for groups in (self.documents.values(), self.classes.values()):
            rows = list(groups)
            if sum(row.count for row in rows) != self.total.count:
                raise ValueError("Counts do not reconstruct corpus labels")
            if self.total.count and abs(sum(r.ce_sum for r in rows) - self.total.ce_sum) / self.total.count > atol:
                raise ValueError("CE sums do not reconstruct corpus CE")


@dataclass(frozen=True)
class BranchCE:
    rms_prefix: float
    taper_prefix: float
    rms_web: float
    rms_python: float
    taper_web: float
    taper_python: float

    def __post_init__(self) -> None:
        if any(not math.isfinite(x) or x < 0 for x in self.__dict__.values()):
            raise ValueError("All branch values must be finite nonnegative web CE")


@dataclass(frozen=True)
class Contrast:
    d: float
    g_web: float
    q: float
    f_rms_web: float
    f_rms_python: float
    f_taper_web: float
    f_taper_python: float


def contrast(ce: BranchCE) -> Contrast:
    d = (ce.taper_python - ce.taper_web) - (ce.rms_python - ce.rms_web)
    g = (ce.taper_prefix - ce.rms_prefix) - (ce.taper_web - ce.rms_web)
    frw, frp = ce.rms_web - ce.rms_prefix, ce.rms_python - ce.rms_prefix
    ftw, ftp = ce.taper_web - ce.taper_prefix, ce.taper_python - ce.taper_prefix
    q = d - g
    if not math.isclose(q, ftp - frp, rel_tol=0, abs_tol=1e-6):
        raise ValueError("Forgetting decomposition does not reconstruct")
    return Contrast(d, g, q, frw, frp, ftw, ftp)


@dataclass(frozen=True)
class SeedUncertainty:
    values: tuple
    mean: float
    sample_sd: float
    descriptive_t95: tuple


def seed_uncertainty(values: Sequence[float]) -> SeedUncertainty:
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError("The descriptive interval requires all three finite paired seeds")
    mean, sd = statistics.mean(values), statistics.stdev(values)
    halfwidth = 4.303 * sd / math.sqrt(3)
    return SeedUncertainty(tuple(values), mean, sd, (mean - halfwidth, mean + halfwidth))


def prefix_slope(points: Sequence[tuple[int, float]]) -> float:
    """OLS CE slope in nats per million supervised tokens, last five observations."""
    if len(points) < 5 or any(points[i][0] >= points[i + 1][0] for i in range(len(points) - 1)):
        raise ValueError("Need at least five chronological prefix points")
    tail = points[-5:]
    xs = [u * TOKENS_PER_UPDATE / 1e6 for u, _ in tail]
    ys = [y for _, y in tail]
    xm, ym = statistics.mean(xs), statistics.mean(ys)
    return sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sum((x - xm) ** 2 for x in xs)


@dataclass(frozen=True)
class AdaptationPoint:
    update: int
    non_w_code_ce: float
    web_ce: float

    def __post_init__(self) -> None:
        if not 0 <= self.update <= CONTINUATION_UPDATES or any(
                not math.isfinite(v) or v < 0 for v in (self.non_w_code_ce, self.web_ce)):
            raise ValueError("Invalid adaptation observation")


@dataclass(frozen=True)
class Match:
    target_code_ce: float
    left: AdaptationPoint
    right: AdaptationPoint
    fraction: float
    update: float
    web_ce: float
    nearest_endpoint_web_ce: float


def first_crossing(points: Sequence[AdaptationPoint], target: float) -> Match | None:
    if not points or points[0].update != 0:
        raise ValueError("Matching needs the s=0 baseline")
    if any(a.update >= b.update for a, b in zip(points, points[1:])):
        raise ValueError("Observations must be strictly chronological")
    if points[0].non_w_code_ce < target:
        return None  # already surpassed; do not manufacture a post-switch match
    for index, point in enumerate(points):
        if point.non_w_code_ce == target:
            return Match(target, point, point, 0.0, float(point.update), point.web_ce, point.web_ce)
        if index == 0:
            continue
        left = points[index - 1]
        if left.non_w_code_ce > target > point.non_w_code_ce:
            if point.update - left.update > MATCH_MAX_BRACKET:
                return None
            fraction = (left.non_w_code_ce - target) / (left.non_w_code_ce - point.non_w_code_ce)
            web = left.web_ce + fraction * (point.web_ce - left.web_ce)
            update = left.update + fraction * (point.update - left.update)
            nearest = left if fraction <= 0.5 else point
            return Match(target, left, point, fraction, update, web, nearest.web_ce)
    return None


def matched_forgetting(rms: Sequence[AdaptationPoint], taper: Sequence[AdaptationPoint], target_update: int):
    if target_update not in PERSISTENCE_POINTS:
        raise ValueError("Target must be a frozen RMS checkpoint")
    targets = [p for p in rms if p.update == target_update]
    if len(targets) != 1:
        raise ValueError("Missing or duplicate RMS target observation")
    target = targets[0]
    match = first_crossing(taper, target.non_w_code_ce)
    if match is None:
        return None, None
    difference = (match.web_ce - taper[0].web_ce) - (target.web_ce - rms[0].web_ce)
    return match, difference


@dataclass(frozen=True)
class DocumentEffect:
    document_id: str
    count: int
    d: float


@dataclass(frozen=True)
class ClassEffect:
    token_class: str
    count: int
    mean_d: float | None
    weighted_contribution: float


def _four_way(tp: LossSum, tw: LossSum, rp: LossSum, rw: LossSum) -> float | None:
    if len({r.count for r in (tp, tw, rp, rw)}) != 1:
        raise ValueError("Different scored-label populations across conditions/branches")
    if not tp.count:
        return None
    return ((tp.ce_sum - tw.ce_sum) - (rp.ce_sum - rw.ce_sum)) / tp.count


def decompose(taper_python: EvaluationSums, taper_web: EvaluationSums,
              rms_python: EvaluationSums, rms_web: EvaluationSums):
    groups = (taper_python, taper_web, rms_python, rms_web)
    for group in groups:
        group.check_reconstruction()
    if any(set(g.documents) != set(taper_python.documents) for g in groups):
        raise ValueError("Different document populations across branches")
    d = _four_way(*(g.total for g in groups))
    documents = [DocumentEffect(doc, taper_python.documents[doc].count,
                                _four_way(*(g.documents[doc] for g in groups)))
                 for doc in sorted(taper_python.documents)]
    classes = []
    for cls in "WAPX":
        effect = _four_way(*(g.classes[cls] for g in groups))
        count = taper_python.classes[cls].count
        classes.append(ClassEffect(cls, count, effect, (effect or 0.0) * count / taper_python.total.count))
    if abs(sum(c.weighted_contribution for c in classes) - d) > 1e-6:
        raise ValueError("Disjoint classes do not reconstruct D")
    return d, tuple(documents), tuple(classes)


@dataclass(frozen=True)
class TailSummary:
    d: float
    unweighted_median: float
    trimmed_token_weighted_mean: float
    trim_count_each_end: int
    top_signed_sum: float
    top_fraction_of_net: float | None
    top_fraction_of_positive_mass: float | None
    outlier_concentrated: bool
    top_contributors: tuple


def summarize_documents(rows: Sequence[DocumentEffect], expected_d: float) -> TailSummary:
    total = sum(r.count for r in rows)
    d = math.fsum(r.count * r.d for r in rows) / total
    if abs(d - expected_d) > 1e-6:
        raise ValueError("Document contributions do not reconstruct full-dev D")
    ranked = sorted(rows, key=lambda r: (r.d, r.document_id))
    trim = math.floor(0.01 * len(rows))
    retained = ranked[trim:len(rows) - trim]
    trimmed = math.fsum(r.count * r.d for r in retained) / sum(r.count for r in retained)
    contributions = sorted(((r.document_id, r.count * r.d / total) for r in rows), key=lambda x: (-x[1], x[0]))
    top_n = math.ceil(0.01 * len(rows))
    top = math.fsum(c for _, c in contributions[:top_n])
    positive = math.fsum(max(c, 0.0) for _, c in contributions)
    return TailSummary(d, statistics.median(r.d for r in rows), trimmed, trim, top,
                       top / d if d > 0 else None, top / positive if positive else None,
                       d >= 0.015 and top > 0.5 * d, tuple(contributions[:top_n]))


@dataclass(frozen=True)
class SeedEvidence:
    seed: int
    d: float
    d3050: float
    q: float
    absolute_relative_prefix_gap: float
    rms_non_w_improvement: float
    taper_non_w_improvement: float
    quick_d: Mapping[int, float]
    matched_differences: Mapping[int, float | None]


@dataclass(frozen=True)
class Decision:
    category: str
    reasons: tuple
    common_target_update: int | None = None
    statistics: Mapping[str, float] = field(default_factory=dict)


def classify(seeds: Sequence[SeedEvidence], *, primary_complete: bool, correctness_and_data_passed: bool,
             nonfinite_primary: bool = False, resource_terminated: bool = False,
             thresholds: Mapping[str, float] = DECISION_THRESHOLDS) -> Decision:
    t = thresholds
    invalid = []
    if not primary_complete or len(seeds) != 3 or {s.seed for s in seeds} != set(SEEDS):
        invalid.append("Missing or duplicate planned primary seed")
    if not correctness_and_data_passed:
        invalid.append("Unresolved correctness/data gate")
    if nonfinite_primary or resource_terminated:
        invalid.append("Nonfinite primary run or resource termination")
    for seed in seeds:
        required = (seed.d, seed.d3050, seed.q, seed.absolute_relative_prefix_gap,
                    seed.rms_non_w_improvement, seed.taper_non_w_improvement)
        if not all(math.isfinite(v) for v in required) or seed.absolute_relative_prefix_gap < 0:
            invalid.append(f"Invalid primary evidence for seed {seed.seed}")
        if set(seed.quick_d) != set(QUICK_CONTINUATION) or not all(math.isfinite(v) for v in seed.quick_d.values()):
            invalid.append(f"Missing/nonfinite quick-dev evidence for seed {seed.seed}")
        if set(seed.matched_differences) != set(PERSISTENCE_POINTS) or any(
                v is not None and not math.isfinite(v) for v in seed.matched_differences.values()):
            invalid.append(f"Missing/nonfinite matching records for seed {seed.seed}")
    if invalid:
        return Decision("INVALID OR INCOMPLETE", tuple(invalid))
    limited = [f"Primary guardrail failed for seed {s.seed}" for s in seeds
               if s.absolute_relative_prefix_gap > t["prefix_gap"] or s.rms_non_w_improvement < t["adaptation"]
               or s.taper_non_w_improvement < t["adaptation"]]
    if limited:
        return Decision("COMPARABILITY / ADAPTATION LIMITED", tuple(limited))
    mean_d = statistics.mean(s.d for s in seeds)
    stats = {"mean_D": mean_d, "mean_D3050": statistics.mean(s.d3050 for s in seeds),
             "mean_Q": statistics.mean(s.q for s in seeds)}
    common = [u for u in PERSISTENCE_POINTS if all(s.matched_differences[u] is not None for s in seeds)]
    target = max(common) if common else None
    if target is not None:
        stats["mean_matched_difference"] = statistics.mean(float(s.matched_differences[target]) for s in seeds)
    if mean_d <= t["opposite_d"]:
        return Decision("OPPOSITE DIRECTION", (f"Mean D <= {t['opposite_d']}",), target, stats)
    if (mean_d >= t["proceed_d"] and all(s.d > 0 for s in seeds) and stats["mean_D3050"] >= t["proceed_d3050"]
            and stats["mean_Q"] > t["q_fraction"] * mean_d and target is not None
            and stats["mean_matched_difference"] > 0):
        return Decision("PROCEED TO DESIGN THE NEXT STUDY",
                        ("All pilot screening criteria met; no confirmatory claim",), target, stats)
    if mean_d < t["small_d"]:
        transient = any(statistics.mean(s.quick_d[u] for s in seeds) >= t["transient_d"]
                        for u in QUICK_CONTINUATION if u <= TRANSIENT_MAX_UPDATE)
        category = "TRANSIENT ONLY" if transient else "STOP — SMALL OBSERVED EFFECT"
        return Decision(category, (f"Mean endpoint D < {t['small_d']}; not evidence of equivalence",), target, stats)
    return Decision("INCONCLUSIVE", ("Screening criteria not all met",), target, stats)


H1_ANSWERS = {
    "PROCEED TO DESIGN THE NEXT STUDY": "YES - H1 supported at the pilot-screening level: Taper-minus shows more "
                                        "persistent web forgetting after the Python switch than RMS, meeting every "
                                        "pre-registered criterion (not a confirmatory claim).",
    "OPPOSITE DIRECTION": "NO - the effect goes the other way: Taper-minus forgets LESS than RMS (mean D <= -0.03).",
    "STOP — SMALL OBSERVED EFFECT": "NO - no persistent excess forgetting >= 0.015 nats/token detected for "
                                    "Taper-minus (pilot stop; not proof of equivalence).",
    "TRANSIENT ONLY": "NO (for persistent forgetting) - only a transient early excess (<= update 1000) that does not "
                      "persist to the endpoint.",
    "INCONCLUSIVE": "UNDECIDED - valid, comparable data, but the result falls between the pre-registered YES and NO "
                    "criteria. Per protocol: no automatic extra seeds.",
    "COMPARABILITY / ADAPTATION LIMITED": "CANNOT ANSWER - a validity guardrail failed (prefix gap > 2% or code "
                                          "adaptation < 0.05 nats).",
    "INVALID OR INCOMPLETE": "NOT YET - the experiment is incomplete or a run failed numerically.",
}


# =============================================================================
# Report: fixed endpoints; incomplete runs never become negative findings.
# =============================================================================
def stats_from_record(record: dict, doc_ids: list[str] | None) -> EvaluationSums:
    sums = EvaluationSums(total=LossSum(*record["total"]), rare=LossSum(*record["rare"]),
                          classes={k: LossSum(*v) for k, v in record["classes"].items()})
    if doc_ids is not None and "doc_sums" in record:
        sums.documents = {doc_ids[i]: LossSum(s, c) for i, (s, c) in
                          enumerate(zip(record["doc_sums"], record["doc_counts"])) if c}
    return sums


def build_report(state: Path, doc_ids: dict | None = None) -> dict:
    cfg = read_json(state / "config.json")
    thresholds = cfg["decision_thresholds"]
    logs = {(seed, condition): EventLog(state / "runs" / f"S{seed}-{condition}" / "events.jsonl")
            for seed in SEEDS for condition in CONDITIONS}
    missing, failures, seeds, details, progress = [], [], [], [], []

    def get(seed, condition, stage, step, role, domain):
        record = logs[seed, condition].get(stage, step, role, domain)
        if record is None:
            missing.append(f"S{seed}-{condition}/{stage}-{step}-{role}-{domain}")
            return None
        expected_update = step if stage == "prefix" else PREFIX_END + step
        if record["global_update"] != expected_update:
            raise ValueError(f"Evaluation clock mismatch S{seed}-{condition} {stage}@{step}")
        expected = (FULL_WINDOWS if role == "full" else QUICK_WINDOWS) * CONTEXT
        stats = stats_from_record(record, None)
        if stats.total.count != expected or sum(v.count for v in stats.classes.values()) != expected:
            raise ValueError("Wrong evaluation label population")
        if abs(sum(v.ce_sum for v in stats.classes.values()) - stats.total.ce_sum) / expected > 1e-6:
            raise ValueError("Class sums do not reconstruct CE")
        if abs(stats.total.mean - record["ce"]) > 1e-9:
            raise ValueError("CE does not match sufficient statistics")
        if role == "full" and (sum(record["doc_counts"]) != expected
                               or abs(math.fsum(record["doc_sums"]) - stats.total.ce_sum) / expected > 1e-6):
            raise ValueError("Document sums do not reconstruct CE")
        if stage != "prefix":
            switch = state / "runs" / f"S{seed}-{condition}" / "switch.pt.json"
            if not switch.exists() or record.get("parent_sha256") != read_json(switch)["sha256"]:
                raise ValueError(f"Branch evaluation does not descend from its frozen prefix (S{seed}-{condition})")
        return record

    def value(record, metric="ce"):
        return record[metric]

    for seed in SEEDS:
        cache = {}
        for condition in CONDITIONS:
            run_dir = state / "runs" / f"S{seed}-{condition}"
            if (run_dir / "failure.json").exists():
                failures.append(read_json(run_dir / "failure.json") | {"run": f"S{seed}-{condition}"})
            if not (run_dir / "completion.json").exists():
                missing.append(f"S{seed}-{condition}/completion")
            for stage, steps in (("prefix", FULL_PREFIX), ("web", FULL_CONTINUATION), ("python", FULL_CONTINUATION)):
                for step in steps:
                    for domain in ("web", "python"):
                        cache[condition, stage, step, "full", domain] = get(seed, condition, stage, step, "full", domain)
            for stage, steps in (("prefix", (PREFIX_END,)), ("web", QUICK_CONTINUATION), ("python", QUICK_CONTINUATION)):
                for step in steps:
                    for domain in ("web", "python"):
                        cache[condition, stage, step, "quick", domain] = get(seed, condition, stage, step, "quick", domain)
            for stage in ("prefix", "web", "python"):
                points = [(key[2], row) for key, row in cache.items() if key[0] == condition and key[1] == stage
                          and key[3:] == ("full", "web") and row is not None]
                if points:
                    step, row = max(points, key=lambda pair: pair[0])
                    code_row = cache[condition, stage, step, "full", "python"]
                    progress.append({"seed": seed, "condition": condition, "stage": stage, "step": step,
                                     "web_full_ce": value(row),
                                     "python_non_w_full_ce": value(code_row, "non_w_ce") if code_row else None})
        if any(row is None for row in cache.values()):
            continue

        def c(condition, stage, step, role="full", domain="web"):
            return cache[condition, stage, step, role, domain]

        def effect(step, role="full"):
            return contrast(BranchCE(
                value(c("RMS", "prefix", PREFIX_END, role)), value(c("Taper-minus", "prefix", PREFIX_END, role)),
                value(c("RMS", "web", step, role)), value(c("RMS", "python", step, role)),
                value(c("Taper-minus", "web", step, role)), value(c("Taper-minus", "python", step, role))))

        endpoint = effect(CONTINUATION_UPDATES)
        points = {condition: [AdaptationPoint(step, value(c(condition, "python", step, "quick", "python"), "non_w_ce"),
                                              value(c(condition, "python", step, "quick", "web")))
                              for step in QUICK_CONTINUATION] for condition in CONDITIONS}
        matches = {u: matched_forgetting(points["RMS"], points["Taper-minus"], u) for u in PERSISTENCE_POINTS}
        prefix_rms = value(c("RMS", "prefix", PREFIX_END))
        prefix_taper = value(c("Taper-minus", "prefix", PREFIX_END))
        adaptation = {condition: value(c(condition, "prefix", PREFIX_END, domain="python"), "non_w_ce")
                      - value(c(condition, "python", CONTINUATION_UPDATES, domain="python"), "non_w_ce")
                      for condition in CONDITIONS}
        evidence = SeedEvidence(seed, endpoint.d, effect(D_MID_POINT).d, endpoint.q,
                                abs(prefix_taper - prefix_rms) / prefix_rms, adaptation["RMS"],
                                adaptation["Taper-minus"], {u: effect(u, "quick").d for u in QUICK_CONTINUATION},
                                {u: pair[1] for u, pair in matches.items()})
        seeds.append(evidence)
        ids = doc_ids["web"] if doc_ids else None
        if ids is not None:
            sums = [stats_from_record(c(cond, branch, CONTINUATION_UPDATES), ids)
                    for cond, branch in (("Taper-minus", "python"), ("Taper-minus", "web"),
                                         ("RMS", "python"), ("RMS", "web"))]
            d, documents, classes = decompose(*sums)
            if abs(d - endpoint.d) > 1e-6:
                raise ValueError("Document decomposition disagrees with primary contrast")
            tail = asdict(summarize_documents(documents, d))
            class_rows = [asdict(x) for x in classes]
        else:
            tail, class_rows = None, None
        slopes = {condition: prefix_slope([(u, value(c(condition, "prefix", u))) for u in FULL_PREFIX])
                  for condition in CONDITIONS}
        details.append({
            "seed": seed, "endpoint": asdict(endpoint), "evidence": asdict(evidence),
            "full_dev_persistence": {u: asdict(effect(u)) for u in PERSISTENCE_POINTS},
            "matched": {u: {"match": asdict(pair[0]) if pair[0] else None, "differential_forgetting": pair[1]}
                        for u, pair in matches.items()},
            "prefix_slopes_nats_per_million_tokens": slopes,
            "prefix_slope_difference": slopes["Taper-minus"] - slopes["RMS"],
            "prefix_full_web_ce": {"RMS": prefix_rms, "Taper-minus": prefix_taper},
            "tail": tail, "class_contributions": class_rows})
    complete = not missing and len(seeds) == 3
    decision = classify(seeds, primary_complete=complete, correctness_and_data_passed=not failures,
                        nonfinite_primary=any(f.get("type") == "numerical" for f in failures),
                        thresholds=thresholds)
    result = {
        "protocol": cfg["protocol"], "config_sha256": cfg["config_sha256"], "complete": complete,
        "decision": asdict(decision), "h1_answer": H1_ANSWERS[decision.category],
        "completed_seed_groups": len(seeds), "missing_event_count": len(missing),
        "missing_events_first_50": missing[:50], "failures": failures, "seeds": details,
        "latest_observations": progress,
        "uncertainty": asdict(seed_uncertainty([s.d for s in seeds])) if len(seeds) == 3 else None,
        "thresholds": thresholds,
        "interpretation": "Fixed-sample development-data screening (pilot), not confirmation or equivalence. "
                          "Positive D = more excess web forgetting for Taper-minus under the Python shift."}
    lines = [f"H1 ANSWER: {result['h1_answer']}", f"Decision category: {decision.category}",
             f"Complete: {complete}; paired seed groups with full evidence: {len(seeds)}/3; "
             f"missing events: {len(missing)}"]
    for s in seeds:
        lines.append(f"Seed {s.seed}: D={s.d:+.5f}  D3050={s.d3050:+.5f}  Q={s.q:+.5f}  "
                     f"prefix gap={s.absolute_relative_prefix_gap:.2%}  code improvement RMS="
                     f"{s.rms_non_w_improvement:.4f} Taper={s.taper_non_w_improvement:.4f}  matched="
                     f"{ {u: (round(v, 5) if v is not None else None) for u, v in s.matched_differences.items()} }")
    if result["uncertainty"]:
        u = result["uncertainty"]
        lines.append(f"Mean D = {u['mean']:+.5f} nats/token, SD {u['sample_sd']:.5f}, descriptive t95 "
                     f"[{u['descriptive_t95'][0]:+.5f}, {u['descriptive_t95'][1]:+.5f}]")
    for key, val in decision.statistics.items():
        lines.append(f"{key} = {val:+.5f}")
    lines += [f"Reason: {r}" for r in decision.reasons]
    for f in failures:
        lines.append(f"FAILURE {f.get('run')}: {f.get('message')}")
    result["summary_lines"] = lines
    return result


def write_report(state: Path, online: Path | None) -> dict:
    doc_ids = None
    if online is not None:
        manifest = read_json(online / "manifest.json")
        doc_ids = {d: [doc["id"] for doc in read_json(online / manifest["splits"][f"{d}_dev"]["documents"])["documents"]]
                   for d in ("web", "python")}
    report = build_report(state, doc_ids)
    write_json(state / "decision-report.json", report)
    (state / "decision-report.txt").write_text("\n".join(report["summary_lines"]) + "\n", encoding="utf-8")
    return report


# =============================================================================
# Orchestrator
# =============================================================================
STATE_MARKER = "H1-STATE.json"
BOOTSTRAP_MARKER = "H1-BOOTSTRAP.json"


def safe_report(state: Path, online: Path | None):
    """Analysis must never block training or lose state."""
    try:
        return write_report(state, online)
    except Exception:
        log("REPORT ERROR (training/state unaffected):\n" + traceback.format_exc())
        return None


def gpu_count() -> int:
    try:
        output = subprocess.check_output(["nvidia-smi", "-L"], text=True, timeout=60)
        return sum(1 for line in output.splitlines() if line.startswith("GPU "))
    except Exception:
        return 0


def find_previous_state(inputs: Path, work_state: Path) -> Path | None:
    candidates = [p.parent for p in inputs.rglob(STATE_MARKER) if p.parent.resolve() != work_state.resolve()]
    if not candidates:
        return None
    ids = {read_json(p / STATE_MARKER)["experiment_id"] for p in candidates}
    if len(ids) != 1:
        raise RuntimeError(f"Multiple different H1 experiments attached as input: {ids}. Attach only one.")
    return max(candidates, key=lambda p: read_json(p / STATE_MARKER).get("sessions_completed", 0))


def progress_summary(state: Path) -> list[str]:
    lines = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            run_dir = state / "runs" / f"S{seed}-{condition}"
            if (run_dir / "completion.json").exists():
                lines.append(f"  S{seed}-{condition}: complete")
            elif (run_dir / "failure.json").exists():
                lines.append(f"  S{seed}-{condition}: FAILED ({read_json(run_dir / 'failure.json')['message']})")
            elif (run_dir / "latest.pt.json").exists():
                p = read_json(run_dir / "latest.pt.json")["progress"]
                lines.append(f"  S{seed}-{condition}: {p['stage']} step {p['step']}/{stage_limit(p['stage'])}")
            else:
                lines.append(f"  S{seed}-{condition}: not started")
    return lines


def remaining_updates(state: Path, condition: str) -> int:
    total = 0
    per_run = PREFIX_END + 2 * CONTINUATION_UPDATES
    for seed in SEEDS:
        run_dir = state / "runs" / f"S{seed}-{condition}"
        if (run_dir / "completion.json").exists():
            continue
        if (run_dir / "latest.pt.json").exists():
            p = read_json(run_dir / "latest.pt.json")["progress"]
            done = {"prefix": 0, "web": PREFIX_END, "python": PREFIX_END + CONTINUATION_UPDATES}[p["stage"]] + p["step"]
            total += per_run - done
        else:
            total += per_run
    return total


def main(args) -> int:
    t0 = float(os.environ.get("H1_T0") or time.time())
    session_hours = float(os.environ.get("H1_SESSION_HOURS", "11.25"))
    allow_fresh = os.environ.get("H1_ALLOW_FRESH_START", "0") == "1"
    inputs = Path(args.inputs)
    work = Path(args.work)
    state = work / "h1state"
    work_deadline = t0 + session_hours * 3600
    hard_deadline = work_deadline + (15 * 60 if not MINI else 60)
    log(f"H1 runner {RUNNER_VERSION}; session budget {session_hours:.2f} h "
        f"({(work_deadline - time.time()) / 3600:.2f} h of work time left)")

    # 1) Carry the previous state forward FIRST, so this run's output always contains it.
    fresh = False
    if (state / STATE_MARKER).exists():
        log("Continuing state already in the working directory")
    else:
        previous = find_previous_state(inputs, state)
        if previous is not None:
            tick = time.time()
            shutil.copytree(previous, state)
            log(f"Restored previous session state from {previous} in {time.time() - tick:.0f}s")
        else:
            bootstrapped = any(True for _ in inputs.rglob(BOOTSTRAP_MARKER))
            if gpu_count() == 0 and not bootstrapped and not allow_fresh:
                write_json(work / BOOTSTRAP_MARKER, {"created_unix": time.time(), "runner": RUNNER_VERSION})
                log("BOOTSTRAP: no GPU and no state -> wrote the start marker. Run the next version on GPU T4 x2.")
                return 0
            if not (allow_fresh or bootstrapped):
                log("ERROR: no previous H1 state in the inputs. The previous version's output is missing "
                    "(was it killed?). Attach this notebook's latest version that HAS an 'h1state' folder. "
                    "Refusing to silently start a new experiment.")
                return 4
            fresh = True

    # 2) Data identity.
    online, orders = locate_data(inputs) if args.online is None else (Path(args.online), Path(args.orders))
    log(f"Prepared corpus: {online}")
    tick = time.time()
    data_identity = verify_data(online, orders)
    log(f"Data hashes verified in {time.time() - tick:.0f}s; C4 rev {data_identity['sources']['web']['revision'][:12]}, "
        f"Stack rev {data_identity['sources']['python']['revision'][:12]}")

    if fresh:
        state.mkdir(parents=True, exist_ok=True)
        cfg = config_document(args.microbatch, SEQUENCES_PER_UPDATE // args.microbatch, args.eval_microbatch)
        cfg["data"] = data_identity
        cfg["config_sha256"] = digest_json(cfg)
        cfg["checkpoint_seconds"] = 900
        write_json(state / "config.json", cfg)
        write_json(state / STATE_MARKER, {"schema": "h1-state-1", "experiment_id": cfg["config_sha256"][:16]
                                          + "-" + hex(int(time.time()))[2:], "created_unix": time.time(),
                                          "sessions_completed": 0})
        log(f"NEW experiment created; config sha256 {cfg['config_sha256'][:16]}")
    cfg = read_json(state / "config.json")
    marker = read_json(state / STATE_MARKER)
    expected = config_document(cfg["microbatch"], cfg["accumulation"], cfg["eval_microbatch"])
    expected["data"] = data_identity
    if digest_json(expected) != cfg["config_sha256"]:
        log("ERROR: the frozen configuration or data identity differs from this code/data. Refusing to mix strata.")
        return 5

    session = {"session": marker.get("sessions_completed", 0) + 1, "started_unix": t0,
               "runner": RUNNER_VERSION, "code_sha256": sha256_file(Path(__file__)),
               "python": sys.version.split()[0], "torch": torch.__version__,
               "cuda": torch.version.cuda, "gpus": gpu_count()}
    try:
        session["gpu_names"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True).strip().splitlines()
    except Exception:
        session["gpu_names"] = []
    log(f"Session {session['session']}: torch {session['torch']} CUDA {session['cuda']}, GPUs {session['gpu_names']}")
    report = safe_report(state, online)
    if report is not None and (report["complete"] or report["failures"]):
        log("Nothing to train: experiment already finished or has a recorded failure.")
        for line in report["summary_lines"]:
            print(line, flush=True)
        return 0

    n_gpus = session["gpus"]
    if os.environ.get("H1_TEST_TWO_WORKERS_ONE_GPU") == "1" and n_gpus == 1:
        assignments = [("0", "RMS"), ("0", "Taper-minus")]  # local plumbing test only
    elif n_gpus >= 2:
        assignments = [("0", "RMS"), ("1", "Taper-minus")]
    elif n_gpus == 1:
        log("WARNING: only one GPU visible; running both conditions sequentially on it (slower).")
        assignments = [("0", "RMS,Taper-minus")]
    else:
        log("ERROR: no GPU visible. Select accelerator 'GPU T4 x2'.")
        return 6

    worker_deadline = work_deadline - WORKER_RESERVE_SECONDS
    restarts = {cond: 0 for _, cond in assignments}
    procs: dict[str, subprocess.Popen] = {}
    logs_dir = state / "logs"
    logs_dir.mkdir(exist_ok=True)

    def launch(device: str, conditions: str) -> subprocess.Popen:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, PYTHONUNBUFFERED="1")
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "worker", "--state", str(state),
                   "--online", str(online), "--orders", str(orders), "--conditions", conditions,
                   "--deadline", str(worker_deadline)] + (["--mini"] if MINI else [])
        proc = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                bufsize=1)
        log_path = logs_dir / f"session{session['session']}-{conditions.replace(',', '+')}.log"

        def pump():
            with log_path.open("a", encoding="utf-8") as sink:
                for line in proc.stdout:
                    sink.write(line)
                    sink.flush()
                    print(f"[{conditions}] {line}", end="", flush=True)
        threading.Thread(target=pump, daemon=True).start()
        return proc

    for device, conditions in assignments:
        procs[conditions] = launch(device, conditions)
    exit_codes: dict[str, int] = {}
    numerical_failure = False
    while procs:
        time.sleep(5)
        for (device, conditions) in assignments:
            proc = procs.get(conditions)
            if proc is None:
                continue
            code = proc.poll()
            if code is None:
                if time.time() > hard_deadline:
                    log(f"Worker {conditions} did not stop by the hard deadline; terminating")
                    proc.terminate()
                continue
            del procs[conditions]
            exit_codes[conditions] = code
            if code == 3:
                numerical_failure = True
                log("Numerical failure recorded; stopping the other worker to preserve quota.")
                for other in procs.values():
                    other.send_signal(signal.SIGTERM)
            elif code not in (0,) and not numerical_failure and \
                    time.time() < worker_deadline - (20 * 60 if not MINI else 5) and restarts[conditions] < 3:
                restarts[conditions] += 1
                log(f"Worker {conditions} crashed (exit {code}); restarting from its latest checkpoint "
                    f"(attempt {restarts[conditions]})")
                write_json(logs_dir / f"crash-session{session['session']}-{conditions}-{restarts[conditions]}.json",
                           {"exit_code": code, "time": time.time()})
                procs[conditions] = launch(device, conditions)
    session["ended_unix"] = time.time()
    session["hours"] = (session["ended_unix"] - t0) / 3600
    session["exit_codes"] = exit_codes
    session["restarts"] = restarts
    with (state / "sessions.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(session) + "\n")
    marker["sessions_completed"] = session["session"]
    write_json(state / STATE_MARKER, marker)
    report = safe_report(state, online) or {"summary_lines": ["(report failed; see traceback above)"],
                                             "complete": False, "failures": []}
    print("\n" + "=" * 100, flush=True)
    print("PROGRESS", flush=True)
    for line in progress_summary(state):
        print(line, flush=True)
    for condition in CONDITIONS:
        log(f"{condition}: {remaining_updates(state, condition):,} optimizer updates remaining")
    print("=" * 100, flush=True)
    for line in report["summary_lines"]:
        print(line, flush=True)
    print("=" * 100, flush=True)
    if not report["complete"] and not report["failures"]:
        print("NOT FINISHED: run this notebook again (Save Version -> Save & Run All). It resumes automatically "
              "from this output.", flush=True)
    return 0


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("main")
    m.add_argument("--inputs", default="/kaggle/input")
    m.add_argument("--work", default="/kaggle/working")
    m.add_argument("--online", default=None)
    m.add_argument("--orders", default=None)
    m.add_argument("--microbatch", type=int, default=8)
    m.add_argument("--eval-microbatch", type=int, default=16)
    m.add_argument("--mini", action="store_true")
    w = sub.add_parser("worker")
    w.add_argument("--state", required=True)
    w.add_argument("--online", required=True)
    w.add_argument("--orders", required=True)
    w.add_argument("--conditions", required=True)
    w.add_argument("--deadline", type=float, required=True)
    w.add_argument("--mini", action="store_true")
    r = sub.add_parser("report")
    r.add_argument("--state", required=True)
    r.add_argument("--online", default=None)
    r.add_argument("--mini", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse()
    if getattr(arguments, "mini", False):
        use_mini_protocol()
    if arguments.command == "main":
        sys.exit(main(arguments))
    elif arguments.command == "worker":
        sys.exit(worker_main(arguments))
    else:
        result = write_report(Path(arguments.state), Path(arguments.online) if arguments.online else None)
        print("\n".join(result["summary_lines"]))
