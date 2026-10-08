#!/usr/bin/env python3
"""Study 2 single-notebook runner for Kaggle T4 x2 (pre-registered; see study2/PROTOCOL.md).

H2 (primary): after a switch from web text to the selected domain X, does internal TaperNorm
(Taper-minus) show more persistent held-out web forgetting than RMSNorm, relative to continuing web
training? X is fixed before any training by the pre-registered selection probe (probe_domains.py).
R2 (secondary): the pilot's H1 (web -> Python) replicated on the fresh seeds 104-106.

One file does everything on the GPU side: verifies the pilot assets Study 2 reuses (the switch
states and evaluation records of seeds 101-103) and every prepared array, trains RMS on GPU 0 and
Taper-minus on GPU 1, evaluates on the frozen schedule, checkpoints, and applies the frozen decision
rules. Jobs per condition, in protocol order:
  seeds 101-103: X branch only, restored from the pilot's switch state (their web branch is
                 already in the pilot's records);
  seeds 104-106: web prefix -> web branch -> Python branch -> X branch.
All state lives in <work>/s2state. On Kaggle that is the notebook's own output, which the next run
of the same notebook receives as input and resumes from automatically.

The model, TaperNorm, optimizer, update step, evaluation, diagnostics and analysis rules are copied
from kaggle_h1/h1_run.py (the pilot runner, which is never modified); test_s2_run.py checks every
copy against it. Only the job list, the third domain and the Study 2 analyses are new.

Commands:
  python s2_run.py main            # orchestrator (Kaggle notebook cell)
  python s2_run.py worker ...      # internal: one GPU, one condition
  python s2_run.py lineage ...     # smoke/pre-flight: re-evaluate the pilot switch states
  python s2_run.py report --state DIR --pilot-state DIR [--pilot-online DIR]
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

RUNNER_VERSION = "s2-single-notebook-1"

# =============================================================================
# Protocol constants: identical to the pilot (protocol v3). ``use_mini_protocol`` shrinks them ONLY
# for plumbing tests (smoke notebook, local tests); it is never used for the experiment.
# =============================================================================
SEEDS = (101, 102, 103, 104, 105, 106)
PILOT_SEEDS = (101, 102, 103)   # X branch only, from the pilot's switch states
FRESH_SEEDS = (104, 105, 106)   # full runs: prefix -> web, Python and X branches
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

# Study 2 additions (pre-registered; the scientific constants above are the pilot's, unchanged).
FRESH_STAGES = ("prefix", "web", "python", "x")
STAGE_DOMAINS = {"prefix": ("web", "python"), "web": ("web", "python"), "python": ("web", "python"),
                 "x": ("web", "x")}
DIAG_DOMAINS = ("web", "python", "x")
LINEAGE_CE_TOLERANCE = 1e-3       # nats/token: a restored switch state must re-evaluate to its record
LINEAGE_ENERGY_TOLERANCE = 1e-3   # relative, per-site diagnostic energies
ANALYSIS = {"t95": {"3": 4.303, "6": 2.571}, "t90": {"3": 2.920, "6": 2.015}, "equivalence_bound": 0.015,
            "bootstrap_replicates": 10_000, "bootstrap_seed": 20261011}
S2_ARRAYS_SCHEMA = "s2-arrays-v1"
S2_ORDERS_SCHEMA = "s2-orders-v1"
PILOT_STATE_MARKER = "H1-STATE.json"

# The pilot assets Study 2 reuses, pinned (reports/h1-kaggle/ in the repository; Kaggle notebook
# danny00/h1-single-notebook-run). ``events_digest`` is line-ending independent (events_digest()).
PILOT_PINS = {
    "config_sha256": "c18441799eee2b6a78b257e2e22c6ac32ddb7c30cd81d6eb57e488b2b173ce10",
    "experiment_id": "c18441799eee2b6a-6ac0f2cb",
    "runner_sha256": "66769e7de30ec36f13a5acfa275d2691cb30251a08b9354a5c5063bd813cb83b",
    "online_manifest_sha256": "290886d93760f5a66f1e55b3c4730a02dfddf1b8153ba12ed78fba6374ab87d7",
    "orders_manifest_sha256": "bd7abf44121a7402acc1d9603d3d29f20a1074a97cd202e0e2f964406bcc9bac",
    "switch_sha256": {
        "101/RMS": "6e890478d28826c4ba020e674717b87ffbbd3dd363d478876b96ef944f0a292d",
        "101/Taper-minus": "223da6d23881954c96928e36d9d5e74270ae16ab296ce0dbffb7a45078f727c2",
        "102/RMS": "0f0f603bf4546d1955f0e75fe3c4dc6af1514dc8bb01c0166f1830b739a82b12",
        "102/Taper-minus": "78fde02da7e9d922edf6918e86487193931a83331196f4f1bbf58c620db251b2",
        "103/RMS": "e7b5c32566bc6ba137f52492f27475891bceb9c0fbfe85bd7ae43b6c706ea94c",
        "103/Taper-minus": "1e457bd6fb889c7a717deac80f3eb9e85a5e007bed7728f587a501c8c80e9f5d"},
    "events_digest": {
        "101/RMS": "2d8aced7560ca17f28fb6b70c19d7fa519920dbe99234dab570fcdb98cd4e02f",
        "101/Taper-minus": "9a013705584eef64ce6ef16c7fe3b0e55037b794491ab6c56234466e12fa216b",
        "102/RMS": "53060b9a3f5e95ebc47a3a9640142e0efaf5b7befef3c8f5b707cf988d455e49",
        "102/Taper-minus": "8f782648ef6759b7c26c586cb26a6b0922663adedfaf8e403f0d497e39e43c28",
        "103/RMS": "65f3caf2ce6016d8f475b19946e317f575e014453ac423432ae5444c840b894c",
        "103/Taper-minus": "d0df7be0c88a3a6ca3a0a8e52e603fe2d73c374f9412c302d375c01386663ea0"},
}


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


def job_stages(seed: int) -> tuple[str, ...]:
    """Pilot seeds continue from the pilot's switch state into the X branch only."""
    return ("x",) if seed in PILOT_SEEDS else FRESH_STAGES


def events_at(stage: str, step: int) -> tuple[list[tuple[str, str]], bool]:
    """(role, domain) evaluations and whether diagnostics run at this stage step."""
    if stage == "prefix":
        roles = (["full"] if step in FULL_PREFIX else []) + (["quick"] if step == PREFIX_END else [])
        diag = step == PREFIX_END
    else:
        roles = (["full"] if step in FULL_CONTINUATION else []) + (["quick"] if step in QUICK_CONTINUATION else [])
        diag = step in DIAGNOSTIC_CONTINUATION
    return [(role, domain) for role in roles for domain in STAGE_DOMAINS[stage]], diag


def stage_data(stage: str, step: int) -> tuple[str, int]:
    """Training domain and order offset of the next update (the global clock continues per branch)."""
    if stage == "prefix":
        return "web", step * SEQUENCES_PER_UPDATE
    if stage == "web":
        return "web", (PREFIX_END + step) * SEQUENCES_PER_UPDATE
    return stage, step * SEQUENCES_PER_UPDATE


def config_document(microbatch: int, accumulation: int, eval_microbatch: int, pilot_pins: dict,
                    domain: dict) -> dict:
    return {
        "protocol": "study2-v1-kaggle-fp16-paired-single-notebook",
        "runner": RUNNER_VERSION,
        "mini_protocol": MINI,
        "hypotheses": {
            "H2": "Primary: excess persistent web forgetting of Taper-minus vs RMS after a web -> X switch "
                  "(X = the pre-registered selection), seeds 101-106.",
            "R2": "Secondary: the pilot's H1 (web -> Python) on independent seeds 104-106, pilot rules."},
        "amendments": [
            "Carried from the pilot (unchanged): Tesla T4 FP16 autocast + GradScaler, FP32 master weights/"
            "moments, FP32 norm/EMA reductions and FP32 loss; RMS and Taper-minus as independent workers on "
            "two T4s; microbatch 8 x accumulation 4; eval microbatch 16; self-resuming single notebook.",
            "Weight snapshots: only switch states and branch-final weights are kept (storage); every full-dev "
            "point keeps per-document/per-class sufficient statistics.",
            "Seeds 101-103 reuse the pilot's switch states and the pilot's prefix/web/Python evaluation "
            "records (pinned by hash); only their X branch is new.",
        ],
        "seeds": list(SEEDS), "pilot_seeds": list(PILOT_SEEDS), "fresh_seeds": list(FRESH_SEEDS),
        "conditions": list(CONDITIONS),
        "jobs": {str(seed): list(job_stages(seed)) for seed in SEEDS},
        "domain": domain,
        "pilot": pilot_pins,
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
                     "quick_windows": QUICK_WINDOWS, "diag_windows": DIAG_WINDOWS,
                     "stage_domains": {k: list(v) for k, v in STAGE_DOMAINS.items()},
                     "diag_domains": list(DIAG_DOMAINS)},
        "lineage_tolerance": {"ce_nats": LINEAGE_CE_TOLERANCE, "energy_relative": LINEAGE_ENERGY_TOLERANCE},
        "decision_thresholds": DECISION_THRESHOLDS,
        "analysis": ANALYSIS,
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


def events_digest(path: Path) -> str:
    """SHA-256 of the canonical JSON of every record: independent of line endings and whitespace."""
    digest = hashlib.sha256()
    for raw in path.read_bytes().splitlines():
        if raw.strip():
            digest.update(json.dumps(json.loads(raw), sort_keys=True, separators=(",", ":")).encode() + b"\n")
    return digest.hexdigest()


# =============================================================================
# Data: the pilot's hash-verified arrays (web, Python) + Study 2's arrays for the selected domain X.
# =============================================================================
CLASS_CODES = {"W": 0, "A": 1, "P": 2, "X": 3}


class DevSet:
    def __init__(self, root: Path, row: dict, domain: str):
        self.tokens = np.fromfile(root / row["file"], dtype="<u2")
        owners = np.fromfile(root / row["owners"], dtype="<u4")
        docs = read_json(root / row["documents"])["documents"]
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


class WindowArray(TrainStream):
    """Packed windows held in memory (the selection probe's candidate samples)."""

    def __init__(self, tokens: np.ndarray):  # noqa: super().__init__ reads a file
        self.tokens = np.asarray(tokens, dtype="<u2")
        self.windows = (len(self.tokens) - 1) // CONTEXT


def locate_inputs(inputs: Path) -> dict[str, Path]:
    """Find the pilot corpus, Study 2's arrays/orders and the pilot's run state by content, not path."""
    found: dict[str, list[Path]] = {"pilot_online": [], "s2_online": [], "s2_orders": [], "pilot_state": []}
    for path in inputs.rglob("manifest.json"):
        try:
            schema = read_json(path).get("schema")
        except Exception:
            continue
        if schema == "pilot-arrays-v1" and path.parent.name == "online":
            found["pilot_online"].append(path.parent)
        elif schema == S2_ARRAYS_SCHEMA:
            found["s2_online"].append(path.parent)
        elif schema == S2_ORDERS_SCHEMA:
            found["s2_orders"].append(path.parent)
    found["pilot_state"] = [path.parent for path in inputs.rglob(PILOT_STATE_MARKER)]
    result = {}
    for key, paths in found.items():
        unique = sorted({p.resolve() for p in paths})
        if len(unique) != 1:
            raise FileNotFoundError(f"Expected exactly one {key} under {inputs}, found {unique}")
        result[key] = unique[0]
    result["pilot_orders"] = result["pilot_online"].parent / "orders"
    if not (result["pilot_orders"] / "manifest.json").exists():
        raise FileNotFoundError("The pilot's 'orders' directory is missing next to its 'online' directory")
    return result


def verify_data(pilot_online: Path, pilot_orders: Path, s2_online: Path, s2_orders: Path) -> dict:
    """Hash every file training/evaluation reads. Reserved test sets are never referenced."""
    manifest = read_json(pilot_online / "manifest.json")
    if manifest.get("schema") != "pilot-arrays-v1":
        raise ValueError("Not the pilot's original-protocol arrays")
    if set(manifest["splits"]) != {"web_train", "python_train", "web_dev", "python_dev"}:
        raise ValueError("Pilot corpus must contain exactly the four train/dev splits")
    if manifest.get("reserved_test_sealed", {}).get("online_path_included") is not False and not MINI:
        raise ValueError("Pilot reserved-test isolation not established")
    pilot_identity = sha256_file(pilot_online / "manifest.json")
    if not MINI and pilot_identity != PILOT_PINS["online_manifest_sha256"]:
        raise ValueError("The pilot corpus is not the one the pilot ran on")
    checked = {}

    def check(root: Path, key: str, row: dict) -> None:
        fields = [("file", "sha256")]
        if key.endswith("dev"):
            fields += [("owners", "owners_sha256"), ("documents", "documents_sha256")]
        for name_field, digest_field in fields:
            path = root / row[name_field]
            actual = sha256_file(path)
            if actual != row[digest_field]:
                raise ValueError(f"Hash mismatch: {path.name}")
            checked[path.name] = actual

    for key, row in manifest["splits"].items():
        check(pilot_online, key, row)
    if sha256_file(pilot_online / "classes.json") != manifest["classes_sha256"]:
        raise ValueError("Token class table hash mismatch")
    pilot_order_manifest = read_json(pilot_orders / "manifest.json")
    if pilot_order_manifest["array_manifest_sha256"] != pilot_identity:
        raise ValueError("Pilot orders were generated for different arrays")
    pilot_orders_identity = sha256_file(pilot_orders / "manifest.json")
    if not MINI and pilot_orders_identity != PILOT_PINS["orders_manifest_sha256"]:
        raise ValueError("The pilot orders are not the ones the pilot ran on")
    for seed in PILOT_SEEDS:
        for key, row in pilot_order_manifest["seeds"][str(seed)].items():
            if sha256_file(pilot_orders / row["file"]) != row["sha256"]:
                raise ValueError(f"Order hash mismatch: {row['file']}")

    s2 = read_json(s2_online / "manifest.json")
    if s2.get("schema") != S2_ARRAYS_SCHEMA or set(s2["splits"]) != {"x_train", "x_dev"}:
        raise ValueError("Study 2 corpus must contain exactly the X train/dev splits")
    if s2.get("reserved_test_sealed", {}).get("online_path_included") is not False and not MINI:
        raise ValueError("Study 2 reserved-test isolation not established")
    if s2["pilot_array_manifest_sha256"] != pilot_identity:
        raise ValueError("Study 2 arrays were deduplicated against a different pilot corpus")
    if s2["tokenizer_hashes"]["id_to_bytes_sha256"] != manifest["tokenizer_hashes"]["id_to_bytes_sha256"] \
            or s2["classes_sha256"] != manifest["classes_sha256"]:
        raise ValueError("Study 2 arrays use a different tokenizer/class table")
    for key, row in s2["splits"].items():
        check(s2_online, key, row)
    s2_identity = sha256_file(s2_online / "manifest.json")
    selection_sha = sha256_file(s2_online / "selection.json")
    selection = read_json(s2_online / "selection.json")
    rank = s2["selection"].get("rank", 0)  # 0 = the selection; 1 = the protocol's pre-registered fallback only
    if selection_sha != s2["selection"]["selection_sha256"] or selection.get("status") != "selected" \
            or selection["ranking"][0] != selection.get("selected") or rank not in (0, 1) \
            or selection["ranking"][rank] != s2["domain"]["id"]:
        raise ValueError("The prepared domain is not the probe's recorded selection")
    order_manifest = read_json(s2_orders / "manifest.json")
    if order_manifest.get("schema") != S2_ORDERS_SCHEMA \
            or order_manifest["pilot_array_manifest_sha256"] != pilot_identity \
            or order_manifest["x_array_manifest_sha256"] != s2_identity:
        raise ValueError("Study 2 orders were generated for different arrays")
    for seed in SEEDS:
        rows = order_manifest["seeds"][str(seed)]
        expected = {"x"} if seed in PILOT_SEEDS else {"web", "python", "rare", "x"}
        if set(rows) != expected:
            raise ValueError(f"Unexpected order files for seed {seed}: {sorted(rows)}")
        for key, row in rows.items():
            if sha256_file(s2_orders / row["file"]) != row["sha256"]:
                raise ValueError(f"Order hash mismatch: {row['file']}")
    if not MINI:
        for split_manifest, key, minimum in ((manifest, "web_train", 260_000_000),
                                             (manifest, "python_train", 110_000_000),
                                             (s2, "x_train", 110_000_000)):
            if split_manifest["splits"][key]["tokens"] < minimum:
                raise ValueError(f"Training quota not met: {key}")
    return {"pilot_online_manifest_sha256": pilot_identity,
            "pilot_orders_manifest_sha256": pilot_orders_identity,
            "s2_online_manifest_sha256": s2_identity,
            "s2_orders_manifest_sha256": sha256_file(s2_orders / "manifest.json"),
            "selection_sha256": selection_sha,
            "sources": {k: {"repo": v.get("repo"), "revision": v.get("revision")}
                        for k, v in manifest.get("sources", {}).items()}
            | {"x": {"repo": s2["domain"]["repo"], "revision": s2["domain"]["revision"]}},
            "files_sha256": checked}


def domain_identity(s2_online: Path) -> dict:
    s2 = read_json(s2_online / "manifest.json")
    return {key: s2["domain"][key] for key in ("id", "label", "repo", "revision")} \
        | {"selection_sha256": s2["selection"]["selection_sha256"], "selection_rank": s2["selection"].get("rank", 0)}


class Data:
    def __init__(self, pilot_online: Path, pilot_orders: Path, s2_online: Path, s2_orders: Path):
        self.pilot_orders, self.s2_orders = pilot_orders, s2_orders
        self.manifest = read_json(pilot_online / "manifest.json")
        self.s2_manifest = read_json(s2_online / "manifest.json")
        classes = read_json(pilot_online / "classes.json")["classes"]
        if len(classes) != VOCAB or not set(classes) <= set("WAPX"):
            raise ValueError("Invalid class table")
        self.class_letters = np.array(classes)
        self.class_codes = np.array([CLASS_CODES[c] for c in classes], dtype=np.int64)
        rows = {"web": (pilot_online, self.manifest["splits"]),
                "python": (pilot_online, self.manifest["splits"]),
                "x": (s2_online, self.s2_manifest["splits"])}
        self.train = {domain: TrainStream(root / splits[f"{domain}_train"]["file"])
                      for domain, (root, splits) in rows.items()}
        self.dev = {domain: DevSet(root, splits[f"{domain}_dev"], domain) for domain, (root, splits) in rows.items()}
        self.dev_sha256 = {domain: splits[f"{domain}_dev"]["sha256"] for domain, (_, splits) in rows.items()}
        self.pilot_order_manifest = read_json(pilot_orders / "manifest.json")
        self.s2_order_manifest = read_json(s2_orders / "manifest.json")

    def probes(self) -> dict:
        return {domain: self.train[domain] for domain in DIAG_DOMAINS}

    def seed_orders(self, seed: int) -> dict:
        sources = [(self.s2_orders, self.s2_order_manifest["seeds"][str(seed)])]
        if seed in PILOT_SEEDS:
            sources.append((self.pilot_orders, self.pilot_order_manifest["seeds"][str(seed)]))
        result = {key: np.load(root / row["file"], allow_pickle=False)
                  for root, rows in sources for key, row in rows.items()}
        if len(result["x"]) < CONTINUATION_UPDATES * SEQUENCES_PER_UPDATE:
            raise ValueError("X order too short")
        if seed in FRESH_SEEDS:
            if len(result["web"]) < (PREFIX_END + CONTINUATION_UPDATES) * SEQUENCES_PER_UPDATE:
                raise ValueError("Web order too short")
            if len(result["python"]) < CONTINUATION_UPDATES * SEQUENCES_PER_UPDATE:
                raise ValueError("Python order too short")
        result["rare"] = np.asarray(result["rare"], dtype=bool)
        return result


# =============================================================================
# Model (verbatim from kaggle_h1/h1_run.py)
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
# Evaluation: exact token-weighted sufficient statistics (vectorized; pilot code, any domain).
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
              "array_sha256": data.dev_sha256[domain],
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
def diagnostics(model, probes: Mapping[str, TrainStream], class_letters: np.ndarray, rare: np.ndarray,
                microbatch: int, device: torch.device) -> dict:
    """Forward-only probes (the pilot's diagnostics, over any named probe streams; "web" is the reference).

    Each probe is the last DIAG_WINDOWS windows of its stream. Descriptive, not a decision input."""
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
        labels_per_domain = {}
        for domain, stream in probes.items():
            current["domain"] = domain
            indices = np.arange(max(0, stream.windows - DIAG_WINDOWS), stream.windows)
            labels_per_domain[domain] = len(indices) * CONTEXT
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
                energies = {domain: row["mean_squared_norm"] for domain, row in view.items()}
                web = energies.get("web")
                view["log_k_vs_web"] = {
                    domain: 0.5 * math.log(max(web, 1e-12) / max(value, 1e-12))
                    if web is not None and math.isfinite(web) and math.isfinite(value) else None
                    for domain, value in energies.items() if domain != "web"}
                view["near_zero_energy"] = min(energies.values()) <= 1e-12
        gains = {name: {"min": float(p.min()), "max": float(p.max()), "rms": float(p.float().square().mean().sqrt())}
                 for name, p in model.named_parameters() if p.ndim == 1}
        c = {name: float(b) for name, b in model.named_buffers() if name.endswith(".c")}
        embeddings = model.token_embedding.weight.float().norm(dim=-1).cpu().numpy()
        groups = [(k, class_letters == k) for k in "WAPX"] + [("R", np.asarray(rare))]
        embedding_groups = {key: {"count": int(mask.sum()),
                                  "mean_norm": float(embeddings[mask].mean()) if mask.any() else None}
                            for key, mask in groups}
        return {"probe": "last training windows of each stream; may overlap trained documents",
                "forward_labels_per_domain": labels_per_domain, "sites": summaries,
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
    payload = {"schema": "s2-ckpt-1", "identity": identity, "progress": progress,
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


def load_checkpoint(path: Path, model, optimizer, scaler, identity: dict, *, schema: str = "s2-ckpt-1") -> dict:
    """Restore a complete state. ``schema="h1-ckpt-1"`` reads a pilot switch state (pilot identity)."""
    sidecar = path.with_name(path.name + ".json")
    if sidecar.exists() and sha256_file(path) != read_json(sidecar)["sha256"]:
        raise ValueError(f"Checkpoint checksum mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["schema"] != schema or payload["identity"] != identity:
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
# The pilot assets Study 2 reuses: switch states and evaluation records of seeds 101-103.
# =============================================================================
class PilotReference:
    """Read-only access to the pilot's run state (h1state), verified against its pins.

    In the mini protocol the "pilot" is a mini run of kaggle_h1/h1_run.py made by the smoke test; its
    pins are derived from that directory instead of PILOT_PINS."""

    def __init__(self, root: Path, pins: dict | None = None):
        self.root = Path(root)
        self.pins = pins if pins is not None else (self.derive_pins(self.root) if MINI else PILOT_PINS)
        self._events: dict[tuple[int, str], EventLog] = {}

    @staticmethod
    def derive_pins(root: Path) -> dict:
        runs = {f"{s}/{c}": root / "runs" / f"S{s}-{c}" for s in PILOT_SEEDS for c in CONDITIONS}
        return {"config_sha256": read_json(root / "config.json")["config_sha256"],
                "experiment_id": read_json(root / PILOT_STATE_MARKER)["experiment_id"],
                "switch_sha256": {k: read_json(p / "switch.pt.json")["sha256"] for k, p in runs.items()},
                "events_digest": {k: events_digest(p / "events.jsonl") for k, p in runs.items()}}

    def run_dir(self, seed: int, condition: str) -> Path:
        return self.root / "runs" / f"S{seed}-{condition}"

    def switch_sha256(self, seed: int, condition: str) -> str:
        return self.pins["switch_sha256"][f"{seed}/{condition}"]

    def identity(self, seed: int, condition: str) -> dict:
        return {"config_sha256": self.pins["config_sha256"], "seed": seed, "condition": condition}

    def verify(self, *, hash_switch_files: bool = True) -> dict:
        if read_json(self.root / "config.json").get("config_sha256") != self.pins["config_sha256"]:
            raise ValueError("Pilot state has a different configuration")
        if read_json(self.root / PILOT_STATE_MARKER).get("experiment_id") != self.pins["experiment_id"]:
            raise ValueError("Pilot state belongs to a different experiment")
        for seed in PILOT_SEEDS:
            for condition in CONDITIONS:
                run, key = self.run_dir(seed, condition), f"{seed}/{condition}"
                expected = self.pins["switch_sha256"][key]
                if read_json(run / "completion.json")["switch_sha256"] != expected \
                        or read_json(run / "switch.pt.json")["sha256"] != expected:
                    raise ValueError(f"Pilot switch receipt mismatch: S{seed}-{condition}")
                if hash_switch_files and sha256_file(run / "switch.pt") != expected:
                    raise ValueError(f"Pilot switch state hash mismatch: S{seed}-{condition}")
                if events_digest(run / "events.jsonl") != self.pins["events_digest"][key]:
                    raise ValueError(f"Pilot evaluation records differ: S{seed}-{condition}")
        return self.pins

    def events(self, seed: int, condition: str) -> EventLog:
        if (seed, condition) not in self._events:
            self._events[seed, condition] = EventLog(self.run_dir(seed, condition) / "events.jsonl")
        return self._events[seed, condition]

    def load_switch(self, seed: int, condition: str, model, optimizer, scaler) -> str:
        """Restore the pilot's complete switch state (weights, optimizer moments, scaler, RNG)."""
        path, expected = self.run_dir(seed, condition) / "switch.pt", self.switch_sha256(seed, condition)
        if sha256_file(path) != expected:
            raise ValueError(f"Pilot switch state hash mismatch: S{seed}-{condition}")
        load_checkpoint(path, model, optimizer, scaler, self.identity(seed, condition), schema="h1-ckpt-1")
        if int(model.completed_updates) != PREFIX_END:
            raise ValueError("Pilot switch state is not at the end of the web prefix")
        return expected

    def decision_report(self) -> dict | None:
        path = self.root / "decision-report.json"
        return read_json(path) if path.exists() else None


class LineageError(Exception):
    pass


def lineage_comparison(records: Mapping[tuple[str, str], dict], reference: Mapping[tuple[str, str], dict],
                       diag: dict | None, reference_diag: dict | None) -> dict:
    """Compare re-evaluations of a restored switch state with the records of that same state."""
    ce_diffs = {f"{role}-{domain}": abs(records[role, domain]["ce"] - reference[role, domain]["ce"])
                for role, domain in reference if (role, domain) in records}
    energy = {}
    if diag is not None and reference_diag is not None:
        for site, views in reference_diag["sites"].items():
            if not site.endswith(".h"):
                continue
            for domain in ("web", "python"):
                if domain in views["all"] and domain in diag["sites"][site]["all"]:
                    ref = views["all"][domain]["mean_squared_norm"]
                    new = diag["sites"][site]["all"][domain]["mean_squared_norm"]
                    energy[f"{site}/{domain}"] = abs(new - ref) / max(abs(ref), 1e-12)
    max_ce = max(ce_diffs.values()) if ce_diffs else None
    max_energy = max(energy.values()) if energy else None
    passed = bool(ce_diffs) and max_ce <= LINEAGE_CE_TOLERANCE and \
        (max_energy is None or max_energy <= LINEAGE_ENERGY_TOLERANCE)
    return {"compared": sorted(ce_diffs), "max_ce_diff": max_ce, "max_energy_rel_diff": max_energy,
            "energy_compared": len(energy), "diag_compared": max_energy is not None, "passed": passed}


def lineage_reference(seed: int, condition: str, run_dir: Path, pilot: PilotReference):
    """Records of the switch state a branch starts from: the pilot's (seeds 101-103) or the run's own."""
    if seed in PILOT_SEEDS:
        log_ = pilot.events(seed, condition)
        diag_domain = "both"
    else:
        log_ = EventLog(run_dir / "events.jsonl")
        diag_domain = "all"
    reference = {(role, domain): log_.get("prefix", PREFIX_END, role, domain)
                 for role in ("full", "quick") for domain in ("web", "python")}
    diag = log_.get("prefix", PREFIX_END, "diag", diag_domain)
    return ({k: v for k, v in reference.items() if v is not None},
            diag["measurement"] if diag is not None and diag.get("status") == "ok" else None)


# =============================================================================
# Worker: one condition on one GPU, jobs in protocol order.
# =============================================================================
class Stop(Exception):
    pass


def run_job(seed: int, condition: str, state: Path, data: Data, cfg: dict, pilot: PilotReference,
            deadline: float, stop_flag: list, device: torch.device) -> str:
    run_dir = state / "runs" / f"S{seed}-{condition}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "completion.json").exists():
        return "complete"
    if (run_dir / "failure.json").exists():
        raise RuntimeError(f"S{seed}-{condition} has a recorded failure; investigate before continuing")
    identity = {"config_sha256": cfg["config_sha256"], "seed": seed, "condition": condition}
    stages = job_stages(seed)
    orders = data.seed_orders(seed)
    model, optimizer, scaler = make_state(seed, condition, device)
    events = EventLog(run_dir / "events.jsonl")
    counters = {"optimizer_seconds": 0.0, "updates": 0, "clipped": 0, "overflow_retries": 0}
    if (run_dir / "latest.pt").exists():
        progress = load_checkpoint(run_dir / "latest.pt", model, optimizer, scaler, identity)
        log(f"S{seed}-{condition}: resumed at {progress['stage']} step {progress['step']} "
            f"(global update {int(model.completed_updates)})")
    elif seed in PILOT_SEEDS:
        switch_sha = pilot.load_switch(seed, condition, model, optimizer, scaler)
        progress = {"stage": stages[0], "step": 0, **counters, "parent_sha256": switch_sha}
        log(f"S{seed}-{condition}: branch 'x' starts from the pilot's switch {switch_sha[:12]}")
    else:
        progress = {"stage": "prefix", "step": 0, **counters, "parent_sha256": None}
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
            diag_pending = diag and not events.has(stage, step, "diag", "all")
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
                        measured = diagnostics(model, data.probes(), data.class_letters, orders["rare"],
                                               eval_microbatch, device)
                        status = "ok"
                    except Exception as exc:  # descriptive only; never blocks the primary endpoint
                        measured, status = {"error": f"{type(exc).__name__}: {exc}"}, "error"
                    events.add({"stage": stage, "step": step, "role": "diag", "domain": "all",
                                "global_update": int(model.completed_updates), "status": status,
                                "measurement": measured, "seconds": time.time() - tick})
                if step in (FULL_PREFIX if stage == "prefix" else FULL_CONTINUATION):
                    rows = [f"{domain} {'non-W ' if domain != 'web' else ''}CE "
                            f"{events.get(stage, step, 'full', domain)['non_w_ce' if domain != 'web' else 'ce']:.4f}"
                            for domain in STAGE_DOMAINS[stage]]
                    log(f"S{seed}-{condition} {stage}@{step}: full-dev " + ", ".join(rows))
            if stage != "prefix" and step == 0 and not events.has(stage, 0, "lineage", "switch"):
                reference, reference_diag = lineage_reference(seed, condition, run_dir, pilot)
                records = {(role, domain): events.get(stage, 0, role, domain) for role, domain in reference
                           if events.has(stage, 0, role, domain)}
                diag_record = events.get(stage, 0, "diag", "all")
                result = lineage_comparison(records, reference,
                                            diag_record["measurement"] if diag_record and
                                            diag_record.get("status") == "ok" else None, reference_diag)
                events.add({"stage": stage, "step": 0, "role": "lineage", "domain": "switch",
                            "parent_sha256": progress["parent_sha256"], **result})
                log(f"S{seed}-{condition} {stage}@0 lineage: max |dCE| {result['max_ce_diff']}, max rel dE "
                    f"{result['max_energy_rel_diff']} -> {'ok' if result['passed'] else 'FAILED'}")
                if not result["passed"]:
                    raise LineageError(f"S{seed}-{condition} {stage}: the restored switch state does not "
                                       f"re-evaluate to its records ({result})")
            if step == stage_limit(stage):
                if stage == "prefix":
                    if not (run_dir / "switch.pt").exists():
                        checkpoint("switch.pt")
                else:
                    save_weights(run_dir / f"{stage}-final.pt", model, identity)
                position = stages.index(stage) + 1
                if position == len(stages):
                    switch_sha = pilot.switch_sha256(seed, condition) if seed in PILOT_SEEDS \
                        else read_json(run_dir / "switch.pt.json")["sha256"]
                    write_json(run_dir / "completion.json", {
                        "seed": seed, "condition": condition, "config_sha256": cfg["config_sha256"],
                        "stages": list(stages), "switch_sha256": switch_sha,
                        "updates": progress["updates"], "optimizer_seconds": progress["optimizer_seconds"],
                        "clipped": progress["clipped"], "overflow_retries": progress["overflow_retries"],
                        "completed_unix": time.time()})
                    for name in ("latest.pt", "latest.pt.json"):
                        (run_dir / name).unlink(missing_ok=True)
                    log(f"S{seed}-{condition}: COMPLETE")
                    return "complete"
                next_stage = stages[position]
                kept = {k: progress[k] for k in ("optimizer_seconds", "updates", "clipped", "overflow_retries")}
                switch_sha = read_json(run_dir / "switch.pt.json")["sha256"]
                load_checkpoint(run_dir / "switch.pt", model, optimizer, scaler, identity)
                progress = {**kept, "stage": next_stage, "step": 0, "parent_sha256": switch_sha}
                checkpoint()
                log(f"S{seed}-{condition}: branch '{next_stage}' starts from switch {switch_sha[:12]}")
                continue
            domain, offset = stage_data(stage, step)
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
    data = Data(Path(args.pilot_online), Path(args.pilot_orders), Path(args.s2_online), Path(args.s2_orders))
    pilot = PilotReference(Path(args.pilot_state), pins=cfg["pilot"])
    only = {int(s) for s in args.only_seeds.split(",")} if args.only_seeds else None
    jobs = [(seed, condition) for seed in SEEDS if only is None or seed in only
            for condition in args.conditions.split(",")]
    for seed, condition in jobs:
        try:
            status = run_job(seed, condition, state, data, cfg, pilot, args.deadline, stop_flag, device)
        except Stop:
            log(f"worker {args.conditions}: paused at deadline/stop; state saved")
            return 0
        except FloatingPointError as exc:
            write_json(state / "runs" / f"S{seed}-{condition}" / "failure.json", {
                "type": "numerical", "message": str(exc), "traceback": traceback.format_exc(),
                "action": "Numerical failure: invalidates a clean H2 interpretation. Investigate; never replace "
                          "the seed or change thresholds."})
            log(f"NUMERICAL FAILURE S{seed}-{condition}: {exc}")
            return 3
        except LineageError as exc:
            write_json(state / "runs" / f"S{seed}-{condition}" / "failure.json", {
                "type": "lineage", "message": str(exc), "traceback": traceback.format_exc(),
                "action": "Correctness failure: a restored switch state does not reproduce its own records. "
                          "Investigate the environment/data before any further training."})
            log(f"LINEAGE FAILURE S{seed}-{condition}: {exc}")
            return 3
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if status != "complete":
            return 0
    log(f"worker {args.conditions}: all assigned runs complete")
    return 0


# =============================================================================
# Analysis primitives (verbatim or minimally generalized from kaggle_h1/h1_run.py)
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
    """Full-dev web CE of the six models a contrast needs; "shift" is the branch that switched domain."""
    rms_prefix: float
    taper_prefix: float
    rms_web: float
    rms_shift: float
    taper_web: float
    taper_shift: float

    def __post_init__(self) -> None:
        if any(not math.isfinite(x) or x < 0 for x in self.__dict__.values()):
            raise ValueError("All branch values must be finite nonnegative web CE")


@dataclass(frozen=True)
class Contrast:
    d: float
    g_web: float
    q: float
    f_rms_web: float
    f_rms_shift: float
    f_taper_web: float
    f_taper_shift: float


def contrast(ce: BranchCE) -> Contrast:
    d = (ce.taper_shift - ce.taper_web) - (ce.rms_shift - ce.rms_web)
    g = (ce.taper_prefix - ce.rms_prefix) - (ce.taper_web - ce.rms_web)
    frw, frs = ce.rms_web - ce.rms_prefix, ce.rms_shift - ce.rms_prefix
    ftw, fts = ce.taper_web - ce.taper_prefix, ce.taper_shift - ce.taper_prefix
    q = d - g
    if not math.isclose(q, fts - frs, rel_tol=0, abs_tol=1e-6):
        raise ValueError("Forgetting decomposition does not reconstruct")
    return Contrast(d, g, q, frw, frs, ftw, fts)


@dataclass(frozen=True)
class SeedUncertainty:
    values: tuple
    mean: float
    sample_sd: float
    descriptive_t95: tuple


def seed_uncertainty(values: Sequence[float], multiplier: float) -> SeedUncertainty:
    """Mean, SD and mean +/- multiplier * SD / sqrt(n) (the pre-registered t quantile for this n)."""
    if len(values) < 2 or not all(math.isfinite(v) for v in values):
        raise ValueError("The descriptive interval requires at least two finite paired seeds")
    mean, sd = statistics.mean(values), statistics.stdev(values)
    halfwidth = multiplier * sd / math.sqrt(len(values))
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


def classify(seeds: Sequence[SeedEvidence], *, expected_seeds: Sequence[int], primary_complete: bool,
             correctness_and_data_passed: bool, nonfinite_primary: bool = False, resource_terminated: bool = False,
             thresholds: Mapping[str, float] = DECISION_THRESHOLDS) -> Decision:
    """The pilot's ordered decision hierarchy; ``expected_seeds`` replaces the pilot's fixed three."""
    t = thresholds
    invalid = []
    if not primary_complete or len(seeds) != len(expected_seeds) or {s.seed for s in seeds} != set(expected_seeds):
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
                        ("All screening criteria met; no confirmatory claim",), target, stats)
    if mean_d < t["small_d"]:
        transient = any(statistics.mean(s.quick_d[u] for s in seeds) >= t["transient_d"]
                        for u in QUICK_CONTINUATION if u <= TRANSIENT_MAX_UPDATE)
        category = "TRANSIENT ONLY" if transient else "STOP — SMALL OBSERVED EFFECT"
        return Decision(category, (f"Mean endpoint D < {t['small_d']}; not evidence of equivalence",), target, stats)
    return Decision("INCONCLUSIVE", ("Screening criteria not all met",), target, stats)


H2_ANSWERS = {
    "PROCEED TO DESIGN THE NEXT STUDY": "YES - H2 supported at the screening level: after the switch to the selected "
                                        "domain, Taper-minus shows more persistent web forgetting than RMS, meeting "
                                        "every pre-registered criterion.",
    "OPPOSITE DIRECTION": "NO - the effect goes the other way: Taper-minus forgets LESS than RMS after the switch to the "
                          "selected domain (mean D_X <= -0.03).",
    "STOP — SMALL OBSERVED EFFECT": "NO - no persistent excess forgetting >= 0.015 nats/token for Taper-minus after the "
                                    "switch to the selected domain (see the pre-registered equivalence analysis "
                                    "before reading this as an absence of effect).",
    "TRANSIENT ONLY": "NO (for persistent forgetting) - only a transient early excess (<= update 1000) that does not "
                      "persist to the endpoint.",
    "INCONCLUSIVE": "UNDECIDED - valid, comparable data, but the result falls between the pre-registered YES and NO "
                    "criteria. Per protocol: no automatic extra seeds.",
    "COMPARABILITY / ADAPTATION LIMITED": "CANNOT ANSWER - a validity guardrail failed (prefix gap > 2% or adaptation "
                                          "to the selected domain < 0.05 nats).",
    "INVALID OR INCOMPLETE": "NOT YET - the experiment is incomplete, or a run failed numerically or a correctness/"
                             "data/lineage gate failed.",
}

R2_ANSWERS = {
    "PROCEED TO DESIGN THE NEXT STUDY": "YES - on the fresh seeds 104-106, Taper-minus shows more persistent web "
                                        "forgetting after the Python switch than RMS, meeting every pilot criterion: "
                                        "the pilot's H1 answer does NOT replicate.",
    "OPPOSITE DIRECTION": "NO - on the fresh seeds, Taper-minus forgets LESS than RMS after the Python switch "
                          "(mean D <= -0.03).",
    "STOP — SMALL OBSERVED EFFECT": "NO - on the fresh seeds, no persistent excess forgetting >= 0.015 nats/token: the "
                                    "pilot's H1 answer replicates.",
    "TRANSIENT ONLY": "NO (for persistent forgetting) - on the fresh seeds, only a transient early excess.",
    "INCONCLUSIVE": "UNDECIDED - the fresh seeds fall between the pilot's YES and NO criteria.",
    "COMPARABILITY / ADAPTATION LIMITED": "CANNOT ANSWER - a validity guardrail failed on a fresh seed.",
    "INVALID OR INCOMPLETE": "NOT YET - the fresh-seed web and Python branches are incomplete or a run failed.",
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


def check_record(record: dict, *, stage: str, step: int, role: str, parent: str | None) -> None:
    """The pilot report's integrity checks for one evaluation record."""
    expected_update = step if stage == "prefix" else PREFIX_END + step
    if record["global_update"] != expected_update:
        raise ValueError("evaluation clock mismatch")
    expected = (FULL_WINDOWS if role == "full" else QUICK_WINDOWS) * CONTEXT
    stats = stats_from_record(record, None)
    if stats.total.count != expected or sum(v.count for v in stats.classes.values()) != expected:
        raise ValueError("wrong evaluation label population")
    if abs(sum(v.ce_sum for v in stats.classes.values()) - stats.total.ce_sum) / expected > 1e-6:
        raise ValueError("class sums do not reconstruct CE")
    if abs(stats.total.mean - record["ce"]) > 1e-9:
        raise ValueError("CE does not match sufficient statistics")
    if role == "full" and (sum(record["doc_counts"]) != expected
                           or abs(math.fsum(record["doc_sums"]) - stats.total.ce_sum) / expected > 1e-6):
        raise ValueError("document sums do not reconstruct CE")
    if stage != "prefix" and record.get("parent_sha256") != parent:
        raise ValueError("branch evaluation does not descend from its frozen switch state")


def expected_records(seed: int) -> list[tuple[str, int, str, str]]:
    """Every (stage, step, role, domain) evaluation the protocol schedules for one seed and condition.

    For seeds 101-103 the prefix, web and Python stages are the pilot's records."""
    keys = []
    for stage, steps in (("prefix", FULL_PREFIX), ("web", FULL_CONTINUATION), ("python", FULL_CONTINUATION),
                         ("x", FULL_CONTINUATION)):
        keys += [(stage, step, "full", domain) for step in steps for domain in STAGE_DOMAINS[stage]]
    for stage, steps in (("prefix", (PREFIX_END,)), ("web", QUICK_CONTINUATION), ("python", QUICK_CONTINUATION),
                         ("x", QUICK_CONTINUATION)):
        keys += [(stage, step, "quick", domain) for step in steps for domain in STAGE_DOMAINS[stage]]
    return keys


class Records:
    """Evaluation records of every model Study 2 analyses: the pilot's (seeds 101-103: prefix, web and Python
    branches) and Study 2's own. Integrity failures become reported problems, never silent values."""

    def __init__(self, state: Path, pilot: PilotReference):
        self.state, self.pilot = state, pilot
        self.s2 = {(s, c): EventLog(state / "runs" / f"S{s}-{c}" / "events.jsonl") for s in SEEDS for c in CONDITIONS}
        self.missing: list[str] = []
        self.problems: list[str] = []
        self.cache: dict[tuple, dict | None] = {}

    @staticmethod
    def source(seed: int, stage: str) -> str:
        return "pilot" if seed in PILOT_SEEDS and stage != "x" else "s2"

    def log_for(self, seed: int, condition: str, stage: str) -> EventLog:
        return self.pilot.events(seed, condition) if self.source(seed, stage) == "pilot" else self.s2[seed, condition]

    def parent(self, seed: int, condition: str) -> str | None:
        if seed in PILOT_SEEDS:
            return self.pilot.switch_sha256(seed, condition)
        sidecar = self.state / "runs" / f"S{seed}-{condition}" / "switch.pt.json"
        return read_json(sidecar)["sha256"] if sidecar.exists() else None

    def get(self, seed: int, condition: str, stage: str, step: int, role: str, domain: str) -> dict | None:
        key = (seed, condition, stage, step, role, domain)
        if key in self.cache:
            return self.cache[key]
        name = f"{self.source(seed, stage)}:S{seed}-{condition}/{stage}-{step}-{role}-{domain}"
        record = self.log_for(seed, condition, stage).get(stage, step, role, domain)
        if record is None:
            self.missing.append(name)
        else:
            try:
                check_record(record, stage=stage, step=step, role=role, parent=self.parent(seed, condition))
            except ValueError as exc:
                self.problems.append(f"{name}: {exc}")
                record = None
        self.cache[key] = record
        return record

    def diag(self, seed: int, condition: str, stage: str, step: int) -> dict | None:
        domain = "both" if self.source(seed, stage) == "pilot" else "all"
        record = self.log_for(seed, condition, stage).get(stage, step, "diag", domain)
        return record["measurement"] if record is not None and record.get("status") == "ok" else None


def shift_keys(shift: str) -> list[tuple[str, int, str, str]]:
    """The records one seed's web -> shift contrast reads (per condition)."""
    keys = [("prefix", u, "full", "web") for u in FULL_PREFIX] + [("prefix", PREFIX_END, "quick", "web")]
    if shift == "python":
        keys.append(("prefix", PREFIX_END, "full", "python"))
    for stage, domains in (("web", ("web",)), (shift, ("web", shift))):
        keys += [(stage, u, "full", d) for u in FULL_CONTINUATION for d in domains]
        keys += [(stage, u, "quick", d) for u in QUICK_CONTINUATION for d in domains]
    return keys


def shift_analysis(records: Records, seed: int, shift: str, doc_ids: list[str] | None):
    """One seed's pilot-defined screening evidence for web -> shift ("python" or "x") plus its details.

    The adaptation baseline is the shifted domain's full-dev non-W CE of the switch state: the prefix@9156 record
    for Python (exactly the pilot's definition) and the X branch's s=0 record for X (the prefix never evaluates X)."""
    rows = {}
    for condition in CONDITIONS:
        for key in shift_keys(shift):
            rows[(condition, *key)] = records.get(seed, condition, *key)
    if any(row is None for row in rows.values()):
        return None, None

    def ce(condition, stage, step, role="full", domain="web", metric="ce"):
        return rows[condition, stage, step, role, domain][metric]

    def effect(step, role="full"):
        return contrast(BranchCE(ce("RMS", "prefix", PREFIX_END, role), ce("Taper-minus", "prefix", PREFIX_END, role),
                                 ce("RMS", "web", step, role), ce("RMS", shift, step, role),
                                 ce("Taper-minus", "web", step, role), ce("Taper-minus", shift, step, role)))

    endpoint = effect(CONTINUATION_UPDATES)
    points = {condition: [AdaptationPoint(step, ce(condition, shift, step, "quick", shift, "non_w_ce"),
                                          ce(condition, shift, step, "quick", "web"))
                          for step in QUICK_CONTINUATION] for condition in CONDITIONS}
    matches = {u: matched_forgetting(points["RMS"], points["Taper-minus"], u) for u in PERSISTENCE_POINTS}
    base = ("prefix", PREFIX_END) if shift == "python" else (shift, 0)
    adaptation = {condition: ce(condition, *base, "full", shift, "non_w_ce")
                  - ce(condition, shift, CONTINUATION_UPDATES, "full", shift, "non_w_ce") for condition in CONDITIONS}
    prefix = {condition: ce(condition, "prefix", PREFIX_END) for condition in CONDITIONS}
    evidence = SeedEvidence(seed, endpoint.d, effect(D_MID_POINT).d, endpoint.q,
                            abs(prefix["Taper-minus"] - prefix["RMS"]) / prefix["RMS"], adaptation["RMS"],
                            adaptation["Taper-minus"], {u: effect(u, "quick").d for u in QUICK_CONTINUATION},
                            {u: pair[1] for u, pair in matches.items()})
    final = {(condition, branch): rows[condition, branch, CONTINUATION_UPDATES, "full", "web"]
             for condition in CONDITIONS for branch in (shift, "web")}
    if doc_ids is not None:
        sums = [stats_from_record(final[key], doc_ids) for key in
                (("Taper-minus", shift), ("Taper-minus", "web"), ("RMS", shift), ("RMS", "web"))]
        d, documents, classes = decompose(*sums)
        if abs(d - endpoint.d) > 1e-6:
            raise ValueError("Document decomposition disagrees with primary contrast")
        tail = asdict(summarize_documents(documents, d))
        class_rows = [asdict(x) for x in classes]
    else:
        tail, class_rows = None, None
    rare = [final[key]["rare"] for key in (("Taper-minus", shift), ("Taper-minus", "web"), ("RMS", shift),
                                           ("RMS", "web"))]
    if len({r[1] for r in rare}) != 1:
        raise ValueError("Different rare-token populations across branches")
    rare_difference = (rare[0][0] - rare[1][0]) - (rare[2][0] - rare[3][0])
    total_labels = final["RMS", "web"]["total"][1]
    slopes = {condition: prefix_slope([(u, ce(condition, "prefix", u)) for u in FULL_PREFIX])
              for condition in CONDITIONS}
    details = {
        "seed": seed, "shift": shift, "endpoint": asdict(endpoint), "evidence": asdict(evidence),
        "full_dev_persistence": {u: asdict(effect(u)) for u in PERSISTENCE_POINTS},
        "full_dev_trajectory_d": {u: effect(u).d for u in FULL_CONTINUATION},
        "matched": {u: {"match": asdict(pair[0]) if pair[0] else None, "differential_forgetting": pair[1]}
                    for u, pair in matches.items()},
        "adaptation_non_w_improvement": adaptation,
        "prefix_slopes_nats_per_million_tokens": slopes,
        "prefix_slope_difference": slopes["Taper-minus"] - slopes["RMS"],
        "prefix_full_web_ce": prefix,
        "rare": {"labels": rare[0][1], "mean_d_on_rare": rare_difference / rare[0][1] if rare[0][1] else None,
                 "contribution_to_d": rare_difference / total_labels},
        "tail": tail, "class_contributions": class_rows}
    return evidence, details


def doc_deltas(records: Records, seeds: Sequence[int], shift: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-document four-way CE-sum differences at the endpoint (rows: seeds) and the shared label counts."""
    deltas, counts = [], None
    for seed in seeds:
        rows = {(c, b): records.get(seed, c, b, CONTINUATION_UPDATES, "full", "web")
                for c in CONDITIONS for b in (shift, "web")}
        if any(r is None for r in rows.values()):
            return None
        arr = {k: np.asarray(r["doc_sums"], dtype=np.float64) for k, r in rows.items()}
        deltas.append((arr["Taper-minus", shift] - arr["Taper-minus", "web"])
                      - (arr["RMS", shift] - arr["RMS", "web"]))
        seed_counts = np.asarray(rows["RMS", "web"]["doc_counts"], dtype=np.int64)
        if counts is not None and not np.array_equal(counts, seed_counts):
            raise ValueError("Document label counts differ across seeds")
        counts = seed_counts
    keep = counts > 0
    return np.asarray(deltas)[:, keep], counts[keep]


def document_bootstrap(deltas: np.ndarray, counts: np.ndarray, replicates: int, seed: int) -> dict:
    """Resample development documents (the same resample for every seed and arm); models held fixed."""
    rng = np.random.default_rng(seed)
    n_docs = len(counts)
    reps = np.empty((replicates, deltas.shape[0]))
    for start in range(0, replicates, 500):
        size = min(500, replicates - start)
        index = rng.integers(0, n_docs, size=(size, n_docs))
        weights = np.stack([np.bincount(row, minlength=n_docs) for row in index])
        reps[start:start + size] = (weights @ deltas.T) / (weights @ counts)[:, None]
    means = reps.mean(1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"replicates": replicates, "seed": seed, "documents": n_docs,
            "mean_d_percentile_95": [float(low), float(high)], "mean_d_se": float(means.std(ddof=1)),
            "per_seed_se": reps.std(0, ddof=1).tolist(),
            "interpretation": "Evaluation-sample noise only (trained models fixed); never seed-level uncertainty."}


def interval_summary(values: Sequence[float]) -> dict:
    """Pre-registered descriptive intervals: 95% t (decision-free) and the 90% t equivalence reading (TOST)."""
    n = str(len(values))
    t95, t90 = ANALYSIS["t95"][n], ANALYSIS["t90"][n]
    u95, u90 = seed_uncertainty(values, t95), seed_uncertainty(values, t90)
    bound = ANALYSIS["equivalence_bound"]
    return {"n": len(values), "values": list(values), "mean": u95.mean, "sample_sd": u95.sample_sd,
            "t95_multiplier": t95, "t95": list(u95.descriptive_t95),
            "t90_multiplier": t90, "t90": list(u90.descriptive_t95),
            "equivalence_bound": bound,
            "equivalent_within_bound": bool(-bound < u90.descriptive_t95[0] and u90.descriptive_t95[1] < bound),
            "t95_excludes_zero": bool(u95.descriptive_t95[0] > 0 or u95.descriptive_t95[1] < 0)}


SITES = tuple(f"{i}.{b}" for i in range(6) for b in ("attention", "mlp"))


def scale_measures(records: Records, seeds: Sequence[int], shift: str) -> dict:
    """Finding A's activation-scale measures (pilot paper, Sec. 5.3), pre-registered here as the manipulation check.

    Natural-log ratios of per-site scales of the input h to the 12 internal normalizers on the probe windows:
    the shift-vs-web gap at the switch, the change of the web-probe scale over each branch (relative to the switch,
    prefix@9156 diagnostics) and its domain-specific part (shift branch minus web branch)."""
    result = {}
    for condition in CONDITIONS:
        gap, web_change, shift_change, specific, used = [], [], [], [], []
        for seed in seeds:
            switch = records.diag(seed, condition, "prefix", PREFIX_END)
            gap_source = records.diag(seed, condition, shift, 0) if shift == "x" else switch
            web_end = records.diag(seed, condition, "web", CONTINUATION_UPDATES)
            shift_end = records.diag(seed, condition, shift, CONTINUATION_UPDATES)
            if any(m is None for m in (switch, gap_source, web_end, shift_end)):
                continue
            used.append(seed)
            for site in SITES:
                def energy(measurement, domain, site=site):
                    return measurement["sites"][f"{site}.h"]["all"][domain]["mean_squared_norm"]
                gap.append(0.5 * math.log(energy(gap_source, shift) / energy(gap_source, "web")))
                web_change.append(0.5 * math.log(energy(web_end, "web") / energy(switch, "web")))
                shift_change.append(0.5 * math.log(energy(shift_end, "web") / energy(switch, "web")))
                specific.append(shift_change[-1] - web_change[-1])
        if not used:
            result[condition] = {"seeds": [], "pairs": 0}
            continue
        result[condition] = {
            "seeds": used, "pairs": len(gap),
            "switch_gap_abs": float(np.mean(np.abs(gap))), "switch_gap_signed": float(np.mean(gap)),
            "web_branch_change_abs": float(np.mean(np.abs(web_change))),
            "shift_branch_change_abs": float(np.mean(np.abs(shift_change))),
            "web_branch_change_signed": float(np.mean(web_change)),
            "shift_branch_change_signed": float(np.mean(shift_change)),
            "specific_change_abs": float(np.mean(np.abs(specific))),
            "web_branch_shrank": int(np.sum(np.array(web_change) < 0)),
            "shift_branch_shrank": int(np.sum(np.array(shift_change) < 0))}
    return result


def clipping_by_phase(state: Path) -> dict:
    """Per-phase gradient-clipping fractions from Study 2's own training logs (descriptive)."""
    phases = {"prefix_calibration": lambda r: r["stage"] == "prefix" and r["u"] <= CALIBRATION_END,
              "prefix_gate_decay": lambda r: r["stage"] == "prefix" and CALIBRATION_END < r["u"] < GATE_END,
              "prefix_gate_zero": lambda r: r["stage"] == "prefix" and r["u"] >= GATE_END,
              "web": lambda r: r["stage"] == "web", "python": lambda r: r["stage"] == "python",
              "x": lambda r: r["stage"] == "x"}
    result = {}
    for condition in CONDITIONS:
        fractions = {name: [] for name in phases}
        for seed in SEEDS:
            path = state / "runs" / f"S{seed}-{condition}" / "train.jsonl"
            if not path.exists():
                continue
            rows = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows[row["stage"], row["u"]] = row  # a resumed session re-logs replayed updates: keep the last
            for name, keep in phases.items():
                selected = [r["clip"] for r in rows.values() if keep(r)]
                if selected:
                    fractions[name].append(float(np.mean(selected)))
        result[condition] = {name: {"mean": float(np.mean(v)), "min": min(v), "max": max(v), "runs": len(v)}
                             for name, v in fractions.items() if v}
    return result


def build_report(state: Path, pilot: PilotReference, doc_ids: list[str] | None = None) -> dict:
    cfg = read_json(state / "config.json")
    thresholds = cfg["decision_thresholds"]
    records = Records(state, pilot)
    failures, progress = [], []
    for seed in SEEDS:
        for condition in CONDITIONS:
            run_dir = state / "runs" / f"S{seed}-{condition}"
            if (run_dir / "failure.json").exists():
                failures.append(read_json(run_dir / "failure.json") | {"run": f"S{seed}-{condition}"})
            for stage, step, role, domain in expected_records(seed):
                records.get(seed, condition, stage, step, role, domain)
            for stage in job_stages(seed):
                done = [step for step in (FULL_PREFIX if stage == "prefix" else FULL_CONTINUATION)
                        if records.s2[seed, condition].has(stage, step, "full", "web")]
                if done:
                    step = max(done)
                    progress.append({"seed": seed, "condition": condition, "stage": stage, "step": step,
                                     "web_full_ce": records.s2[seed, condition].get(stage, step, "full", "web")["ce"]})
    incomplete_runs = [f"S{s}-{c}" for s in SEEDS for c in CONDITIONS
                       if not (state / "runs" / f"S{s}-{c}" / "completion.json").exists()]
    # Lineage: every branch's s=0 re-evaluation must reproduce the records of the switch state it restored.
    lineage = {}
    for seed in SEEDS:
        for condition in CONDITIONS:
            for stage in job_stages(seed):
                if stage == "prefix":
                    continue
                result = records.s2[seed, condition].get(stage, 0, "lineage", "switch")
                if result is not None:
                    lineage[f"S{seed}-{condition}/{stage}"] = {k: result[k] for k in
                                                               ("max_ce_diff", "max_energy_rel_diff", "passed")}
                    if not result["passed"]:
                        records.problems.append(f"Lineage check failed: S{seed}-{condition}/{stage}")
                elif (state / "runs" / f"S{seed}-{condition}" / "completion.json").exists():
                    records.problems.append(f"Lineage check missing: S{seed}-{condition}/{stage}")
    numerical = any(f.get("type") == "numerical" for f in failures)

    def run_shift(seeds, shift):
        evidence, details = [], []
        for seed in seeds:
            try:
                row, detail = shift_analysis(records, seed, shift, doc_ids)
            except ValueError as exc:
                records.problems.append(f"Seed {seed} {shift} analysis: {exc}")
                continue
            if row is not None:
                evidence.append(row)
                details.append(detail)
        return evidence, details

    h2_evidence, h2_details = run_shift(SEEDS, "x")
    py_evidence, py_details = run_shift(SEEDS, "python")
    r2_evidence = [e for e in py_evidence if e.seed in FRESH_SEEDS]
    if len(py_evidence) == len(SEEDS):
        # The pilot seeds' web -> Python contrast, recomputed from the pilot's records, must equal the published one.
        published = pilot.decision_report()
        for row in (published or {}).get("seeds", []):
            mine = next(e for e in py_evidence if e.seed == row["seed"])
            if abs(mine.d - row["endpoint"]["d"]) > 1e-12:
                records.problems.append(f"Pilot seed {row['seed']}: recomputed D differs from the pilot's report")
    correctness = not failures and not records.problems
    h2_complete = not records.missing and not incomplete_runs and len(h2_evidence) == len(SEEDS)
    r2_complete = len(r2_evidence) == len(FRESH_SEEDS)
    h2 = classify(h2_evidence, expected_seeds=SEEDS, primary_complete=h2_complete,
                  correctness_and_data_passed=correctness, nonfinite_primary=numerical, thresholds=thresholds)
    r2 = classify(r2_evidence, expected_seeds=FRESH_SEEDS, primary_complete=r2_complete,
                  correctness_and_data_passed=correctness, nonfinite_primary=numerical, thresholds=thresholds)
    fresh_h2 = [e.d for e in h2_evidence if e.seed in FRESH_SEEDS]
    manipulation = {"x": scale_measures(records, SEEDS, "x"), "python": scale_measures(records, SEEDS, "python")}
    if all(manipulation[d][c].get("pairs") for d in ("x", "python") for c in CONDITIONS):
        manipulation["reading"] = {
            "switch_gap_larger_for_x": all(manipulation["x"][c]["switch_gap_abs"]
                                           > manipulation["python"][c]["switch_gap_abs"] for c in CONDITIONS),
            "specific_change_larger_for_x": all(manipulation["x"][c]["specific_change_abs"]
                                                > manipulation["python"][c]["specific_change_abs"]
                                                for c in CONDITIONS)}
    bootstrap = None
    if h2_complete and doc_ids is not None:
        deltas = doc_deltas(records, SEEDS, "x")
        if deltas is not None:
            bootstrap = document_bootstrap(*deltas, ANALYSIS["bootstrap_replicates"], ANALYSIS["bootstrap_seed"])
    result = {
        "protocol": cfg["protocol"], "config_sha256": cfg["config_sha256"], "domain": cfg["domain"],
        "complete": h2_complete and r2_complete,
        "h2": {"decision": asdict(h2), "answer": H2_ANSWERS[h2.category], "complete": h2_complete,
               "seeds": h2_details,
               "uncertainty": interval_summary([e.d for e in h2_evidence]) if len(h2_evidence) == len(SEEDS) else None,
               "fresh_seed_sensitivity": interval_summary(fresh_h2) if len(fresh_h2) == len(FRESH_SEEDS) else None},
        "r2": {"decision": asdict(r2), "answer": R2_ANSWERS[r2.category], "complete": r2_complete,
               "seeds": [d for d in py_details if d["seed"] in FRESH_SEEDS],
               "uncertainty": interval_summary([e.d for e in r2_evidence]) if r2_complete else None},
        "pooled_python_descriptive": interval_summary([e.d for e in sorted(py_evidence, key=lambda e: e.seed)])
        | {"note": "Seeds 101-103 are the pilot's published values; descriptive only, never a decision input."}
        if len(py_evidence) == len(SEEDS) else None,
        "manipulation_check": manipulation, "document_bootstrap": bootstrap,
        "lineage": lineage, "clipping": clipping_by_phase(state),
        "missing_event_count": len(records.missing), "missing_events_first_50": records.missing[:50],
        "incomplete_runs": incomplete_runs, "failures": failures, "problems": records.problems,
        "latest_observations": progress, "thresholds": thresholds, "analysis": cfg["analysis"],
        "interpretation": "Fixed-sample development-data screening with pre-registered rules. Positive D = more "
                          "excess web forgetting for Taper-minus under the domain shift."}
    lines = [f"H2 ANSWER (web -> {cfg['domain']['id']}): {result['h2']['answer']}",
             f"H2 decision category: {h2.category}",
             f"R2 ANSWER (web -> Python, seeds 104-106): {result['r2']['answer']}",
             f"R2 decision category: {r2.category}",
             f"Complete: H2 {h2_complete}, R2 {r2_complete}; seed groups with full evidence: H2 {len(h2_evidence)}/6, "
             f"R2 {len(r2_evidence)}/3; missing events: {len(records.missing)}; incomplete runs: {len(incomplete_runs)}"]
    for label, evidence in (("H2", h2_evidence), ("R2", r2_evidence)):
        for s in evidence:
            lines.append(f"{label} seed {s.seed}: D={s.d:+.5f}  D3050={s.d3050:+.5f}  Q={s.q:+.5f}  prefix gap="
                         f"{s.absolute_relative_prefix_gap:.2%}  adaptation RMS={s.rms_non_w_improvement:.4f} "
                         f"Taper={s.taper_non_w_improvement:.4f}  matched="
                         f"{ {u: (round(v, 5) if v is not None else None) for u, v in s.matched_differences.items()} }")
    for label, block in (("H2", result["h2"]["uncertainty"]), ("R2", result["r2"]["uncertainty"]),
                         ("Pooled web->Python (descriptive)", result["pooled_python_descriptive"])):
        if block:
            lines.append(f"{label}: mean D = {block['mean']:+.5f}, SD {block['sample_sd']:.5f}, t95 "
                         f"[{block['t95'][0]:+.5f}, {block['t95'][1]:+.5f}], t90 [{block['t90'][0]:+.5f}, "
                         f"{block['t90'][1]:+.5f}] (equivalent within +/-{block['equivalence_bound']}: "
                         f"{block['equivalent_within_bound']})")
    for label, decision in (("H2", h2), ("R2", r2)):
        for key, val in decision.statistics.items():
            lines.append(f"{label} {key} = {val:+.5f}")
        lines += [f"{label} reason: {r}" for r in decision.reasons]
    if "reading" in manipulation:
        lines.append(f"Manipulation check: {manipulation['reading']}")
    for f in failures:
        lines.append(f"FAILURE {f.get('run')}: {f.get('message')}")
    for p in records.problems[:20]:
        lines.append(f"PROBLEM: {p}")
    result["summary_lines"] = lines
    return result


def write_report(state: Path, pilot: PilotReference, pilot_online: Path | None) -> dict:
    doc_ids = None
    if pilot_online is not None:
        manifest = read_json(pilot_online / "manifest.json")
        doc_ids = [doc["id"] for doc in read_json(pilot_online / manifest["splits"]["web_dev"]["documents"])["documents"]]
    report = build_report(state, pilot, doc_ids)
    write_json(state / "decision-report.json", report)
    (state / "decision-report.txt").write_text("\n".join(report["summary_lines"]) + "\n", encoding="utf-8")
    return report


# =============================================================================
# Orchestrator
# =============================================================================
STATE_MARKER = "S2-STATE.json"
BOOTSTRAP_MARKER = "S2-BOOTSTRAP.json"


def safe_report(state: Path, pilot: PilotReference, pilot_online: Path | None):
    """Analysis must never block training or lose state."""
    try:
        return write_report(state, pilot, pilot_online)
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
        raise RuntimeError(f"Multiple different Study 2 experiments attached as input: {ids}. Attach only one.")
    return max(candidates, key=lambda p: read_json(p / STATE_MARKER).get("sessions_completed", 0))


def run_updates(seed: int) -> int:
    return PREFIX_END + 3 * CONTINUATION_UPDATES if seed in FRESH_SEEDS else CONTINUATION_UPDATES


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


def remaining_updates(state: Path, condition: str, seeds: Iterable[int] = SEEDS) -> int:
    total = 0
    for seed in seeds:
        run_dir = state / "runs" / f"S{seed}-{condition}"
        if (run_dir / "completion.json").exists():
            continue
        if (run_dir / "latest.pt.json").exists():
            p = read_json(run_dir / "latest.pt.json")["progress"]
            stages = job_stages(seed)
            done = sum(stage_limit(s) for s in stages[:stages.index(p["stage"])]) + p["step"]
            total += run_updates(seed) - done
        else:
            total += run_updates(seed)
    return total


def resolve_inputs(args) -> dict[str, Path]:
    explicit = {"pilot_online": args.pilot_online, "pilot_orders": args.pilot_orders, "s2_online": args.s2_online,
                "s2_orders": args.s2_orders, "pilot_state": args.pilot_state}
    if all(v is not None for v in explicit.values()):
        return {k: Path(v) for k, v in explicit.items()}
    found = locate_inputs(Path(args.inputs))
    return {k: Path(v) if v is not None else found[k] for k, v in explicit.items()}


def main(args) -> int:
    t0 = float(os.environ.get("S2_T0") or time.time())
    session_hours = float(os.environ.get("S2_SESSION_HOURS", "11.25"))
    allow_fresh = os.environ.get("S2_ALLOW_FRESH_START", "0") == "1"
    only_seeds = os.environ.get("S2_ONLY_SEEDS", "")  # smoke tests only; never set for the experiment
    inputs = Path(args.inputs)
    work = Path(args.work)
    state = work / "s2state"
    work_deadline = t0 + session_hours * 3600
    hard_deadline = work_deadline + (15 * 60 if not MINI else 60)
    log(f"Study 2 runner {RUNNER_VERSION}; session budget {session_hours:.2f} h "
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
                log("ERROR: no previous Study 2 state in the inputs. The previous version's output is missing "
                    "(was it killed?). Attach this notebook's latest version that HAS an 's2state' folder. "
                    "Refusing to silently start a new experiment.")
                return 4
            fresh = True

    # 2) Identity of every input: the pilot's assets and Study 2's arrays.
    paths = resolve_inputs(args)
    for key, path in paths.items():
        log(f"{key}: {path}")
    tick = time.time()
    pilot = PilotReference(paths["pilot_state"])
    pilot_pins = pilot.verify()
    data_identity = verify_data(paths["pilot_online"], paths["pilot_orders"], paths["s2_online"], paths["s2_orders"])
    domain = domain_identity(paths["s2_online"])
    log(f"Pilot assets and data hashes verified in {time.time() - tick:.0f}s; selected domain {domain['id']} "
        f"({domain['repo']} @ {domain['revision'][:12]})")

    if fresh:
        state.mkdir(parents=True, exist_ok=True)
        cfg = config_document(args.microbatch, SEQUENCES_PER_UPDATE // args.microbatch, args.eval_microbatch,
                              pilot_pins, domain)
        cfg["data"] = data_identity
        cfg["config_sha256"] = digest_json(cfg)
        cfg["checkpoint_seconds"] = 900
        write_json(state / "config.json", cfg)
        write_json(state / STATE_MARKER, {"schema": "s2-state-1", "experiment_id": cfg["config_sha256"][:16]
                                          + "-" + hex(int(time.time()))[2:], "created_unix": time.time(),
                                          "sessions_completed": 0})
        log(f"NEW experiment created; config sha256 {cfg['config_sha256'][:16]}")
    cfg = read_json(state / "config.json")
    marker = read_json(state / STATE_MARKER)
    expected = config_document(cfg["microbatch"], cfg["accumulation"], cfg["eval_microbatch"], pilot_pins, domain)
    expected["data"] = data_identity
    if digest_json(expected) != cfg["config_sha256"]:
        log("ERROR: the frozen configuration or data identity differs from this code/data. Refusing to mix strata.")
        return 5

    session = {"session": marker.get("sessions_completed", 0) + 1, "started_unix": t0,
               "runner": RUNNER_VERSION, "code_sha256": sha256_file(Path(__file__)),
               "python": sys.version.split()[0], "torch": torch.__version__,
               "cuda": torch.version.cuda, "gpus": gpu_count(), "only_seeds": only_seeds or None}
    try:
        session["gpu_names"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True).strip().splitlines()
    except Exception:
        session["gpu_names"] = []
    log(f"Session {session['session']}: torch {session['torch']} CUDA {session['cuda']}, GPUs {session['gpu_names']}")
    report = safe_report(state, pilot, paths["pilot_online"])
    if report is not None and (report["complete"] or report["failures"]):
        log("Nothing to train: experiment already finished or has a recorded failure.")
        for line in report["summary_lines"]:
            print(line, flush=True)
        return 0

    n_gpus = session["gpus"]
    if os.environ.get("S2_TEST_TWO_WORKERS_ONE_GPU") == "1" and n_gpus == 1:
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
                   "--pilot-online", str(paths["pilot_online"]), "--pilot-orders", str(paths["pilot_orders"]),
                   "--s2-online", str(paths["s2_online"]), "--s2-orders", str(paths["s2_orders"]),
                   "--pilot-state", str(paths["pilot_state"]), "--conditions", conditions,
                   "--deadline", str(worker_deadline)] + (["--only-seeds", only_seeds] if only_seeds else []) \
            + (["--mini"] if MINI else [])
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
    hard_failure = False
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
                hard_failure = True
                log("Numerical or lineage failure recorded; stopping the other worker to preserve quota.")
                for other in procs.values():
                    other.send_signal(signal.SIGTERM)
            elif code not in (0,) and not hard_failure and \
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
    report = safe_report(state, pilot, paths["pilot_online"]) or {
        "summary_lines": ["(report failed; see traceback above)"], "complete": False, "failures": []}
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


def lineage_main(args) -> int:
    """Re-evaluate every pilot switch state at s=0 (web/Python dev, diagnostics) and compare with the pilot's
    records. Forward passes only. The main run performs the same check before each X branch trains."""
    paths = resolve_inputs(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pilot = PilotReference(paths["pilot_state"])
    pilot.verify()
    data = Data(paths["pilot_online"], paths["pilot_orders"], paths["s2_online"], paths["s2_orders"])
    results = {}
    for seed in PILOT_SEEDS:
        for condition in CONDITIONS:
            model, optimizer, scaler = make_state(seed, condition, device)
            pilot.load_switch(seed, condition, model, optimizer, scaler)
            rare = data.seed_orders(seed)["rare"]
            reference, reference_diag = lineage_reference(seed, condition, Path("."), pilot)
            records = {(role, domain): evaluate(model, data, domain, role, rare, args.eval_microbatch, device)
                       for role, domain in reference}
            diag = diagnostics(model, data.probes(), data.class_letters, rare, args.eval_microbatch, device)
            results[f"S{seed}-{condition}"] = lineage_comparison(records, reference, diag, reference_diag)
            log(f"S{seed}-{condition}: {results[f'S{seed}-{condition}']}")
            del model, optimizer, scaler
            if device.type == "cuda":
                torch.cuda.empty_cache()
    passed = all(r["passed"] for r in results.values())
    write_json(Path(args.output), {"passed": passed, "torch": torch.__version__, "cuda": torch.version.cuda,
                                   "device": str(device), "runs": results})
    log(f"Lineage {'PASSED' if passed else 'FAILED'} -> {args.output}")
    return 0 if passed else 7


def add_input_arguments(parser) -> None:
    parser.add_argument("--inputs", default="/kaggle/input")
    for name in ("--pilot-online", "--pilot-orders", "--s2-online", "--s2-orders", "--pilot-state"):
        parser.add_argument(name, default=None)


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("main")
    add_input_arguments(m)
    m.add_argument("--work", default="/kaggle/working")
    m.add_argument("--microbatch", type=int, default=8)
    m.add_argument("--eval-microbatch", type=int, default=16)
    m.add_argument("--mini", action="store_true")
    w = sub.add_parser("worker")
    w.add_argument("--state", required=True)
    for name in ("--pilot-online", "--pilot-orders", "--s2-online", "--s2-orders", "--pilot-state"):
        w.add_argument(name, required=True)
    w.add_argument("--conditions", required=True)
    w.add_argument("--deadline", type=float, required=True)
    w.add_argument("--only-seeds", default="")
    w.add_argument("--mini", action="store_true")
    li = sub.add_parser("lineage")
    add_input_arguments(li)
    li.add_argument("--output", required=True)
    li.add_argument("--eval-microbatch", type=int, default=16)
    li.add_argument("--mini", action="store_true")
    r = sub.add_parser("report")
    r.add_argument("--state", required=True)
    r.add_argument("--pilot-state", required=True)
    r.add_argument("--pilot-online", default=None)
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
    elif arguments.command == "lineage":
        sys.exit(lineage_main(arguments))
    else:
        state_dir = Path(arguments.state)
        pilot_ref = PilotReference(Path(arguments.pilot_state), pins=read_json(state_dir / "config.json")["pilot"])
        result = write_report(state_dir, pilot_ref, Path(arguments.pilot_online) if arguments.pilot_online else None)
        print("\n".join(result["summary_lines"]))
