"""Explicit full-state persistence; use only trusted, locally produced checkpoints."""

from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import random
import uuid

import numpy as np
import torch

from domain_shift_forgetting.models.transformer import Transformer


@dataclass(frozen=True)
class DataCursor:
    array_manifest_sha256: str
    order_sha256: str
    next_window: int

    def __post_init__(self) -> None:
        if self.next_window < 0:
            raise ValueError("Negative data cursor")
        for digest in (self.array_manifest_sha256, self.order_sha256):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Data identities require lowercase SHA-256")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_complete(path: Path, *, model: Transformer, optimizer: torch.optim.AdamW,
                  cursors: dict[str, DataCursor], references: dict[str, str]) -> str:
    """Save at a completed-update boundary; caller owns durable upload verification."""
    if path.exists():
        raise FileExistsError("Use a new checkpoint name; retain previous evidence")
    required = {"code_commit", "config_sha256", "manifest_sha256", "run_id", "attempt_id"}
    if not required <= references.keys() or any(not references[k] for k in required):
        raise ValueError("Missing checkpoint provenance")
    if not cursors:
        raise ValueError("Require explicit data cursors, including branch stream state")
    for name, buffer in model.named_buffers():
        if "pending_" in name and bool(torch.any(buffer != 0)):
            raise ValueError("Cannot checkpoint an unfinished accumulation/calibration update")
    payload = {
        "schema_version": 1,
        "condition": model.condition.value,
        "completed_updates": int(model.completed_updates),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "optimizer_class": "AdamW",
        "model_training": model.training,
        "scaler": None,  # BF16 protocol uses no loss scaler.
        "cursors": {name: asdict(cursor) for name, cursor in cursors.items()},
        "references": dict(references),
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Same-directory hard-link publication is atomic and refuses to replace
        # evidence on both Windows and POSIX. Unsupported filesystems fail closed.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return file_sha256(path)


def restore_complete(path: Path, *, expected_sha256: str, model: Transformer,
                     optimizer: torch.optim.AdamW,
                     expected_references: dict[str, str]) -> dict[str, DataCursor]:
    """Restore trusted pickle state, checking identity before changing live state.

    Instantiate models/optimizer before this call so their RNG consumption is
    overwritten. Both branch children must restore the same prefix checksum.
    """
    if file_sha256(path) != expected_sha256:
        raise ValueError("Checkpoint checksum mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["schema_version"] != 1 or payload["condition"] != model.condition.value:
        raise ValueError("Checkpoint schema/condition mismatch")
    if any(payload["references"].get(k) != v for k, v in expected_references.items()):
        raise ValueError("Checkpoint provenance mismatch")
    if payload["completed_updates"] != int(payload["model"]["completed_updates"]):
        raise ValueError("Checkpoint clock mismatch")
    cursors = {name: DataCursor(**record) for name, record in payload["cursors"].items()}
    cuda_rng = payload["rng"]["torch_cuda"]
    if cuda_rng is not None and (not torch.cuda.is_available() or len(cuda_rng) != torch.cuda.device_count()):
        raise ValueError("CUDA RNG topology differs; do not silently change strata")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    optimizer.zero_grad(set_to_none=True)
    model.train(payload["model_training"])
    random.setstate(payload["rng"]["python"])
    np.random.set_state(payload["rng"]["numpy"])
    torch.set_rng_state(payload["rng"]["torch_cpu"])
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    return cursors
