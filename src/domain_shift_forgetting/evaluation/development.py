"""Explicit CE-only evaluation over caller-supplied development batches."""

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor
from torch.nn import functional as F

from domain_shift_forgetting.models.transformer import Transformer
from .statistics import EvaluationSums


@dataclass(frozen=True)
class DevelopmentBatch:
    inputs: Tensor
    labels: Tensor
    label_document_ids: tuple[str, ...]  # Flattened row-major label order.
    label_classes: tuple[str, ...]
    label_rare: tuple[bool, ...]
    split: str
    role: str
    array_sha256: str


@torch.no_grad()
def evaluate_development(model: Transformer, batches: Iterable[DevelopmentBatch], *,
                         role: str, expected_array_sha256: str,
                         expected_labels: int, bf16: bool = True) -> EvaluationSums:
    if role not in {"quick", "full"}:
        raise ValueError("Only quick/full development evaluation is supported")
    quota = 262144 if role == "quick" else 2097152
    if expected_labels != quota or len(expected_array_sha256) != 64:
        raise ValueError("Require the frozen evaluation quota and array identity")
    device = next(model.parameters()).device
    if bf16 and device.type != "cuda":
        raise ValueError("BF16 scientific evaluation requires the selected CUDA GPU")
    was_training = model.training
    model.eval()
    result = EvaluationSums()
    try:
        for batch in batches:
            if batch.split != "dev" or batch.role != role or batch.array_sha256 != expected_array_sha256:
                raise ValueError("Rejected non-development or mismatched evaluation data")
            if batch.inputs.ndim != 2 or batch.inputs.shape != batch.labels.shape or batch.inputs.shape[1] != 512:
                raise ValueError("Expected aligned 512-label development windows")
            count = batch.labels.numel()
            if any(len(items) != count for items in (batch.label_document_ids, batch.label_classes, batch.label_rare)):
                raise ValueError("Metadata not aligned to evaluation labels")
            if batch.inputs.device != device or batch.labels.device != device:
                raise ValueError("Move evaluation batches to model device explicitly")
            if batch.inputs.dtype != torch.long or batch.labels.dtype != torch.long:
                raise ValueError("Evaluation token IDs must be torch.long")
            if bool(((batch.labels < 0) | (batch.labels >= 50257)).any()):
                raise ValueError("Padding/ignored labels cannot enter evaluation CE")
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
                logits, _ = model(batch.inputs, collect=False, auxiliary=False)
                losses = F.cross_entropy(logits.float().reshape(-1, 50257), batch.labels.reshape(-1), reduction="none")
            result.add(losses.cpu().tolist(), batch.label_document_ids, batch.label_classes, batch.label_rare)
            if result.total.count > expected_labels:
                raise ValueError("Evaluation exceeded the frozen label quota")
    finally:
        model.train(was_training)
    if result.total.count != expected_labels:
        raise ValueError("Incomplete evaluation; do not publish a partial corpus CE")
    result.check_reconstruction()
    return result
