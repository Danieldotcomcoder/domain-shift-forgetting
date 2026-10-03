"""One effective update; orchestration and scientific-run admission are deferred."""

from dataclasses import dataclass
from typing import Sequence
import math
import torch
from torch import Tensor
from torch.nn import functional as F

from domain_shift_forgetting.models.transformer import Transformer
from domain_shift_forgetting.protocol import CALIBRATION_END, learning_rate


def make_optimizer(model: Transformer) -> torch.optim.AdamW:
    matrices, gains = [], []
    for parameter in model.parameters():
        if parameter.dtype != torch.float32:
            raise ValueError("Protocol requires FP32 master parameters")
        (matrices if parameter.ndim >= 2 else gains).append(parameter)
    return torch.optim.AdamW([
        {"params": matrices, "weight_decay": 0.1},
        {"params": gains, "weight_decay": 0.0},
    ], lr=0.0006, betas=(0.9, 0.95), eps=1e-8, foreach=False)


@dataclass(frozen=True)
class UpdateResult:
    completed_update: int
    supervised_tokens: int
    ce: float
    auxiliary: float
    preclip_gradient_norm: float
    clipped: bool


def effective_update(model: Transformer, optimizer: torch.optim.AdamW,
                     microbatches: Sequence[tuple[Tensor, Tensor]], *,
                     bf16: bool = True) -> UpdateResult:
    """Library primitive, deliberately without data I/O or an experiment loop.

    The future runner must enforce a validated freeze and budget before calling.
    CPU/FP32 is available for deterministic validation, not scientific runs.
    """
    if not microbatches:
        raise ValueError("No microbatches")
    device = next(model.parameters()).device
    if bf16 and device.type != "cuda":
        raise ValueError("Scientific BF16 update requires the selected CUDA GPU")
    labels = 0
    for inputs, targets in microbatches:
        if inputs.ndim != 2 or inputs.shape != targets.shape or inputs.shape[1] != 512:
            raise ValueError("Require aligned 512-label sequences")
        if inputs.device != device or targets.device != device:
            raise ValueError("Move batches to model device explicitly")
        if inputs.dtype != torch.long or targets.dtype != torch.long:
            raise ValueError("Token IDs must be torch.long")
        if bool(((targets < 0) | (targets >= 50257)).any()):
            raise ValueError("All labels must be GPT-2 tokens; padding/ignored labels are forbidden")
        labels += targets.numel()
    if labels != 16384:
        raise ValueError("Effective batch must contain exactly 16,384 labels")
    update = int(model.completed_updates) + 1
    lr = learning_rate(update)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group["lr"] = lr
    ce_total = aux_total = 0.0
    for inputs, targets in microbatches:
        weight = targets.numel() / labels
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
            logits, aux = model(inputs, update=update, collect=update <= CALIBRATION_END,
                                auxiliary=True)
            ce = F.cross_entropy(logits.float().reshape(-1, 50257), targets.reshape(-1))
            loss = ce if aux is None else ce + aux
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite loss; abandon attempt and restore a known full state")
        (loss * weight).backward()
        ce_total += float(ce.detach()) * weight
        aux_total += (float(aux.detach()) if aux is not None else 0.0) * weight
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True))
    if not math.isfinite(grad_norm):
        raise FloatingPointError("Nonfinite gradient norm")
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise FloatingPointError("Nonfinite parameters after optimizer step; retain failed attempt")
    model.finish_update(update)
    return UpdateResult(update, labels, ce_total, aux_total, grad_norm, grad_norm > 1.0)
