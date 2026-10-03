"""Forward-statistic helpers; probe selection/hook orchestration remains pending."""

from dataclasses import dataclass
import math
import torch
from torch import Tensor


@dataclass(frozen=True)
class EnergySummary:
    positions: int
    mean_squared_norm: float
    norm_p50: float
    norm_p99: float
    p99_over_p50: float | None


def sensitivity_mask(inputs: Tensor, *, eos: int = 50256) -> Tensor:
    if inputs.ndim != 2:
        raise ValueError("Expected batch by sequence inputs")
    mask = torch.ones_like(inputs, dtype=torch.bool)
    mask[:, :16] = False
    mask[:, 1:] &= inputs[:, :-1] != eos
    return mask


@torch.no_grad()
def energy_summary(activations: Tensor, mask: Tensor | None = None) -> EnergySummary:
    if activations.ndim != 3:
        raise ValueError("Expected batch, sequence, channel activations")
    if mask is not None and (mask.shape != activations.shape[:2] or mask.dtype != torch.bool):
        raise ValueError("Mask shape/type mismatch")
    with torch.autocast(device_type=activations.device.type, enabled=False):
        x = activations.detach().float()
        squared = x.square().sum(dim=-1)
        values = squared.flatten() if mask is None else squared[mask]
        if not values.numel() or not bool(torch.isfinite(values).all()):
            raise ValueError("Empty or nonfinite diagnostic pool")
        norms = values.sqrt()
        p50, p99 = (float(v) for v in torch.quantile(norms, norms.new_tensor([0.5, 0.99])))
        return EnergySummary(values.numel(), float(values.mean()), p50, p99, p99 / p50 if p50 > 0 else None)


def log_energy_shift(web: EnergySummary, python: EnergySummary) -> tuple[float, bool]:
    ew, ec = web.mean_squared_norm, python.mean_squared_norm
    if not all(math.isfinite(v) and v >= 0 for v in (ew, ec)):
        raise ValueError("Invalid energy")
    return 0.5 * math.log(max(ew, 1e-12) / max(ec, 1e-12)), min(ew, ec) <= 1e-12
