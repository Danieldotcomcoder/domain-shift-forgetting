"""Protocol-transcribed RMS/Taper operators; published-source verification pending."""

import torch
from torch import Tensor, nn

from domain_shift_forgetting.protocol import CALIBRATION_END, gate


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
            raise RuntimeError("Taper requires completed calibration before update 764")
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
            # Match RMSNorm's arithmetic at the RMS endpoint. sqrt followed by
            # division is mathematically equivalent but rounds differently.
            normalized = (x * torch.rsqrt(mean_square)) * self.gamma.float()
            if g == 1.0:
                # gamma_tilde.grad remains None; Adam creates no inactive state.
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
            self.c.copy_((self.numerator_ema / correction) /
                         (self.denominator_ema / correction + 1e-12))
            self.gamma_tilde.copy_(self.gamma)
            self.calibrated.fill_(True)
