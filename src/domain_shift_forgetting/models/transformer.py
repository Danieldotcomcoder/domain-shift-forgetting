"""Fixed six-layer pre-norm causal language model for the H1 pilot."""

import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from domain_shift_forgetting.protocol import CALIBRATION_END, Condition
from .norms import RMSNorm, TaperNorm


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
        self.token_embedding = nn.Embedding(50257, 256)
        self.position_embedding = nn.Embedding(512, 256)
        self.blocks = nn.ModuleList(Block(self.condition) for _ in range(6))
        self.final_norm = RMSNorm(256)
        # Output projection uses token_embedding.weight directly (no duplicate state).
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
        if tokens.ndim != 2 or not 1 <= tokens.shape[1] <= 512:
            raise ValueError("Expected batch by sequence tokens, length 1..512")
        completed = int(self.completed_updates)
        update = max(1, completed) if update is None else update
        if collect and (not self.training or update != completed + 1 or update > CALIBRATION_END):
            raise ValueError("Calibration requires the next training update in warmup")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x, update=update, collect=collect)
        aux = None
        if self.condition == Condition.TAPER_PLUS and (collect or auxiliary):
            with torch.autocast(device_type=x.device.type, enabled=False):
                rms = torch.sqrt(x.float().square().mean(dim=-1) + 1e-6)
                if collect:
                    with torch.no_grad():
                        self.pending_rms.add_(rms.detach().sum())
                        self.pending_positions.add_(rms.numel())
                if auxiliary and update > CALIBRATION_END:
                    if not bool(self.target_frozen):
                        raise RuntimeError("Auxiliary target not frozen")
                    aux = 0.1 * (rms - self.s_target).square().mean()
        logits = F.linear(self.final_norm(x), self.token_embedding.weight)
        return logits, aux

    @torch.no_grad()
    def finish_update(self, update: int) -> None:
        if update != int(self.completed_updates) + 1:
            raise ValueError("Updates must finish sequentially")
        for module in self.modules():
            if isinstance(module, TaperNorm):
                module.finish_update(update)
        if self.condition == Condition.TAPER_PLUS and update <= CALIBRATION_END:
            if int(self.target_updates) != update - 1 or int(self.pending_positions) <= 0:
                raise RuntimeError("Auxiliary EMA requires all warmup update samples")
            self.target_ema.mul_(0.99).add_(self.pending_rms / self.pending_positions, alpha=0.01)
            self.target_updates.add_(1)
            self.pending_rms.zero_()
            self.pending_positions.zero_()
            if update == CALIBRATION_END:
                self.s_target.copy_(self.target_ema / (1 - 0.99 ** int(self.target_updates)))
                self.target_frozen.fill_(True)
        self.completed_updates.fill_(update)


@torch.no_grad()
def copy_canonical_initialization(canonical: Transformer, destination: Transformer) -> None:
    """Explicitly pair all shared tensors; call before constructing optimizers."""
    if canonical.condition != Condition.RMS or int(canonical.completed_updates) != 0 or int(destination.completed_updates) != 0:
        raise ValueError("Require fresh RMS canonical and destination models")
    shared = dict(canonical.named_parameters())
    copied = set()
    for name, parameter in destination.named_parameters():
        if name.endswith("gamma_tilde"):
            parameter.fill_(1.0)  # Inactive until copied from trained gamma at 763.
            continue
        if name not in shared or parameter.shape != shared[name].shape:
            raise ValueError(f"Canonical tensor mismatch: {name}")
        parameter.copy_(shared[name])
        copied.add(name)
    if copied != set(shared):
        raise ValueError("Not every canonical parameter was copied")
