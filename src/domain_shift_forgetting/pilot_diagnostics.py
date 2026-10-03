"""Fixed, forward-only training probes; no dev/test exposure or optimizer updates."""
from pathlib import Path
import math

import numpy as np
import torch

from .evaluation.diagnostics import sensitivity_mask


@torch.no_grad()
def diagnostics(model, arrays, rare, *, microbatch, precision, clock=None) -> dict:
    device = next(model.parameters()).device
    model.eval()
    norm_samples, ratios, attention = {}, {}, {}
    current = {"domain": None, "mask": None, "attention": False}
    handles = []

    def collect(name, tensor):
        squared = tensor.detach().float().square().sum(-1)
        for view, values in (("all", squared.flatten()), ("position_mask", squared[current["mask"]])):
            key = (current["domain"], name, view)
            norm_samples.setdefault(key, []).append(values.cpu())

    for index, block in enumerate(model.blocks):
        for branch, norm in (("attention", block.attention_norm), ("mlp", block.mlp_norm)):
            def hook(module, inputs, output, name=f"{index}.{branch}"):
                collect(name + ".h", inputs[0])
                collect(name + ".z", output)
            handles.append(norm.register_forward_hook(hook))
        residual = {}

        def pre_attention(module, inputs, stash=residual):
            stash["h"] = inputs[0].detach().float().square().mean().sqrt()

        def post_attention(module, inputs, output, stash=residual, name=f"{index}.attention"):
            denominator = float(stash["h"])
            numerator = float(output.detach().float().square().mean().sqrt())
            ratios.setdefault((current["domain"], name), []).append(numerator / denominator if denominator else None)

        # Residual input is the norm input, not its normalized output.
        handles.append(block.attention_norm.register_forward_pre_hook(pre_attention))
        handles.append(block.attention.register_forward_hook(post_attention))
        mlp_residual = {}
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
                entropy_values = -(logp.exp() * logp.masked_fill(mask, 0)).sum(-1).mean((0, 2))
                if not torch.isfinite(entropy_values).all():
                    raise FloatingPointError("Nonfinite attention entropy")
                values = entropy_values.cpu().tolist()
            attention.setdefault((current["domain"], name), []).append(values)
        handles.append(block.attention.qkv.register_forward_hook(entropy))
    try:
        for domain in ("web", "python"):
            current["domain"] = domain
            stream = arrays.streams[f"{domain}_train"]
            indices = np.arange(stream.windows - 256, stream.windows)
            if indices[0] < 0:
                raise ValueError("Calibration audit needs 131,072 labels per domain")
            for start in range(0, len(indices), microbatch):
                if clock is not None and clock.must_stop(10):
                    raise TimeoutError("Pause before diagnostic reserve")
                rows = torch.from_numpy(stream.take(indices[start:start + microbatch])).to(device)
                current["mask"] = sensitivity_mask(rows[:, :-1])
                current["attention"] = False
                with torch.autocast(device.type, dtype=torch.float16 if precision == "fp16" else torch.bfloat16,
                                    enabled=precision != "fp32"):
                    model(rows[:, :-1], collect=False)
            # Attention entropy: a separate forward pass on a fixed two-window probe.
            current["attention"] = True
            rows = torch.from_numpy(stream.take(indices[:2])).to(device)
            current["mask"] = sensitivity_mask(rows[:, :-1])
            with torch.autocast(device.type, dtype=torch.float16 if precision == "fp16" else torch.bfloat16,
                                enabled=precision != "fp32"):
                model(rows[:, :-1], collect=False)
        summaries = {}
        for (domain, site, view), chunks in norm_samples.items():
            # Entropy probe also fires hooks; exclude its final two-window contribution.
            values = torch.cat(chunks[:-1])
            if not torch.isfinite(values).all():
                raise FloatingPointError("Nonfinite diagnostic activations")
            norms = values.sqrt()
            p50, p99 = torch.quantile(norms, torch.tensor([0.5, 0.99])).tolist()
            summaries.setdefault(site, {}).setdefault(view, {})[domain] = {
                "positions": len(values), "mean_squared_norm": float(values.mean()), "norm_p50": p50,
                "norm_p99": p99, "p99_over_p50": p99 / p50 if p50 > 0 else None}
        for site in summaries.values():
            for view in site.values():
                web, code = view["web"]["mean_squared_norm"], view["python"]["mean_squared_norm"]
                view["log_k"] = 0.5 * math.log(max(web, 1e-12) / max(code, 1e-12))
                view["near_zero_energy"] = min(web, code) <= 1e-12
        gains = {name: {"min": float(p.min()), "max": float(p.max()), "rms": float(p.float().square().mean().sqrt())}
                 for name, p in model.named_parameters() if p.ndim == 1}
        c = {name: float(b) for name, b in model.named_buffers() if name.endswith(".c")}
        embeddings = model.token_embedding.weight.float().norm(dim=-1).cpu().numpy()
        embedding_groups = {key: {"count": int(mask.sum()), "mean_norm": float(embeddings[mask].mean()) if mask.any() else None}
                            for key, mask in list((k, arrays.classes == k) for k in "WAPX") + [("R", np.asarray(rare))]}
        return {"probe": "last 256 windows of each training stream; exposure may overlap trained documents",
            "gradient_tokens": 0, "forward_labels_per_domain": 131072, "sites": summaries,
            "branch_residual_rms_ratios": {f"{a}/{b}": float(np.mean([v for v in x[:-1] if v is not None]))
                if any(v is not None for v in x[:-1]) else None for (a, b), x in ratios.items()},
            "attention_entropy_nats_per_head": {f"{a}/{b}": np.mean(x, axis=0).tolist() for (a, b), x in attention.items()},
            "gains": gains, "calibration_c": c, "embedding_norms": embedding_groups}
    finally:
        for handle in handles:
            handle.remove()
