"""WSM checkpoint merging (arXiv:2507.17634).

Three weight schemes for averaging the last N constant-LR checkpoints:
- mean:    uniform; approximates linear decay
- ema:     geometric c_j ~ alpha^(N-1-j); approximates exponential decay
- theorem: from paper's Theorem 3.1; approximates 1 - sqrt(t) decay
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Sequence

import torch


def get_weights(method: str, n: int, ema_alpha: float = 0.5) -> List[float]:
    """Mixing weights for n checkpoints sorted oldest -> newest. Sums to 1."""
    if n < 1:
        raise ValueError("need >= 1 checkpoint")
    if method == "mean":
        return [1.0 / n] * n
    if method == "ema":
        raw = [ema_alpha ** (n - 1 - j) for j in range(n)]
        s = sum(raw)
        return [r / s for r in raw]
    if method == "theorem":
        # Theorem 3.1: given gradient envelope w_1 >= ... >= w_k in [0, 1],
        #   c_0 = 1 - w_1,  c_j = w_j - w_{j+1},  c_k = w_k.
        # For 1 - sqrt(t) decay matched to wsd_schedule(decay_type="sqrt",
        # final_lr_factor=0): the i-th gradient (i=1..k) is applied with LR
        # factor (1 - sqrt((i-1)/k)), since PyTorch advances the scheduler
        # AFTER optimizer.step(). So w_1 = 1 (first gradient gets peak LR),
        # w_k = 1 - sqrt((k-1)/k). This puts zero weight on the oldest ckpt
        # (c_0 = 1 - w_1 = 0), consistent with WSD's first decay step.
        k = n - 1
        if k == 0:
            return [1.0]
        w = [1.0 - math.sqrt((i - 1) / k) for i in range(1, k + 1)]
        cs = [1.0 - w[0]] + [w[j - 1] - w[j] for j in range(1, k)] + [w[-1]]
        cs = [max(0.0, c) for c in cs]
        s = sum(cs)
        return [c / s for c in cs]
    raise ValueError(f"unknown method: {method}")


def load_model_state(ckpt_path) -> dict:
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return obj["model"] if isinstance(obj, dict) and "model" in obj else obj


@torch.no_grad()
def merge_state_dicts(state_dicts: Sequence[dict], weights: Sequence[float]) -> dict:
    """Weighted average; float params in fp32 then cast back, int buffers taken from latest."""
    out = {}
    for key, ref in state_dicts[-1].items():
        if not torch.is_tensor(ref) or not ref.is_floating_point():
            out[key] = ref.clone() if torch.is_tensor(ref) else ref
            continue
        acc = torch.zeros_like(ref, dtype=torch.float32)
        for w, sd in zip(weights, state_dicts):
            acc.add_(sd[key].to(torch.float32), alpha=float(w))
        out[key] = acc.to(ref.dtype)
    return out


def discover_ckpts(exp_dir) -> List[Path]:
    """Permanent ckpt dirs under exp_dir/ckpts/<iter>/, sorted by iter ascending."""
    entries = []
    for p in (Path(exp_dir) / "ckpts").iterdir():
        if not p.is_dir() or p.name == "latest":
            continue
        try:
            it = int(p.name)
        except ValueError:
            continue
        if (p / "main.pt").exists():
            entries.append((it, p))
    return [p for _, p in sorted(entries)]
