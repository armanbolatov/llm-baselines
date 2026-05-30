"""WSM checkpoint merging (arXiv:2507.17634).

Three weight schemes for averaging the last N constant-LR checkpoints:
- mean:    uniform; approximates linear decay
- ema:     geometric c_j ~ alpha^(N-1-j); approximates exponential decay
- theorem: from paper's Theorem 3.1; approximates 1 - sqrt(t) decay
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import List, Sequence, Tuple

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


@torch.no_grad()
def online_merge_state_dicts(
    state_dicts: Sequence[dict], weights: Sequence[float]
) -> dict:
    """Mathematically identical to merge_state_dicts, computed incrementally.

    Demonstrates how WSM merging can run *online*: maintain a single fp32
    running buffer and accumulate `w_i * theta_i` as each checkpoint arrives.
    Storage during training drops from N ckpts to 1 buffer.

    Output is bit-identical (modulo fp summation order) to merge_state_dicts.
    """
    if len(state_dicts) != len(weights):
        raise ValueError("length mismatch")
    last = state_dicts[-1]
    running = {}
    for key, ref in last.items():
        if torch.is_tensor(ref) and ref.is_floating_point():
            running[key] = torch.zeros_like(ref, dtype=torch.float32)
    for sd, w in zip(state_dicts, weights):
        wf = float(w)
        if wf == 0.0:
            continue
        for key, buf in running.items():
            buf.add_(sd[key].to(torch.float32), alpha=wf)
    out = {}
    for key, ref in last.items():
        if torch.is_tensor(ref) and ref.is_floating_point():
            out[key] = running[key].to(ref.dtype)
        else:
            out[key] = ref.clone() if torch.is_tensor(ref) else ref
    return out


def sample_indices(weights: Sequence[float], k: int, seed: int = 0) -> List[int]:
    """Sample k indices ~ Categorical(weights) with replacement (inverse CDF)."""
    n = len(weights)
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("non-positive weight sum")
    cdf = []
    acc = 0.0
    for w in weights:
        acc += float(w) / total
        cdf.append(acc)
    cdf[-1] = 1.0
    rng = random.Random(seed)
    out = []
    for _ in range(k):
        u = rng.random()
        lo, hi = 0, n - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if u <= cdf[mid]:
                hi = mid
            else:
                lo = mid + 1
        out.append(lo)
    return out


@torch.no_grad()
def sampled_merge_state_dicts(
    state_dicts: Sequence[dict],
    base_weights: Sequence[float],
    k: int,
    seed: int = 0,
) -> Tuple[dict, List[int], List[float]]:
    """Importance-sampled merge.

    Sample k indices ~ Categorical(base_weights) with replacement, then
    return the uniform average of the sampled checkpoints. By Monte Carlo,
    E[result] = sum(base_weights[j] * theta_j) — same target as the
    deterministic merge, but with variance that decreases as k grows.
    Storage drops from N ckpts to <=k ckpts (since duplicates collapse).
    """
    indices = sample_indices(base_weights, k, seed=seed)
    sampled = [state_dicts[i] for i in indices]
    uniform = [1.0 / k] * k
    merged = merge_state_dicts(sampled, uniform)
    # Effective per-ckpt weight = count(j) / k. Useful as metadata.
    eff = [indices.count(j) / k for j in range(len(state_dicts))]
    return merged, indices, eff


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
