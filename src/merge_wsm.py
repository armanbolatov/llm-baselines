"""Merge the last N permanent ckpts of an experiment. Output is a model-only checkpoint.

Methods:
- mean / ema / theorem : deterministic offline weighted average (paper Theorem 3.1)
- online               : same result as theorem/mean/ema but computed incrementally
                         (shows storage-efficient algorithm; one buffer instead of N)
- sampled              : importance-sampled estimate of the deterministic merge.
                         Sample k indices ~ Categorical(base_weights) with replacement,
                         then uniform-average the sampled ckpts. E[result] = deterministic
                         merge; variance decreases with k. Trades storage for variance.
"""

import argparse
from pathlib import Path

import torch

from optim.merge import (discover_ckpts, get_weights, load_model_state,
                         merge_state_dicts, online_merge_state_dicts,
                         sampled_merge_state_dicts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", required=True)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--method", default="theorem",
                   choices=["mean", "ema", "theorem", "online", "sampled"])
    p.add_argument("--ema_alpha", type=float, default=0.5)
    p.add_argument("--base", default="theorem", choices=["mean", "ema", "theorem"],
                   help="For online/sampled: which distribution to use.")
    p.add_argument("--k", type=int, default=1,
                   help="For sampled: number of Monte Carlo samples.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    exp = Path(args.exp_dir)
    ckpts = discover_ckpts(exp)[-args.n:]
    if len(ckpts) < args.n:
        raise SystemExit(f"need {args.n} ckpts, found {len(ckpts)} under {exp}/ckpts")
    states = [load_model_state(c / "main.pt") for c in ckpts]
    iters = [int(c.name) for c in ckpts]

    if args.method == "sampled":
        base_w = get_weights(args.base, len(states), ema_alpha=args.ema_alpha)
        merged, indices, eff = sampled_merge_state_dicts(states, base_w, args.k, args.seed)
        tag = f"merged_sampled_{args.base}_k{args.k}_s{args.seed}"
        meta = {"method": "sampled", "base": args.base, "k": args.k, "seed": args.seed,
                "base_weights": base_w, "sampled_indices": indices,
                "effective_weights": eff, "iters": iters}
        weights_str = eff
    elif args.method == "online":
        base_w = get_weights(args.base, len(states), ema_alpha=args.ema_alpha)
        merged = online_merge_state_dicts(states, base_w)
        tag = f"merged_online_{args.base}"
        meta = {"method": "online", "base": args.base, "weights": base_w, "iters": iters}
        weights_str = base_w
    else:
        weights = get_weights(args.method, len(states), ema_alpha=args.ema_alpha)
        merged = merge_state_dicts(states, weights)
        tag = f"merged_{args.method}{args.ema_alpha if args.method == 'ema' else ''}_n{len(ckpts)}"
        meta = {"method": args.method, "weights": weights, "iters": iters}
        weights_str = weights

    out = exp / "ckpts" / tag / "main.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": merged, "merge": meta}, out)
    print(f"Wrote {out}  weights={[round(w, 4) for w in weights_str]}")


if __name__ == "__main__":
    main()
