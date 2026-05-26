"""Merge the last N permanent ckpts of an experiment. Output is a model-only checkpoint."""

import argparse
from pathlib import Path

import torch

from optim.merge import (discover_ckpts, get_weights, load_model_state,
                         merge_state_dicts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", required=True)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--method", default="theorem", choices=["mean", "ema", "theorem"])
    p.add_argument("--ema_alpha", type=float, default=0.5)
    args = p.parse_args()

    exp = Path(args.exp_dir)
    ckpts = discover_ckpts(exp)[-args.n:]
    if len(ckpts) < args.n:
        raise SystemExit(f"need {args.n} ckpts, found {len(ckpts)} under {exp}/ckpts")

    states = [load_model_state(c / "main.pt") for c in ckpts]
    weights = get_weights(args.method, len(states), ema_alpha=args.ema_alpha)
    merged = merge_state_dicts(states, weights)

    tag = f"merged_{args.method}{args.ema_alpha if args.method == 'ema' else ''}_n{len(ckpts)}"
    out = exp / "ckpts" / tag / "main.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"method": args.method, "weights": weights,
            "iters": [int(c.name) for c in ckpts]}
    torch.save({"model": merged, "merge": meta}, out)
    print(f"Wrote {out}  weights={[round(w, 4) for w in weights]}")


if __name__ == "__main__":
    main()
