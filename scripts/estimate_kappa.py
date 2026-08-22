"""Empirical estimate of the heavy-tail index kappa
of Assumption 2, from minibatch gradient noise sampled at saved checkpoints.

At each checkpoint: a reference gradient (mean over --ref_batches minibatches)
approximates the true gradient; then --samples minibatch gradients give noise
samples n_i = ||G_i - G_ref||_F. The tail index is estimated with the Hill
estimator on the top --tail_frac order statistics (alpha_hill), reported as
kappa_hat = min(alpha_hill, 2) to match Assumption 2's kappa in (1, 2].

Usage:
    python scripts/estimate_kappa.py --exp exps/wt_base_lionmuon_k2_kappa_ckpts \
        --dataset wikitext --samples 512 --ref_batches 256
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402
from data.utils import get_dataset  # noqa: E402
from models.utils import get_model  # noqa: E402


def build_args(device, dataset, model_name):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config_format", default="base")
    ns, rem = parser.parse_known_args(
        ["--model", model_name, "--device", device, "--dataset", dataset]
    )
    return config.parse_args_with_format("base", parser, rem, ns)


def hill_estimator(samples, tail_frac):
    """Hill tail-index estimate on the top tail_frac fraction of |samples|."""
    x = np.sort(samples)[::-1]
    k = max(int(len(x) * tail_frac), 10)
    logs = np.log(x[:k]) - np.log(x[k])
    return 1.0 / logs.mean()


def minibatch_grad(model, data, idx_rng, bs, seq, device, type_ctx, micro_bs=16):
    """Gradient of one size-bs minibatch, accumulated in micro-batches to fit memory."""
    ix = idx_rng.integers(0, len(data) - seq - 1, size=bs)
    model.zero_grad(set_to_none=True)
    for lo in range(0, bs, micro_bs):
        sub = ix[lo : lo + micro_bs]
        x = torch.stack([torch.from_numpy(data[i : i + seq].astype(np.int64)) for i in sub]).to(device)
        y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + seq].astype(np.int64)) for i in sub]).to(device)
        with type_ctx:
            out = model(x, targets=y)
        (out["loss"] * len(sub) / bs).backward()
    return [p.grad.detach().float().clone() for p in model.parameters() if p.grad is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="experiment dir with ckpts/<iter>/main.pt")
    ap.add_argument("--dataset", default="wikitext")
    ap.add_argument("--model", default="base")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=32, help="paper minibatch size")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--ref_batches", type=int, default=256)
    ap.add_argument("--tail_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/kappa_estimate.json")
    cli = ap.parse_args()

    args = build_args(cli.device, cli.dataset, cli.model)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(cli.device)
    torch.cuda.set_device(device)
    type_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    data = get_dataset(args)["train"]  # np.memmap of uint16 tokens, or a path to one
    if isinstance(data, str):
        data = np.memmap(data, dtype=np.uint16, mode="r")
    idx_rng = np.random.default_rng(cli.seed)

    ckpt_dirs = sorted(Path(cli.exp, "ckpts").glob("[0-9]*"), key=lambda p: int(p.name))
    assert ckpt_dirs, f"no checkpoints under {cli.exp}/ckpts"
    results = []
    for ck in ckpt_dirs:
        model = get_model(args).to(device)
        state = torch.load(ck / "main.pt", map_location=device)["model"]
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
        model.load_state_dict(state)
        model.train()

        # reference (approx. true) gradient
        ref = None
        for _ in range(cli.ref_batches):
            g = minibatch_grad(model, data, idx_rng, cli.batch_size, cli.seq_len, device, type_ctx)
            ref = g if ref is None else [r + x for r, x in zip(ref, g)]
        ref = [r / cli.ref_batches for r in ref]

        # noise norm samples
        norms = []
        for _ in range(cli.samples):
            g = minibatch_grad(model, data, idx_rng, cli.batch_size, cli.seq_len, device, type_ctx)
            sq = sum((x - r).pow(2).sum().item() for x, r in zip(g, ref))
            norms.append(sq ** 0.5)
        norms = np.array(norms)

        alpha_hill = hill_estimator(norms, cli.tail_frac)
        kappa_hat = min(alpha_hill, 2.0)
        # Gaussian reference: excess kurtosis of noise norms ~ 0 under light tails
        z = (norms - norms.mean()) / norms.std()
        res = {
            "ckpt_iter": int(ck.name),
            "alpha_hill": float(alpha_hill),
            "kappa_hat": float(kappa_hat),
            "noise_norm_mean": float(norms.mean()),
            "noise_norm_p99_over_median": float(np.quantile(norms, 0.99) / np.median(norms)),
            "excess_kurtosis": float((z**4).mean() - 3),
            "n_samples": len(norms),
        }
        results.append(res)
        print(json.dumps(res), flush=True)
        del model, ref
        torch.cuda.empty_cache()

    Path(cli.out).parent.mkdir(exist_ok=True)
    with open(cli.out, "w") as f:
        json.dump({"exp": cli.exp, "dataset": cli.dataset, "batch_size": cli.batch_size,
                   "tail_frac": cli.tail_frac, "results": results}, f, indent=2)
    print(f"saved {cli.out}")


if __name__ == "__main__":
    main()
