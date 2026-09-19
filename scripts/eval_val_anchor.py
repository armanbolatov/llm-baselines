"""Anchor our validation numbers: score a model on the full FineWeb val split and on the
exact subset a training-time eval uses.

Why: a training eval scores cfg.eval_batches x batch_size windows only (256 windows = 131k
tokens in the 4-GPU DDP runs, where the per-rank batch is 8; 1024 windows = 524k tokens in the
single-GPU grid runs), so an absolute loss carries the sampling error of that subset, and the two
settings do not even use the same windows. The full split here is 10.4M tokens, the same size the
nanogpt-speedrun scores, which makes the number comparable to published ones.

  python scripts/eval_val_anchor.py --model gpt2                         # public GPT-2 124M
  python scripts/eval_val_anchor.py --model ckpt --ckpt exps_ddp124/ddp124_muon_seed0/ckpts/latest/main.pt

Reports, for each sequence length: the mean loss over all non-overlapping windows of the split,
and over the training-eval subsets, so the offset between them is visible.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def load_gpt2():
    """-> f(x, y) = summed token loss. fp32 cross entropy, a few sequences at a time so the
    fp32 logits of a whole batch never have to fit in memory."""
    from transformers import GPT2LMHeadModel
    m = GPT2LMHeadModel.from_pretrained("gpt2").cuda().eval()

    def fwd(x, y, chunk=4):
        lg = m(x).logits[..., :50257]
        out = 0.0
        for j in range(0, x.size(0), chunk):
            c = lg[j:j + chunk].float()
            out = out + F.cross_entropy(c.reshape(-1, c.size(-1)),
                                        y[j:j + chunk].reshape(-1), reduction="sum")
        return out
    return fwd


def load_ckpt(path, seq):
    """Our own checkpoint, through the repo's model code."""
    import argparse as ap
    import config
    from models.utils import get_model
    parser = ap.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config_format", default="base")
    flags = ("--config_format base --model base --n_layer 12 --n_head 12 --n_embd 768 "
             f"--sequence_length {seq}").split()
    a, rem = parser.parse_known_args(flags)
    args = config.parse_args_with_format(format="base", base_parser=parser, args=rem, namespace=a)
    model = get_model(args).cuda().eval()
    sd = torch.load(path, map_location="cuda", weights_only=False)
    sd = sd.get("model", sd)
    sd = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded {path}: {len(missing)} missing, {len(unexpected)} unexpected keys")
    assert not unexpected, f"unexpected keys in {path}: {unexpected[:5]}"
    # the model's own loss, i.e. the exact computation a training-time eval does, so these
    # numbers are comparable with the losses in summary.json. targets are required: without
    # them GPTBase.forward returns logits for the last position only.
    return lambda x, y: model(x, targets=y)["loss"] * y.numel()


@torch.no_grad()
def loss_over(forward, data, starts, seq, batch):
    """Mean loss per token, where forward(x, y) returns the summed token loss of the batch.
    Accumulates on the GPU so there is one sync at the end."""
    tot = torch.zeros((), device="cuda", dtype=torch.float64)
    n = 0
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    for i in range(0, len(starts) - len(starts) % batch, batch):
        idx = starts[i:i + batch]
        xy = np.stack([data[s:s + seq + 1] for s in idx]).astype(np.int64)
        x = torch.from_numpy(xy[:, :-1]).cuda()
        y = torch.from_numpy(xy[:, 1:]).cuda()
        with ctx:
            tot += forward(x, y).double()
        n += y.numel()
    return (tot / n).item(), n // seq


def eval_subset_starts(data, seq, batch, batches, seed=1337):
    """The windows a training eval sees: DataReader(with_replacement=False), set_step(0)."""
    rng = np.random.default_rng(seed)
    n_valid = len(data) - seq - 1
    order = rng.permutation(n_valid // seq - 1)
    offset = rng.integers(seq)
    return (order[:batch * batches] * seq + offset).tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["gpt2", "ckpt"], default="gpt2")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--val", default="/home/shared/datasets/fineweb-100BT/val.bin")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seqs", default="512,1024")
    p.add_argument("--windows", type=int, default=4000,
                   help="windows of the full split to score, spread evenly over it; 0 = all. "
                        "4000 windows of 512 are 2M tokens, enough for about +-0.01")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    data = np.memmap(args.val, dtype=np.uint16, mode="r")
    res = {"val": args.val, "val_tokens": len(data), "model": args.ckpt or args.model, "runs": []}
    print(f"{args.val}: {len(data)} tokens")
    for seq in [int(s) for s in args.seqs.split(",")]:
        fwd = load_gpt2() if args.model == "gpt2" else load_ckpt(args.ckpt, seq)
        full = list(range(0, len(data) - seq - 1, seq))
        if args.windows and args.windows < len(full):
            step = len(full) / args.windows
            full = [full[int(k * step)] for k in range(args.windows)]
        for name, starts in [(f"full_{len(full)}w", full),
                             ("eval_subset_ddp_b8x32", eval_subset_starts(data, seq, 8, 32)),
                             ("eval_subset_grid_b32x32", eval_subset_starts(data, seq, 32, 32))]:
            loss, n = loss_over(fwd, data, starts, seq, args.batch)
            res["runs"].append({"seq": seq, "set": name, "windows": n, "loss": round(loss, 4)})
            print(f"seq {seq:>4}  {name:<24} {n:>6} windows  loss {loss:.4f}", flush=True)
        del fwd
        torch.cuda.empty_cache()
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
