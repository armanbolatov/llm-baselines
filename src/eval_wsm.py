"""Evaluate a list of checkpoints on the full validation set. Writes JSON summary."""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch

import config
from data.utils import DataReader, get_dataset
from models.utils import get_model
from optim.utils import eval as eval_loop


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--config_format", default="base")
    args, rem = p.parse_known_args()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--output", default=None)
    args = config.parse_args_with_format(format="base", base_parser=p, args=rem, namespace=args)

    torch.manual_seed(args.seed)
    args.datasets_dir = str(Path(args.datasets_dir).expanduser())
    args.world_size = 1

    val = DataReader(
        data_src=get_dataset(args)["val"],
        batch_size=args.batch_size, sequence_length=args.sequence_length,
        seed=args.data_seed, with_replacement=False, auto_shard=False,
        keep_in_ram=args.data_in_ram,
    )
    model = get_model(args).to(args.device).eval()
    ctx = (torch.amp.autocast("cuda", dtype={"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype])
           if "cuda" in args.device else nullcontext())

    rows = []
    for ck in args.ckpts:
        obj = torch.load(ck, map_location="cpu", weights_only=False)
        sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
        meta = obj.get("merge") if isinstance(obj, dict) else None
        model.load_state_dict(sd, strict=False)
        val.set_step(0)
        acc, loss, pp = eval_loop(model, val, args.device, val.num_batches(), ctx, args)
        print(f"{ck}: loss={loss:.4f} pp={pp:.2f} acc={acc:.4f}")
        rows.append({"ckpt": ck, "val_loss": float(loss), "val_pp": float(pp),
                     "val_acc": float(acc), "merge_meta": meta})

    if args.output:
        out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(out, "w"), indent=2)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
