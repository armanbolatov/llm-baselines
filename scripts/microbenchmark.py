"""Per-step wall-clock microbenchmark.

Times forward / backward / optimizer.step separately on synthetic batches for
AdamW, Lion, Signum, Muon, SignMuon and LionMuon at the paper's 124M config.
For alternating optimizers the per-step optimizer times are recorded
individually so Muon-step vs sign-step cost (and hence the NS share) can be
reported directly. No dataset needed.

Usage:
    python scripts/microbenchmark.py --device cuda:0 --steps 100 [--model base]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402
from models.utils import get_model  # noqa: E402
from optim.lion import Lion  # noqa: E402
from optim.lion_muon import LionMuon  # noqa: E402
from optim.muon import Muon  # noqa: E402
from optim.sign import Signum  # noqa: E402
from optim.sign_muon import SignMuon  # noqa: E402


def build_args(device, model_name):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config_format", default="base")
    ns, rem = parser.parse_known_args(
        ["--model", model_name, "--device", device, "--dataset", "fineweb",
         "--n_layer", "12", "--n_head", "12", "--n_embd", "768"]  # true 124M
    )
    return config.parse_args_with_format("base", parser, rem, ns)


def make_optimizer(name, model, args):
    """Mirror the optimizer construction in src/main.py (paper LRs from App. E)."""
    param_list = list(model.parameters())
    if name in ("adamw", "lion", "signum"):
        group_specs = model.get_parameter_group_specs()
        mapping = {n: p for n, p in model.named_parameters()}
        for g in group_specs:
            g["params"] = [mapping[n] for n in g["params"]]
        if name == "adamw":
            return torch.optim.AdamW(
                group_specs, lr=1e-3, betas=(0.8, 0.999), weight_decay=0.1, fused=True
            )
        if name == "lion":
            return Lion(group_specs, lr=2e-5, betas=(0.9, 0.99), weight_decay=0.1)
        return Signum(group_specs, lr=2e-5, momentum=0.9, weight_decay=0.1)
    adamw_kw = dict(
        adamw_params=None, adamw_lr=1e-3, adamw_betas=(0.8, 0.999),
        adamw_eps=1e-8, adamw_wd=0.1,
    )
    if name == "muon":
        return Muon(
            muon_params=param_list, lr=1e-3, momentum=0.9, nesterov=False,
            ns_steps=5, weight_decay=0.1, **adamw_kw,
        )
    if name.startswith("signmuon"):
        k = int(name.split("_k")[1])
        return SignMuon(
            muon_params=param_list, lr=3e-3, cheap_lr=2e-5, momentum=0.9,
            nesterov=False, ns_steps=5, muon_every_k=k, cheap_mode="sign",
            sign_scaling="muon", weight_decay=0.1, srank_alpha=0.0, **adamw_kw,
        )
    if name.startswith("lionmuon"):
        k = int(name.split("_k")[1])
        return LionMuon(
            muon_params=param_list, lr=3e-3, lion_lr=2e-5, beta1=0.9, beta2=0.99,
            ns_steps=5, muon_every_k=k, weight_decay=0.1, srank_alpha=0.0, **adamw_kw,
        )
    raise ValueError(name)


def bench(name, args, device, steps, warmup, batch):
    torch.manual_seed(0)
    model = get_model(args).to(device)
    model.train()
    opt = make_optimizer(name, model, args)

    x = torch.randint(0, args.vocab_size, batch, device=device)
    y = torch.randint(0, args.vocab_size, batch, device=device)

    # Match the training loop in src/optim/base.py (bf16 autocast).
    type_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    def one_step(timers=None):
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        ev[0].record()
        with type_ctx:
            out = model(x, targets=y)
        ev[1].record()
        out["loss"].backward()
        ev[2].record()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        opt.zero_grad(set_to_none=True)
        ev[3].record()
        if timers is not None:
            torch.cuda.synchronize()
            timers.append(tuple(ev[i].elapsed_time(ev[i + 1]) for i in range(3)))

    for _ in range(warmup):
        one_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    timers = []
    for _ in range(steps):
        one_step(timers)

    peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
    fwd = [t[0] for t in timers]
    bwd = [t[1] for t in timers]
    optt = [t[2] for t in timers]

    res = {
        "optimizer": name,
        "fwd_ms": sum(fwd) / len(fwd),
        "bwd_ms": sum(bwd) / len(bwd),
        "opt_ms": sum(optt) / len(optt),
        "opt_ms_per_step": optt,
        "peak_mem_gb": peak_gb,
    }
    res["step_ms"] = res["fwd_ms"] + res["bwd_ms"] + res["opt_ms"]
    # Alternating methods: split optimizer time into Muon-step vs sign-step cost.
    if "_k" in name:
        k = int(name.split("_k")[1])
        muon_steps = [t for i, t in enumerate(optt) if (warmup + i) % k == 0]
        sign_steps = [t for i, t in enumerate(optt) if (warmup + i) % k != 0]
        if muon_steps and sign_steps:
            res["opt_ms_muon_step"] = sum(muon_steps) / len(muon_steps)
            res["opt_ms_sign_step"] = sum(sign_steps) / len(sign_steps)
    del model, opt
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="base", choices=["base", "llama"])
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--out", default="results/microbenchmark.json")
    cli = ap.parse_args()

    args = build_args(cli.device, cli.model)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(torch.device(cli.device))

    names = ["adamw", "lion", "signum", "muon",
             "signmuon_k2", "lionmuon_k2", "lionmuon_k5"]
    results = []
    for name in names:
        print(f"benchmarking {name} ...", flush=True)
        results.append(bench(name, args, cli.device, cli.steps, cli.warmup,
                             (cli.batch_size, cli.seq_len)))

    gpu = torch.cuda.get_device_name(cli.device)
    tokens = cli.batch_size * cli.seq_len
    print(f"\nGPU: {gpu} | model={cli.model} 124M | batch {cli.batch_size} x seq {cli.seq_len}\n")
    hdr = (f"{'optimizer':<14}{'fwd ms':>8}{'bwd ms':>8}{'opt ms':>8}"
           f"{'opt(Muon)':>11}{'opt(sign)':>11}{'step ms':>9}{'tok/s':>9}{'mem GB':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['optimizer']:<14}{r['fwd_ms']:>8.2f}{r['bwd_ms']:>8.2f}"
              f"{r['opt_ms']:>8.2f}"
              f"{r.get('opt_ms_muon_step', float('nan')):>11.2f}"
              f"{r.get('opt_ms_sign_step', float('nan')):>11.2f}"
              f"{r['step_ms']:>9.2f}{tokens / r['step_ms'] * 1000:>9.0f}"
              f"{r['peak_mem_gb']:>8.2f}")

    os.makedirs(os.path.dirname(cli.out), exist_ok=True)
    with open(cli.out, "w") as f:
        json.dump({"gpu": gpu, "model": cli.model, "batch": cli.batch_size,
                   "seq_len": cli.seq_len, "results": results}, f, indent=2)
    print(f"\nsaved {cli.out}")


if __name__ == "__main__":
    main()
