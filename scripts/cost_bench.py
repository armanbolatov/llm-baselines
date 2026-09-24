"""Measured cost of one training step at the 124M DDP config: FLOPs, communication, step time.

The optimizers and flags are those of the 124M DDP campaign (scripts/ddp_train_124m.sh);
SignMuon P=2 does the same work per step as LionMuon P=2, so it is not listed separately.

  flops  one process. FLOPs counted by torch.utils.flop_counter on the code that runs: forward +
         backward of one sequence, times the global batch (FLOPs are linear in it), plus the
         optimizer step (with gradient clipping) averaged over 10 steps, which covers whole P=2
         and P=5 cycles. Matmul-type ops only, the counter's convention (elementwise updates
         count 0); QR is added for Dion. Single-device FLOPs, i.e. what the algorithm costs,
         not how a DDP implementation spreads it over ranks.
  comm   torchrun. Bytes handed to collectives per step, by source and dtype: DDP gradient
         buckets through a counting comm hook, optimizer all-reduces through a counting wrapper
         of torch.distributed.all_reduce. Payload per call; a ring all-reduce puts
         2(N-1)/N of it on the wire per rank.
  time   torchrun. Time of a full step (forward, backward, gradient sync, clipping, optimizer
         step and its all-reduces) in blocks of 10 steps, the slowest rank's time. Every round
         runs each method once in a random order shared by all ranks, so load from other jobs
         on a shared node falls on all methods alike; ratios are taken within a round.

    python scripts/cost_bench.py --mode flops --out results/benchmarks/cost_flops_124m.json
    NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=4 scripts/cost_bench.py --mode comm --out ...
    torchrun --nproc_per_node=4 scripts/cost_bench.py --mode time --rounds 30 --out ...
"""
import argparse
import json
import os
import random
import statistics as st
import sys
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import config  # noqa: E402
from models.utils import get_model  # noqa: E402
from optim.dion import Dion  # noqa: E402
from optim.lion_muon import LionMuon  # noqa: E402
from optim.muonbp import MuonBP  # noqa: E402

TRAIN_FLAGS = ("--config_format base --model base --n_layer 12 --n_head 12 --n_embd 768 "
               "--sequence_length 512 --weight_decay 0.1 --grad_clip 0.5 --muon_ns_steps 5 "
               "--lr 1e-3").split()
METHODS = ["adamw", "muon", "lionmuon_p2", "lionmuon_p5", "muonbp_p5", "dion", "dion_r2"]
STEPS = 10          # one block: whole cycles of P=2 and P=5
SEQ = 512


def model_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config_format", default="base", choices=config.registered_formats())
    args, rem = parser.parse_known_args(TRAIN_FLAGS)
    return config.parse_args_with_format(format=args.config_format, base_parser=parser,
                                         args=rem, namespace=args)


def build(method, raw):
    """Same constructor arguments as src/main.py with the flags of scripts/ddp_train_124m.sh."""
    params = list(raw.parameters())
    backup = dict(weight_decay=0.1, adamw_params=None, adamw_lr=1e-3, adamw_betas=(0.8, 0.999),
                  adamw_eps=1e-8, adamw_wd=0.1)
    if method == "adamw":
        named = dict(raw.named_parameters())
        specs = raw.get_parameter_group_specs()
        for g in specs:
            g["params"] = [named[n] for n in g["params"]]
        return torch.optim.AdamW(specs, lr=1e-3, betas=(0.8, 0.999), weight_decay=0.1, fused=True)
    if method in ("muon", "lionmuon_p2", "lionmuon_p5"):
        k, lr, lion_lr, beta2 = {"muon": (1, 1e-3, None, 0.9),
                                 "lionmuon_p2": (2, 1e-3, 1e-4, 0.99),
                                 "lionmuon_p5": (5, 3e-3, 1e-4, 0.99)}[method]
        return LionMuon(muon_params=params, lr=lr, lion_lr=lion_lr, beta1=0.9, beta2=beta2,
                        ns_steps=5, muon_every_k=k, nesterov=False, srank_alpha=0.0, **backup)
    if method == "muonbp_p5":
        return MuonBP(muon_params=params, lr=1e-3, momentum=0.9, n_blocks=4, period=5,
                      block_lr_ratio=1.0, ns_steps=5, **backup)
    if method == "dion":
        return Dion(muon_params=params, lr=1e-3, rank_frac=0.25, **backup)
    if method == "dion_r2":
        return Dion(muon_params=params, lr=1e-3, rank_frac=0.5, **backup)
    raise ValueError(method)


def train_step(model, raw, opt, method, x, y, ctx, distributed):
    if distributed:            # Dion syncs low-rank factors instead of gradients (see main.py)
        model.require_backward_grad_sync = method not in ("dion", "dion_r2")
    with ctx:
        loss = model(x, targets=y)["loss"]
    loss.backward()
    torch.nn.utils.clip_grad_norm_(raw.parameters(), 0.5)
    opt.step()
    opt.zero_grad(set_to_none=True)


def _qr_flops(a_shape, *args, out_shape=None, **kwargs):
    """Householder QR of an m x n matrix and the thin Q: about 4 k^2 (m - k/3), k = min(m, n)."""
    m, n = a_shape[-2], a_shape[-1]
    k = min(m, n)
    batch = 1
    for s in a_shape[:-2]:
        batch *= s
    return int(batch * 4 * k * k * (max(m, n) - k / 3))


def run_flops(args, out):
    from torch.utils.flop_counter import FlopCounterMode
    qr = {getattr(torch.ops.aten, name).default: _qr_flops
          for name in ("linalg_qr", "geqrf") if hasattr(torch.ops.aten, name)}
    dev = "cuda:0"
    torch.manual_seed(0)
    raw = get_model(model_args()).to(dev)
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    x = torch.randint(0, 50304, (1, SEQ), device=dev)
    y = torch.randint(0, 50304, (1, SEQ), device=dev)
    with FlopCounterMode(display=False) as fc:
        with ctx:
            loss = raw(x, targets=y)["loss"]
        loss.backward()
    fwd_bwd = fc.get_total_flops() * args.global_batch
    raw.zero_grad(set_to_none=True)
    n_params = sum(p.numel() for p in raw.parameters())
    res = {"mode": "flops", "global_batch": args.global_batch, "seq": SEQ, "n_params": n_params,
           "fwd_bwd_per_step": fwd_bwd, "six_N_T": 6 * n_params * SEQ * args.global_batch,
           "methods": {}}
    for m in METHODS:
        opt = build(m, raw)
        opt_flops = 0
        for i in range(1 + STEPS):          # step 0 initialises state (Dion's first QR), not counted
            with ctx:
                loss = raw(x, targets=y)["loss"]
            loss.backward()
            with (FlopCounterMode(display=False, custom_mapping=qr) if i else nullcontext()) as fc:
                torch.nn.utils.clip_grad_norm_(raw.parameters(), 0.5)
                opt.step()
            if i:
                opt_flops += fc.get_total_flops()
            opt.zero_grad(set_to_none=True)
        res["methods"][m] = {"optimizer_per_step": opt_flops / STEPS,
                             "total_per_step": fwd_bwd + opt_flops / STEPS}
        print(f"{m:<12} optimizer {opt_flops / STEPS:.3e}  total {fwd_bwd + opt_flops / STEPS:.4e}", flush=True)
        del opt
    print(f"fwd+bwd {fwd_bwd:.4e} (6NT {res['six_N_T']:.4e})")
    return res


def setup_ddp(args):
    dist.init_process_group(backend="nccl")
    rank, local = dist.get_rank(), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    torch.manual_seed(0)                     # same init on every rank, as DDP broadcasts anyway
    raw = get_model(model_args()).cuda()
    model = torch.nn.parallel.DistributedDataParallel(raw, device_ids=[local])
    b = args.global_batch // dist.get_world_size()
    g = torch.Generator(device="cuda").manual_seed(rank)
    x = torch.randint(0, 50304, (b, SEQ), device="cuda", generator=g)
    y = torch.randint(0, 50304, (b, SEQ), device="cuda", generator=g)
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return rank, local, model, raw, x, y, ctx


def run_comm(args, out):
    from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
    rank, _, model, raw, x, y, ctx = setup_ddp(args)
    tally = {}
    in_hook = [False]

    def add(source, t):
        key = f"{source}:{str(t.dtype).replace('torch.', '')}"
        c = tally.setdefault(key, [0, 0])
        c[0] += t.numel() * t.element_size()
        c[1] += 1

    def hook(state, bucket):
        add("grad_sync", bucket.buffer())
        in_hook[0] = True
        try:
            return default_hooks.allreduce_hook(state, bucket)
        finally:
            in_hook[0] = False

    real_all_reduce = dist.all_reduce

    def counting_all_reduce(tensor, *a, **kw):
        if not in_hook[0]:
            add("optimizer", tensor)
        return real_all_reduce(tensor, *a, **kw)

    model.register_comm_hook(state=None, hook=hook)
    dist.all_reduce = counting_all_reduce
    res = {"mode": "comm", "world_size": dist.get_world_size(), "global_batch": args.global_batch,
           "methods": {}}
    for m in METHODS:
        opt = build(m, raw)
        for _ in range(2):                   # state init and DDP's bucket rebuild after step 1
            train_step(model, raw, opt, m, x, y, ctx, True)
        tally.clear()
        for _ in range(STEPS):
            train_step(model, raw, opt, m, x, y, ctx, True)
        torch.cuda.synchronize()
        per = {k: {"MB_per_step": v[0] / STEPS / 1e6, "calls_per_step": v[1] / STEPS}
               for k, v in sorted(tally.items())}
        res["methods"][m] = {"by_source": per,
                             "total_MB_per_step": sum(v["MB_per_step"] for v in per.values())}
        if rank == 0:
            print(f"{m:<12} {res['methods'][m]['total_MB_per_step']:8.1f} MB/step  "
                  + "  ".join(f"{k} {v['MB_per_step']:.1f} MB x{v['calls_per_step']:g}"
                              for k, v in per.items()), flush=True)
        del opt
    dist.all_reduce = real_all_reduce
    return res if rank == 0 else None


def _ci(vals, n=2000):
    rng = random.Random(0)
    meds = sorted(st.median(rng.choices(vals, k=len(vals))) for _ in range(n))
    return [meds[int(0.025 * n)], meds[int(0.975 * n) - 1]]


def run_time(args, out):
    rank, local, model, raw, x, y, ctx = setup_ddp(args)
    opts = {m: build(m, raw) for m in METHODS}

    def block(m):
        torch.cuda.synchronize()
        dist.barrier(device_ids=[local])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(STEPS):
            train_step(model, raw, opts[m], m, x, y, ctx, True)
        torch.cuda.synchronize()
        dt = torch.tensor([time.perf_counter() - t0], device="cuda")
        dist.all_reduce(dt, op=dist.ReduceOp.MAX)        # the slowest rank sets the step time
        return dt.item() / STEPS

    for _ in range(2):
        for m in METHODS:
            block(m)
    rows = []
    t_start = time.time()
    for r in range(args.rounds):
        order = METHODS[:]
        random.Random(1000 + r).shuffle(order)            # same order on every rank
        rows.append({m: block(m) for m in order})
        if rank == 0 and (r + 1) % 5 == 0:
            print(f"round {r + 1}/{args.rounds}  " + "  ".join(f"{m} {rows[-1][m]:.3f}" for m in METHODS),
                  flush=True)
    res = {"mode": "time", "world_size": dist.get_world_size(), "global_batch": args.global_batch,
           "steps_per_block": STEPS, "rounds": args.rounds, "wall_s": time.time() - t_start,
           "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE", "0"), "summary": {}, "rounds_raw": rows}
    for m in METHODS:
        v = [row[m] for row in rows]
        q = st.quantiles(v, n=4)
        rel_muon = [row[m] / row["muon"] for row in rows]
        rel_adamw = [row[m] / row["adamw"] for row in rows]
        res["summary"][m] = {"s_per_step_median": st.median(v), "s_per_step_iqr": [q[0], q[2]],
                             "vs_muon_median": st.median(rel_muon), "vs_muon_ci95": _ci(rel_muon),
                             "vs_adamw_median": st.median(rel_adamw), "vs_adamw_ci95": _ci(rel_adamw)}
        if rank == 0:
            s = res["summary"][m]
            print(f"{m:<12} {s['s_per_step_median']:.4f} s/step [IQR {q[0]:.4f}-{q[2]:.4f}]  "
                  f"vs Muon {s['vs_muon_median']:.3f} [{s['vs_muon_ci95'][0]:.3f}, {s['vs_muon_ci95'][1]:.3f}]  "
                  f"vs AdamW {s['vs_adamw_median']:.3f} [{s['vs_adamw_ci95'][0]:.3f}, {s['vs_adamw_ci95'][1]:.3f}]")
    return res if rank == 0 else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["flops", "comm", "time"], required=True)
    p.add_argument("--global_batch", type=int, default=32)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    res = {"flops": run_flops, "comm": run_comm, "time": run_time}[args.mode](args, args.out)
    if res is not None:
        res["host"] = os.uname().nodename
        res["gpu"] = torch.cuda.get_device_name(0)
        res["date"] = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(res, f, indent=1)
            print(f"wrote {args.out}")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
