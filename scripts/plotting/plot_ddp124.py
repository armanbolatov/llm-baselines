"""124M at 1x Chinchilla under 4-GPU DDP: validation loss vs tokens and vs wall-clock.

Reads the round-robin logs written by scripts/ddp_train_124m.sh (one file per
method, several launches per file, each launch preceded by
"Starting Experiment: ddp124_<method>_seed<s>"), or a compact export of them.

Time axis (--time):
  bench   iterations x the step time of the controlled 4-GPU benchmark
          (results/benchmarks/ddp_bench_124m.json, PCIe, 8 seq/GPU). Default:
          the training node is shared, so its own step times mostly measure the
          neighbours; the benchmark isolates the optimizer.
  median  iterations x the method's median s/step over all its training slices
  actual  the summed training time (restarts and replayed iterations excluded)

    python scripts/plotting/plot_ddp124.py logs_ddp124 --export results/ddp124/runs.jsonl   # on the training node
    python scripts/plotting/plot_ddp124.py --runs results/ddp124/runs.jsonl                  # anywhere
"""
import argparse
import glob
import json
import os
import re
import statistics as st
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

TOKENS_PER_ITER = 32 * 512
LABEL = {"adamw": "AdamW", "muon": "Muon (P=1)", "lionmuon_p2": "LionMuon P=2",
         "lionmuon_p5": "LionMuon P=5", "signmuon_p2": "SignMuon P=2"}
# Same colours as plot_results.STYLE, kept here so the script runs on any machine.
STYLE = {"adamw":       {"color": "#888888", "ls": "-",  "lw": 1.8},
         "muon":        {"color": "#8b0000", "ls": "-",  "lw": 2.2},
         "signmuon_p2": {"color": "#cc1111", "ls": "--", "lw": 2.2},
         "lionmuon_p2": {"color": "#1144cc", "ls": "--", "lw": 2.2},
         "lionmuon_p5": {"color": "#3377ee", "ls": "-.", "lw": 2.2},
         "dion":        {"color": "#2e8b57", "ls": ":",  "lw": 2.0},
         "muonbp":      {"color": "#e69500", "ls": ":",  "lw": 2.0}}
ORDER = ["adamw", "muon", "muonbp", "dion", "signmuon_p2", "lionmuon_p2", "lionmuon_p5"]
# Which benchmark row gives a method's step time. SignMuon P=2 does the same work as
# LionMuon P=2 (one buffer, same NS schedule, same all-reduce), so it shares the row.
BENCH_ROW = {"adamw": "adamw", "muon": "muon", "lionmuon_p2": "lionmuon_p2",
             "lionmuon_p5": "lionmuon_p5", "signmuon_p2": "lionmuon_p2",
             "dion": "dion", "muonbp": "muonbp_p5"}

RE_START = re.compile(r"Starting Experiment: (ddp124_(\S+)_seed(\d+))")
RE_TRAIN = re.compile(r"Train: Iter=(\d+) .*wall=([0-9.]+)")
RE_EVAL = re.compile(r"Eval: Iter=(\d+) .*val_loss=([0-9.]+)")


def parse(logfile):
    """-> {(method, seed): {"evals": {iter: loss}, "dt": {end_iter: seconds}}}"""
    runs = defaultdict(lambda: {"evals": {}, "dt": {}})
    key, prev = None, None
    with open(logfile, errors="replace") as f:
        for line in f:
            m = RE_START.search(line)
            if m:
                key, prev = (m.group(2), int(m.group(3))), None   # new launch: no interval across it
                continue
            if key is None:
                continue
            m = RE_TRAIN.search(line)
            if m:
                it, wall = int(m.group(1)), float(m.group(2))
                if prev is not None and it > prev[0]:
                    runs[key]["dt"][it] = wall - prev[1]           # last observation wins
                prev = (it, wall)
                continue
            m = RE_EVAL.search(line)
            if m:
                runs[key]["evals"][int(m.group(1))] = float(m.group(2))
    return runs


def cumulative_hours(dt):
    """iter -> hours of pure training up to that iteration."""
    t, out = 0.0, {0: 0.0}
    for it in sorted(dt):
        t += dt[it]
        out[it] = t / 3600
    return out


def hours_at(cum, it):
    keys = [k for k in cum if k <= it]
    return cum[max(keys)] if keys else 0.0


def records_from_logs(logdir):
    """Compact per-run records: iters, losses, median s/step, actual hours per eval."""
    runs = {}
    for f in sorted(glob.glob(os.path.join(logdir, "*.log"))):
        runs.update(parse(f))
    out = []
    for (method, seed), r in runs.items():
        if len(r["evals"]) < 2 or not r["dt"]:
            continue
        cum = cumulative_hours(r["dt"])
        its = sorted(r["evals"])
        out.append(dict(method=method, seed=seed, iters=its, loss=[r["evals"][i] for i in its],
                        actual_hours=[round(hours_at(cum, i), 4) for i in its],
                        med=st.median(r["dt"].values()) / 50, last_iter=max(r["dt"])))
    return out


def base_of(method):
    return method.split("_")[0] if method.startswith(("dion", "muonbp", "muon_")) else method


def style_of(method):
    return STYLE[method if method in STYLE else base_of(method)]


def label_of(method):
    if method in LABEL:
        return LABEL[method]
    base, lr = method.split("_", 1)
    return {"dion": "Dion", "muonbp": "MuonBP (P=5)", "muon": "Muon (P=1)"}[base] + f" lr={lr}"


def bench_step_time(bench_file, ebs=32):
    rows = json.load(open(bench_file))
    return {r["opt"]: r["s_per_step"] for r in rows if r["ebs"] == ebs and r["ngpu"] == 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logdir", nargs="?", default="results/ddp124/logs")
    ap.add_argument("--runs", help="read the compact export instead of logs")
    ap.add_argument("--export", help="write the compact export and exit")
    ap.add_argument("--out", default="results/figures/loss_vs_time_ddp_124m.png")
    ap.add_argument("--summary", default="results/ddp124/summary.md")
    ap.add_argument("--bench", default="results/benchmarks/ddp_bench_124m.json")
    ap.add_argument("--time", choices=["bench", "median", "actual"], default="bench")
    ap.add_argument("--all-lrs", action="store_true", help="plot every Dion/MuonBP LR, not just the best")
    args = ap.parse_args()

    if args.runs:
        recs = [json.loads(l) for l in open(args.runs) if l.strip()]
    else:
        recs = records_from_logs(args.logdir)
    if not recs:
        sys.exit("no runs with evals")
    if args.export:
        os.makedirs(os.path.dirname(args.export) or ".", exist_ok=True)
        with open(args.export, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"{args.export}: {len(recs)} runs")
        return

    bench = bench_step_time(args.bench) if args.time == "bench" else {}
    per_method = defaultdict(dict)
    for r in recs:
        m = r["method"]
        if args.time == "bench":
            sps = bench[BENCH_ROW[base_of(m)]]
            hours = [i * sps / 3600 for i in r["iters"]]
        elif args.time == "median":
            hours = [i * r["med"] / 3600 for i in r["iters"]]
        else:
            hours = r["actual_hours"]
        per_method[m][r["seed"]] = dict(iters=r["iters"], loss=r["loss"], hours=hours, med=r["med"],
                                        last_iter=r["last_iter"], total_h=r["actual_hours"][-1])

    # one LR per baseline in the figure: the best final loss
    shown = list(per_method)
    if not args.all_lrs:
        for base in ("dion", "muonbp", "muon"):
            cands = [m for m in per_method if base_of(m) == base]
            if len(cands) > 1:
                best = min(cands, key=lambda m: min(min(s["loss"]) for s in per_method[m].values()))
                shown = [m for m in shown if base_of(m) != base or m == best]
    shown.sort(key=lambda m: ORDER.index(base_of(m)) if base_of(m) in ORDER else 99)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    handles = []
    floor = min(min(s["loss"]) for m in shown for s in per_method[m].values())
    for m in shown:
        s = style_of(m)
        seeds = per_method[m]
        common = sorted(set.intersection(*(set(v["iters"]) for v in seeds.values())))
        loss = {i: [v["loss"][v["iters"].index(i)] for v in seeds.values()] for i in common}
        hrs = {i: st.mean(v["hours"][v["iters"].index(i)] for v in seeds.values()) for i in common}
        y = [st.mean(loss[i]) for i in common]
        for ax, x in zip(axes, ([i * TOKENS_PER_ITER / 1e9 for i in common], [hrs[i] for i in common])):
            ax.plot(x, y, color=s["color"], ls=s["ls"], lw=s["lw"])
            if len(seeds) > 1:
                ax.fill_between(x, [min(loss[i]) for i in common], [max(loss[i]) for i in common],
                                color=s["color"], alpha=0.15, lw=0)
        handles.append(Line2D([0], [0], color=s["color"], ls=s["ls"], lw=s["lw"],
                              label=label_of(m) + (f" ({len(seeds)} seeds)" if len(seeds) > 1 else "")))
    axes[0].set_xlabel("Tokens (B)", fontsize=11)
    axes[1].set_xlabel({"bench": "Wall-clock (hours), 4×A100 PCIe step time × iterations",
                        "median": "Wall-clock (hours), median step time on 4×H200",
                        "actual": "Wall-clock (hours), measured on shared 4×H200"}[args.time], fontsize=11)
    axes[0].set_ylabel("Val Loss", fontsize=11)
    for ax in axes:
        ax.set_ylim(floor - 0.02, floor + 0.5)
        ax.grid(alpha=0.3)
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6), fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"{args.out} written")

    # summary table
    ref = None
    if "muon" in per_method:
        ref = st.mean(min(v["loss"]) for v in per_method["muon"].values())
    rows = ["| method | seeds | last iter | best val loss | s/step (node median) | s/step (bench) | "
            "hours to finish | iters to Muon's final loss |",
            "|---|---|---|---|---|---|---|---|"]
    for m in sorted(per_method, key=lambda m: ORDER.index(base_of(m)) if base_of(m) in ORDER else 99):
        seeds = per_method[m]
        best = [min(v["loss"]) for v in seeds.values()]
        bl = f"{st.mean(best):.3f}" + (f" ± {st.stdev(best):.3f}" if len(best) > 1 else "")
        med = st.mean(v["med"] for v in seeds.values())
        bsps = bench.get(BENCH_ROW.get(base_of(m), ""), float("nan"))
        hrs = st.mean(v["hours"][-1] for v in seeds.values())
        last = min(v["last_iter"] for v in seeds.values())
        reach = "—"
        if ref is not None:
            hit = []
            for v in seeds.values():
                ok = [i for i, l in zip(v["iters"], v["loss"]) if l <= ref]
                hit.append(ok[0] if ok else None)
            reach = f"{st.mean(hit):.0f}" if all(h is not None for h in hit) else "not reached"
        rows.append(f"| {label_of(m)} | {len(seeds)} | {last} | {bl} | {med:.3f} | {bsps:.3f} | {hrs:.1f} | {reach} |")
    os.makedirs(os.path.dirname(args.summary) or ".", exist_ok=True)
    with open(args.summary, "w") as f:
        f.write("# 124M, 1× Chinchilla, 4-GPU DDP (loss from opt2 H200 runs; step time from the "
                f"controlled benchmark, --time {args.time})\n\n" + "\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
