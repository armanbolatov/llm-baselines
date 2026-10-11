"""Two-panel cost figure: 355M best loss vs FLOPs, and 124M under 4-GPU DDP vs wall-clock.

Left panel: the 355M runs with hyperparameters transferred from the tuned 124M grid,
collected by scripts/plotting/collect_scaling.sh (one JSON-lines file per machine).
Right panel: the 124M runs at 1x Chinchilla under real 4-GPU DDP. Loss comes from those
runs; the x axis is the measured cost of a step, from scripts/cost_bench.py: MB sent per step
(DDP_X=comm, the default), the same over the whole run in TB (DDP_X=comm_total), or counted
FLOPs (DDP_X=flops). All of these are counted, not estimated, and none depends on the hardware. DDP_X=time uses the step time of results/benchmarks/ddp_bench_124m.json
instead, which is an estimate: the training node is shared, so its own step times mostly measure
the neighbours. Muon, Dion and MuonBP are drawn at their best LR.

    python scripts/plotting/plot_loss_vs_cost.py scaling_opt.json scaling_opt2.json

Marker and colour come from plot_results.STYLE, so this matches the other figures.
"""
import json
import os
import re
import statistics as st
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_results import (ERR_KW, LEGEND_KW, POINT_KW, compute_flops, finish_axis,
                          legend_handle, style_for)  # noqa: E402

ARCH_355M = dict(n_layer=24, n_embd=1024, seq_len=1024, batch_size=512, iterations=15650)
INF = 10_000_000
ORDER = ["AdamW", "Lion (P=∞)", "Signum (P=∞)", "Muon (P=1)",
         "LionMuon P=1", "LionMuon P=2", "LionMuon P=5", "LionMuon P=20",
         "SignMuon P=2", "SignMuon P=5", "SignMuon P=20"]
# method name in the DDP runs -> (legend label, row of the benchmark that times its step).
# SignMuon P=2 does the same work per step as LionMuon P=2 (one buffer, same NS schedule,
# same all-reduce), so it shares that row.
DDP_METHODS = {
    "adamw":       ("AdamW", "adamw"),
    "muon":        ("Muon (P=1)", "muon"),
    "signmuon_p2": ("SignMuon P=2", "lionmuon_p2"),
    "lionmuon_p2": ("LionMuon P=2", "lionmuon_p2"),
    "lionmuon_p5": ("LionMuon P=5", "lionmuon_p5"),
    "dion":        ("Dion", "dion"),
    "muonbp":      ("MuonBP (P=5)", "muonbp_p5"),
    "lionmuon_p1": ("LionMuon P=1", "muon"),
    "signmuon_p5": ("SignMuon P=5", "lionmuon_p5"),
    "signum":      ("Signum (P=∞)", "adamw"),
    "lion":        ("Lion (P=∞)", "adamw"),
    "dion_r2":     ("Dion, rank 1/2", "dion_r2"),
}
def label_355m(name):
    """Map a 355M experiment dir name to its legend label."""
    if "_adamw" in name:
        return "AdamW", compute_flops(opt="adamw", **ARCH_355M)[0]
    m = re.search(r"_(lionmuon|signmuon)_p(\d+)", name)
    if not m:
        return None, None
    fam, p = m.group(1), int(m.group(2))
    if fam == "signmuon":
        label = "Muon (P=1)" if p == 1 else "Signum (P=∞)" if p == INF else f"SignMuon P={p}"
    else:
        label = "Lion (P=∞)" if p == INF else f"LionMuon P={p}"
    return label, compute_flops(opt="lion_muon", K=p, ns_steps=5, **ARCH_355M)[0]


def cells_355m(paths):
    """label -> (flops, [losses], all_finished)"""
    rows = []
    for pth in paths:
        rows += [json.loads(l) for l in open(pth) if l.strip()]
    # the Muon cross-check ran on both machines: keep one copy so it is not a second seed
    by_name = {}
    for r in rows:
        prev = by_name.get(r["name"])
        if prev is None or (r["done"] and not prev["done"]):
            by_name[r["name"]] = r
    cells = {}
    for r in by_name.values():
        if r["scale"] != "355m":
            continue
        if os.environ.get("SEED0_ONLY") and not r["name"].endswith("_seed0"):
            continue                     # one seed per method, so every point is drawn the same way
        label, flops = label_355m(r["name"])
        if label is None:
            continue
        cells.setdefault(label, [flops, {True: [], False: []}])[1][r["done"]].append(r["best"])
    # a seed still running has a much higher "best so far"; show it only if nothing finished
    return {lab: (flops, v[True] or v[False], bool(v[True])) for lab, (flops, v) in cells.items()}


def ddp_family(method):
    """'dion_3e-4' -> ('dion', '3e-4'); 'muon_3e-3' -> ('muon', '3e-3'); plain 'muon' ran at 1e-3."""
    for fam in ("dion_r2", "dion", "muonbp", "muon", "lion"):   # dion_r2 before dion: prefix
        if method == fam:
            return fam, "1e-3"
        if method.startswith(fam + "_"):
            return fam, method[len(fam) + 1:]
    return method, None


def per_step_cost(mode, ebs=32):
    """(row of the cost table -> cost, axis label, whether to multiply by the iteration count).
    'comm' and 'flops' are counted on the running code (scripts/cost_bench.py) and do not depend
    on the hardware; 'time' is a step time measured on one machine under one load, so it is an
    estimate and is labelled as one. Every run here trains for the same number of iterations, so
    a per-step axis and a whole-run axis differ only in their units."""
    if mode in ("comm", "comm_total"):
        d = json.load(open(os.environ.get("COMM_JSON", "results/benchmarks/cost/comm_124m_4gpu.json")))
        mb = {k: v["total_MB_per_step"] for k, v in d["methods"].items()}
        if mode == "comm":
            return mb, "MB sent per step (4 GPUs), measured", False
        return {k: v / 1e6 for k, v in mb.items()}, "Communication (TB), measured", True
    if mode == "flops":
        d = json.load(open(os.environ.get("FLOPS_JSON", "results/benchmarks/cost/flops_124m.json")))
        return ({k: v["total_per_step"] / 1e18 for k, v in d["methods"].items()},
                "FLOPs ($\\times 10^{18}$), counted", True)
    rows = json.load(open(os.environ.get("BENCH_JSON", "results/benchmarks/ddp_bench_124m.json")))
    return ({r["opt"]: r["s_per_step"] / 3600 for r in rows if r["ebs"] == ebs and r["ngpu"] == 4},
            "Estimated time (h) on 4×A100 PCIe", True)


def runs_ddp(ddp_file, cost, per_run=True):
    """(family, lr) -> [x, [losses of finished seeds], [(iters, loss) of seeds still training]]"""
    out = {}
    for r in json.load(open(ddp_file))["runs"]:
        fam, lr = ddp_family(r["method"])
        cell = out.setdefault((fam, lr), [None, [], []])
        if r["done"]:
            c = cost[DDP_METHODS[fam][1]]
            cell[0] = r["iters"] * c if per_run else c
            cell[1].append(r["loss"])
        else:
            cell[2].append((r["iters"], r["loss"]))
    return out


def best_lrs(runs):
    """family -> the LR with the lowest mean loss over finished seeds"""
    best = {}
    for (fam, lr), (_, losses, _) in runs.items():
        if losses and (fam not in best or st.mean(losses) < st.mean(runs[(fam, best[fam])][1])):
            best[fam] = lr
    return best


def cells_ddp(runs):
    """label -> (hours, [losses], finished). Baselines with an LR sweep (Muon, Dion, MuonBP)
    are drawn at their best LR. A run short of the full token budget sits far above the others
    and would squash the interesting range, so only finished seeds count; main() prints the rest."""
    return {DDP_METHODS[fam][0]: (runs[(fam, lr)][0], runs[(fam, lr)][1], True)
            for fam, lr in best_lrs(runs).items()}


def draw(ax, cells, xlabel, title):
    handles = {}
    for label, (x, vals, done) in cells.items():
        s = style_for(label)
        kw = dict(POINT_KW)
        if not done:                        # hollow marker for a run that has not finished
            kw.update(edgecolor=s["color"], linewidth=1.5)
        ax.scatter(x, st.mean(vals), marker=s["marker"],
                   color=s["color"] if done else "white", **kw)
        if len(vals) > 1:
            # black and above the marker: the seed spread is often smaller than the marker itself
            ax.errorbar(x, st.mean(vals), yerr=st.stdev(vals), color="black", **{**ERR_KW, "zorder": 6})
        handles[label] = legend_handle(label)
    finish_axis(ax, xlabel, title)
    return handles


def main(paths):
    ddp_file = os.environ.get("DDP_JSON", "results/ddp124/final_losses.json")
    mode = os.environ.get("DDP_X", "comm")          # comm (default), flops, or time
    cost, xlabel, per_run = per_step_cost(mode)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    handles = draw(axes[0], cells_ddp(runs_ddp(ddp_file, cost, per_run)), xlabel,
                   "124M, 2.46B tokens, 4-GPU data parallel")
    handles.update(draw(axes[1], cells_355m(paths), "Total Training FLOPs",
                        "355M, 8.2B tokens (transferred hyperparameters)"))
    axes[1].ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))
    axes[0].set_ylabel("Best Val Loss", fontsize=11)
    ordered = [handles[k] for k in ORDER if k in handles] + \
              [h for k, h in handles.items() if k not in ORDER]
    fig.legend(handles=ordered, ncol=min(len(ordered), 6), **LEGEND_KW)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    os.makedirs("results/figures", exist_ok=True)
    out = os.environ.get("OUT_PNG", "results/figures/loss_vs_cost_355m_ddp.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"{out} written  (right panel x = {mode})")
    # the same two panels as separate figures, sized for a wrapfigure: legend below in three columns
    for name, cells, xl, title in [
            ("loss_vs_cost_ddp124", cells_ddp(runs_ddp(ddp_file, cost, per_run)), xlabel,
             "124M, 2.46B tokens, 4-GPU data parallel"),
            ("loss_vs_flops_355m", cells_355m(paths), "Total Training FLOPs",
             "355M, 8.2B tokens, transferred hyperparameters")]:
        f1, ax = plt.subplots(figsize=(5.4, 3.6))
        hs = draw(ax, cells, xl, title)
        if name.endswith("355m"):
            ax.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))
        ax.set_ylabel("Best Val Loss", fontsize=11)
        ordered = [hs[k] for k in ORDER if k in hs] + [h for k, h in hs.items() if k not in ORDER]
        f1.legend(handles=ordered, ncol=3, loc="lower center", fontsize=8, frameon=False,
                  bbox_to_anchor=(0.5, -0.02))
        f1.tight_layout(rect=[0, 0.10 + 0.05 * ((len(ordered) - 1) // 3), 1, 1])
        f1.savefig(f"results/figures/{name}.png", dpi=150, bbox_inches="tight")
        print(f"results/figures/{name}.png written")
    runs = runs_ddp(ddp_file, cost, per_run)
    best = best_lrs(runs)
    for (fam, lr), (x, vals, running) in sorted(runs.items()):
        label = DDP_METHODS[fam][0]
        if vals:
            note = "" if best.get(fam) == lr else "   not the best LR, not plotted"
            print(f"  ddp  {label:<14} {lr or '':<5} {st.mean(vals):.4f} (n={len(vals)})  x={x:6.1f}{note}")
        for it, loss in running:
            print(f"  ddp  {label:<14} {lr or '':<5} {loss:.4f} at {it} iters   still training, not plotted")


if __name__ == "__main__":
    main(sys.argv[1:])
