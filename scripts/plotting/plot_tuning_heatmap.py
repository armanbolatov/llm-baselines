"""Learning-rate / ratio grid at 124M, from the full-budget (64k-step) runs the tables quote.

Reads results/tuning/grid_124m.jsonl, which is extracted from the run directories themselves
(exps/fw124g_*_i64k_seed0 and exps_124m/wt124g_*_i64k_seed0). Only cells that ran the full 64k
steps carry a number; cells that scripts/prune.sh stopped early — they were already far behind
their group at 6k or 16k steps — are drawn grey and marked "cut", so the sweep's coverage is
visible and a hole never looks like a cell nobody tried.

Axes are the two knobs a user actually sets: the spectral-step learning rate eta_M and the ratio
alpha = eta_M / eta_sign. The cell used in the paper (the best of each panel) is boxed, which is
what shows the optimum is interior rather than at an edge.

    python scripts/plotting/plot_tuning_heatmap.py            # FineWeb -> tuning_heatmap_124m.png
    TAG=wt python scripts/plotting/plot_tuning_heatmap.py     # WikiText-103
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = os.environ.get("GRID_JSONL", "results/tuning/grid_124m.jsonl")
TAG = os.environ.get("TAG", "fw")
DATASET = {"fw": "FineWeb", "wt": "WikiText-103"}[TAG]
FAMILY = {"09-09": "SignMuon", "09-099": "LionMuon"}      # (beta1, beta2) -> family
PERIODS = [1, 2, 5, 20]
MIN_EVALS = 100        # a full 64k-step run logs 128 evaluations; pruned cells log ~12


def load():
    """(family, P) -> ({(eta_M, alpha): loss} for full runs, {(eta_M, alpha)} stopped early)"""
    cells, pruned = {}, {}
    for line in open(SRC):
        r = json.loads(line)
        if r["tag"] != TAG or r["betas"] not in FAMILY or r["p"] not in PERIODS:
            continue
        key = (FAMILY[r["betas"]], r["p"])
        try:                     # a P=1 run takes no sign step, so its name carries no ratio
            alpha = float(r["alpha"])
        except ValueError:
            alpha = 1.0
        if r["evals"] >= MIN_EVALS:
            cells.setdefault(key, {})[(float(r["lr"]), alpha)] = r["loss"]
        else:
            pruned.setdefault(key, set()).add((float(r["lr"]), alpha))
    return cells, pruned


def panel(ax, grid, cut, title, vmin, vmax):
    lrs = sorted({lr for lr, _ in grid} | {lr for lr, _ in cut})
    als = sorted({a for _, a in grid} | {a for _, a in cut})
    z = np.full((len(als), len(lrs)), np.nan)
    for (lr, a), v in grid.items():
        z[als.index(a), lrs.index(lr)] = v
    im = ax.imshow(z, cmap="RdYlGn_r", vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
    best = min(grid.values())
    for i in range(len(als)):
        for j in range(len(lrs)):
            if np.isnan(z[i, j]):
                continue
            is_best = abs(z[i, j] - best) < 1e-9
            ax.text(j, i, f"{z[i, j]:.3f}", ha="center", va="center", fontsize=7,
                    fontweight="bold" if is_best else "normal")
            if is_best:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="black", linewidth=2))
    for (lr, a) in cut:                     # swept, but stopped early by scripts/prune.sh
        j, i = lrs.index(lr), als.index(a)
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="0.85", edgecolor="white"))
        ax.text(j, i, "cut", ha="center", va="center", fontsize=6, color="0.35", style="italic")
    ax.set_xticks(range(len(lrs)))
    ax.set_xticklabels([f"{lr:g}" for lr in lrs], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(als)))
    ax.set_yticklabels(["n/a" if len(als) == 1 and a == 1.0 else f"{a:g}" for a in als], fontsize=7)
    ax.set_title(title, fontsize=9)
    return im


def main():
    cells, pruned = load()
    vals = [v for g in cells.values() for v in g.values()]
    vmin, vmax = min(vals), np.percentile(vals, 85)      # the worst cells would flatten the rest
    fig, axes = plt.subplots(2, len(PERIODS), figsize=(3.3 * len(PERIODS), 7),
                             layout="constrained")
    im = None
    for row, fam in enumerate(["SignMuon", "LionMuon"]):
        for col, p in enumerate(PERIODS):
            ax = axes[row][col]
            grid = cells.get((fam, p))
            name = {("SignMuon", 1): "Muon (P=1)", ("LionMuon", 1): "LionMuon P=1"}.get(
                (fam, p), f"{fam} P={p}")
            if not grid:
                ax.set_visible(False)
                continue
            im = panel(ax, grid, pruned.get((fam, p), set()), name, vmin, vmax)
            if col == 0:
                ax.set_ylabel(r"$\alpha=\eta_M/\eta_{\rm sign}$", fontsize=9)
            if row == 1:
                ax.set_xlabel(r"$\eta_M$", fontsize=9)
    fig.suptitle(f"124M, {DATASET}, 64k steps: best validation loss over the tuning grid "
                 f"(grey = stopped early by the pruning rule)", fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.75, label="best val loss")
    out = os.environ.get("OUT_PNG", f"results/figures/tuning_heatmap_124m{'' if TAG == 'fw' else '_wt'}.png")
    os.makedirs("results/figures", exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"{out} written")
    for (fam, p), g in sorted(cells.items()):
        b = min(g, key=g.get)
        print(f"  {fam:<9} P={p:<3} {len(g):>2} cells   best {g[b]:.4f} at eta_M={b[0]:g} alpha={b[1]:g}")


if __name__ == "__main__":
    main()
