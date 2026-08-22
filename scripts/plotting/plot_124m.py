"""124M figure: best val loss vs total training FLOPs.

Style and FLOPs accounting come from plot_results.py, so this matches the
paper's Pareto figure. Error bars are the spread over seeds.

    python scripts/plotting/plot_124m.py
"""
import glob
import json
import os
import re
import statistics as st
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_results import STYLE, compute_flops  # noqa: E402

ARCH = dict(n_layer=12, n_embd=768, seq_len=512, batch_size=32, iterations=64000)
INF = 10_000_000                      # muon_every_k used for P = infinity


def label_of(p, fam):
    if fam == "sign":
        return "Muon (P=1)" if p == 1 else "Signum (P=∞)" if p == INF else f"SignMuon P={p}"
    return "Lion (P=∞)" if p == INF else f"LionMuon P={p}"


def best(path):
    v = min(json.load(open(path))["val_loss"])
    return None if v != v else v


def find(name):
    for src in ("exps", "exps_124m"):
        if os.path.exists(f"{src}/{name}/summary.json"):
            return best(f"{src}/{name}/summary.json")
    return None


def runs(tag):
    """(P, family) -> seed losses of the tuned winner."""
    cells = {}
    for src in ("exps", "exps_124m"):
        for f in glob.glob(f"{src}/{tag}124g_*_i64k_seed0/summary.json"):
            m = re.match(rf"{tag}124g_p(\d+)_.*_b(09-09|09-099)_i64k_seed0$", f.split("/")[1])
            v = best(f) if m else None
            if v is None:
                continue
            key = (int(m.group(1)), "sign" if m.group(2) == "09-09" else "lion")
            cells[key] = min(v, cells.get(key, 9e9))
    out = {}
    for (p, fam), v0 in cells.items():
        out[(p, fam)] = [v0] + [v for s in (1, 2)
                                if (v := find(f"{tag}124t_{fam}muon_p{p}_seed{s}")) is not None]
    for f in glob.glob(f"exps*/{tag}124b_adamw_lr*_seed0/summary.json"):
        if (v := best(f)) is not None:
            out[("adamw",)] = [min(v, out.get(("adamw",), [9e9])[0])]
    return out


fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
handles = {}
for ax, (tag, title) in zip(axes, [("fw", "FineWeb"), ("wt", "WikiText-103")]):
    for key, vals in sorted(runs(tag).items(), key=lambda kv: str(kv[0])):
        if key == ("adamw",):
            label, flops = "AdamW", compute_flops(opt="adamw", **ARCH)[0]
        else:
            p, fam = key
            label = label_of(p, fam)
            flops = compute_flops(opt="lion_muon", K=p, ns_steps=5, **ARCH)[0]
        s = STYLE[label]
        ax.scatter(flops, st.mean(vals), s=110, zorder=5, color=s["color"],
                   marker=s["marker"], alpha=0.9, edgecolor="black", linewidth=0.5)
        if len(vals) > 1:
            ax.errorbar(flops, st.mean(vals), yerr=st.stdev(vals), zorder=4,
                        color=s["color"], capsize=3, linewidth=1)
        handles.setdefault(label, Line2D([0], [0], marker=s["marker"], linestyle="None",
                                         markerfacecolor=s["color"], markeredgecolor="black",
                                         markeredgewidth=0.5, markersize=8, label=label))
    ax.set_xlabel("Total Training FLOPs", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Best Val Loss", fontsize=11)
fig.legend(handles=list(handles.values()), loc="center right", fontsize=9, title="Optimizer")
fig.tight_layout(rect=[0, 0, 0.82, 1])
out = "results/figures/124m_loss_vs_flops.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"{out} written")
