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
from plot_results import (ERR_KW, LEGEND_KW, POINT_KW, compute_flops, finish_axis,
                          legend_handle, style_for)  # noqa: E402

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
            m = re.match(rf"{tag}124g_p(\d+)_.*_b(09-09|09-099)_i64k_seed0$",
                         os.path.basename(os.path.dirname(f)))
            v = best(f) if m else None
            if v is None:
                continue
            key = (int(m.group(1)), "sign" if m.group(2) == "09-09" else "lion")
            cells[key] = min(v, cells.get(key, 9e9))
    out = {}
    for (p, fam), v0 in cells.items():
        out[(p, fam)] = [v0] + [v for s in (1, 2)
                                if (v := find(f"{tag}124t_{fam}muon_p{p}_seed{s}")) is not None]
        # the pure-sign seeds were run through the grid, so they carry the grid name
        if p == INF and len(out[(p, fam)]) == 1:
            for src in ("exps", "exps_124m"):
                for s in (1, 2):
                    for f in glob.glob(f"{src}/{tag}124g_p{p}_*_b{'09-09' if fam == 'sign' else '09-099'}_i64k_seed{s}/summary.json"):
                        if (v := best(f)) is not None:
                            out[(p, fam)].append(v)
    adamw = {}                                   # lr -> [seed losses], best lr by seed 0
    for src in ("exps", "exps_124m"):
        for f in glob.glob(f"{src}/{tag}124b_adamw_lr*_seed*/summary.json"):
            m = re.search(r"_lr([^_]+)_seed(\d)", f)
            if m and (v := best(f)) is not None:
                adamw.setdefault(m.group(1), {})[int(m.group(2))] = v
    if adamw:
        lr = min(adamw, key=lambda k: adamw[k].get(0, 9e9))
        out[("adamw",)] = [adamw[lr][s] for s in sorted(adamw[lr])]
    return out


fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
handles = {}
for ax, (tag, title) in zip(axes, [("fw", "FineWeb"), ("wt", "WikiText-103")]):
    for key, vals in sorted(runs(tag).items(), key=lambda kv: str(kv[0])):
        if key == ("adamw",):
            label, flops = "AdamW", compute_flops(opt="adamw", **ARCH)[0]
        else:
            p, fam = key
            label = label_of(p, fam)
            flops = compute_flops(opt="lion_muon", K=p, ns_steps=5, **ARCH)[0]
        s = style_for(label)
        ax.scatter(flops, st.mean(vals), color=s["color"], marker=s["marker"], **POINT_KW)
        if len(vals) > 1:
            ax.errorbar(flops, st.mean(vals), yerr=st.stdev(vals), **ERR_KW)
        handles.setdefault(label, legend_handle(label))
    finish_axis(ax, "Total Training FLOPs", title)
    ax.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))
axes[0].set_ylabel("Best Val Loss", fontsize=11)
# legend below the panels, in a fixed order: baselines first, then by P within each family
order = ["AdamW", "Lion (P=∞)", "Signum (P=∞)", "Muon (P=1)",
         "LionMuon P=1", "LionMuon P=2", "LionMuon P=5", "LionMuon P=20",
         "SignMuon P=2", "SignMuon P=5", "SignMuon P=20"]
ordered = [handles[k] for k in order if k in handles] + [h for k, h in handles.items() if k not in order]
fig.legend(handles=ordered, ncol=min(len(ordered), 6), **LEGEND_KW)
fig.tight_layout(rect=[0, 0.08, 1, 1])
out = "results/figures/loss_vs_flops_124m.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"{out} written")
