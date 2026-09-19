"""355M / 720M figure: best val loss vs total training FLOPs, same style as plot_124m.py.

Input: JSON lines from scripts/plotting/collect_scaling.sh (one file per machine).
Runs still in progress are drawn with a hollow marker.

    python scripts/plotting/plot_scaling.py curves_opt.json curves_opt2.json
"""
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

ARCH = {
    "355m": dict(n_layer=24, n_embd=1024, seq_len=1024, batch_size=512, iterations=15650),
    "720m": dict(n_layer=12, n_embd=2048, seq_len=512, batch_size=1984, iterations=14000),
}
TITLE = {"355m": "355M (8.2B tokens, 23 TPP)", "720m": "720M (14.2B tokens, 20 TPP)"}
INF = 10_000_000
ORDER = ["AdamW", "Lion (P=∞)", "Signum (P=∞)", "Muon (P=1)",
         "LionMuon P=1", "LionMuon P=2", "LionMuon P=5", "LionMuon P=20",
         "SignMuon P=2", "SignMuon P=5", "SignMuon P=20"]


def label_and_flops(name, scale):
    """Map an experiment dir name to (legend label, FLOPs)."""
    a = ARCH[scale]
    if "_adamw" in name:
        return "AdamW", compute_flops(opt="adamw", **a)[0]
    m = re.search(r"_(lionmuon|signmuon)_p(\d+)", name)          # seeds mode
    if not m:
        m = re.search(r"g_p(\d+)_.*_b(09-09|09-099)", name)       # grid mode
        if not m:
            return None, None
        p, fam = int(m.group(1)), ("signmuon" if m.group(2) == "09-09" else "lionmuon")
    else:
        fam, p = m.group(1), int(m.group(2))
    if fam == "signmuon":
        label = "Muon (P=1)" if p == 1 else "Signum (P=∞)" if p == INF else f"SignMuon P={p}"
    else:
        label = "Lion (P=∞)" if p == INF else f"LionMuon P={p}"
    return label, compute_flops(opt="lion_muon", K=p, ns_steps=5, **a)[0]


def main(paths):
    rows = []
    for pth in paths:
        for line in open(pth):
            if line.strip():
                rows.append(json.loads(line))
    # The same run exists on both machines (the Muon cross-check); keep one copy, preferring the
    # finished one, so it does not turn into a fake second seed.
    by_name = {}
    for r in rows:
        prev = by_name.get(r["name"])
        if prev is None or (r["done"] and not prev["done"]):
            by_name[r["name"]] = r
    rows = list(by_name.values())

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    handles = {}
    for ax, scale in zip(axes, ("355m", "720m")):
        cells = {}  # label -> (flops, {done: [values], running: [values]})
        for r in rows:
            if r["scale"] != scale:
                continue
            label, flops = label_and_flops(r["name"], scale)
            if label is None:
                continue
            c = cells.setdefault(label, [flops, {True: [], False: []}])
            c[1][r["done"]].append(r["best"])
        # a seed still running has a much higher "best so far" and would drag the mean down;
        # show it only when nothing is finished for that method yet
        cells = {lab: (flops, v[True] or v[False], bool(v[True]))
                 for lab, (flops, v) in cells.items()}
        for label, (flops, vals, done) in cells.items():
            s = STYLE[label]
            ax.scatter(flops, st.mean(vals), s=110, zorder=5, marker=s["marker"],
                       color=s["color"] if done else "white", edgecolor=s["color"] if not done else "black",
                       linewidth=1.5 if not done else 0.5, alpha=0.9)
            if len(vals) > 1:
                ax.errorbar(flops, st.mean(vals), yerr=st.stdev(vals), zorder=4,
                            color=s["color"], capsize=3, linewidth=1)
            handles.setdefault(label, Line2D([0], [0], marker=s["marker"], linestyle="None",
                                             markerfacecolor=s["color"], markeredgecolor="black",
                                             markeredgewidth=0.5, markersize=8, label=label))
        ax.set_xlabel("Total Training FLOPs", fontsize=11)
        ax.set_title(TITLE[scale], fontsize=12)
        ax.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))
        ax.margins(x=0.18, y=0.18)     # a panel with two points would put them on the frame
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Best Val Loss", fontsize=11)
    ordered = [handles[k] for k in ORDER if k in handles]
    fig.legend(handles=ordered, loc="lower center", ncol=6, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    os.makedirs("results/figures", exist_ok=True)
    out = "results/figures/scaling_loss_vs_flops.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"{out} written")
    for r in sorted(rows, key=lambda r: (r["scale"], r["best"])):
        print(f"  {r['scale']} {r['name']:<40} {r['machine']:<5} {r['best']:.4f} {'' if r['done'] else '(running)'}")


if __name__ == "__main__":
    main(sys.argv[1:])
