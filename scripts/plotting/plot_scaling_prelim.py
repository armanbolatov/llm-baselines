"""Preliminary 355M / 720M figure from the ICLR rerun.

Reads the JSON-lines curves produced by extract_curves.sh (one file per
machine) and draws validation loss vs iteration at each scale. Runs still in
progress are drawn with a dashed line and an open marker at their last point.

    python scripts/plotting/plot_scaling_prelim.py curves_opt.json curves_opt2.json
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Same palette as the paper's figures: Muon dark red, LionMuon blue, SignMuon
# mid red, AdamW grey.
STYLE = {
    "muon": ("Muon ($P{=}1$)", "#8b0000", "^"),
    "signmuon_p2": ("SignMuon $P{=}2$", "#cc1111", "^"),
    "signmuon_p5": ("SignMuon $P{=}5$", "#e04040", "^"),
    "lionmuon_p2": ("LionMuon $P{=}2$", "#1144cc", "o"),
    "lionmuon_p5": ("LionMuon $P{=}5$", "#3377ee", "o"),
    "adamw": ("AdamW", "#888888", "s"),
}
TOTAL = {"355m": 15650, "720m": 14000}
TOKENS = {"355m": "8.2B tokens, 23 TPP", "720m": "14.2B tokens, 20 TPP"}


def key_of(run):
    base = run.rsplit("_s", 1)[0]
    return base, run.rsplit("_s", 1)[1] if "_s" in run else "0"


def load(paths):
    rows = []
    for p in paths:
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def main(paths):
    rows = load(paths)
    scales = [s for s in ("355m", "720m") if any(r["scale"] == s for r in rows)]
    fig, axes = plt.subplots(1, len(scales), figsize=(6.2 * len(scales), 4.6))
    axes = [axes] if len(scales) == 1 else list(axes)

    for ax, scale in zip(axes, scales):
        sub = [r for r in rows if r["scale"] == scale]
        finals = []
        for r in sorted(sub, key=lambda r: r["run"]):
            base, seed = key_of(r["run"])
            if base not in STYLE:
                continue
            label, color, marker = STYLE[base]
            it, ls_ = r["iters"], r["loss"]
            if len(it) < 2:
                continue
            done = it[-1] >= TOTAL[scale]
            # seed 0 solid; other seeds thin
            lw = 2.0 if seed == "0" else 1.0
            ax.plot(it[1:], ls_[1:], color=color, lw=lw,
                    ls="-" if done else "--", zorder=3)
            ax.scatter([it[-1]], [ls_[-1]], s=70, color=color, marker=marker,
                       edgecolor="black", linewidth=0.6, zorder=5,
                       facecolor=color if done else "white")
            tag = f"{label} ({r['machine']})" if base == "muon" else label
            finals.append((ls_[-1], it[-1], done, tag, color))

        ax.set_xlabel("Iteration", fontsize=14)
        ax.set_ylabel("Validation loss", fontsize=14)
        ax.set_title(f"{scale.upper()[:-1]}M on FineWeb  ({TOKENS[scale]})", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_xlim(0, TOTAL[scale] * 1.02)
        if finals:
            lo = min(f[0] for f in finals)
            ax.set_ylim(lo - 0.05, lo + 0.55)
        # legend ordered by final loss, with the number in the label
        handles = []
        for loss, it, done, tag, color in sorted(finals):
            mark = "" if done else f" @{it}"
            handles.append(plt.Line2D([], [], color=color, lw=2,
                                      ls="-" if done else "--",
                                      label=f"{tag}: {loss:.3f}{mark}"))
        ax.legend(handles=handles, fontsize=10, loc="upper right", framealpha=0.95)

    fig.suptitle("Preliminary: 124M-tuned $(\\eta_M,\\alpha,\\beta_1,\\beta_2)$ transferred to scale; "
                 "dashed = still running", fontsize=11, y=0.99)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs("results/figures", exist_ok=True)
    out = "results/figures/scaling_prelim.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")
    for scale in scales:
        print(f"\n{scale}:")
        sub = [r for r in rows if r["scale"] == scale]
        for r in sorted(sub, key=lambda r: r["loss"][-1]):
            base, _ = key_of(r["run"])
            st = "done" if r["iters"][-1] >= TOTAL[scale] else f"at {r['iters'][-1]}"
            print(f"  {r['run']:<22} {r['machine']:<5} {r['loss'][-1]:.4f}  ({st})")


if __name__ == "__main__":
    main(sys.argv[1:] or ["curves_opt.json", "curves_opt2.json"])
