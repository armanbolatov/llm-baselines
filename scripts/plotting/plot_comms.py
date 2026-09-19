"""Communication figure, in bytes only — no wall-clock anywhere.

Left: what one step sends, measured by counting every collective in a real 4-GPU run
(scripts/cost_bench.py --mode comm, results/benchmarks/cost/comm_124m_4gpu.json). Each bar is
split into the gradient sync, which DDP overlaps with the backward pass, and the optimizer's own
all-reduce, which sits on the critical path. Byte counts are identical on PCIe and on NVLink.

Right: what it costs to reach Muon's own final loss (3.418) — steps from
results/ddp124/steps_to_target.json times the per-step bytes on the left. Dion never gets there.

    python scripts/plotting/plot_comms.py
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_results import STYLE  # noqa: E402

COMM = os.environ.get("COMM_JSON", "results/benchmarks/cost/comm_124m_4gpu.json")
STEPS = os.environ.get("STEPS_JSON", "results/ddp124/steps_to_target.json")
EXTRA = {"Dion": "#2e8b57", "MuonBP (P=5)": "#e69500"}
# legend label -> (row of the measured byte table, key in the steps-to-target table)
ROWS = [("AdamW", "adamw", "adamw"),
        ("Muon (P=1)", "muon", "muon"),
        ("SignMuon P=2", "lionmuon_p2", "signmuon_p2"),
        ("LionMuon P=2", "lionmuon_p2", "lionmuon_p2"),
        ("LionMuon P=5", "lionmuon_p5", "lionmuon_p5"),
        ("MuonBP (P=5)", "muonbp_p5", "muonbp_1e-3"),
        ("Dion", "dion", "dion_1e-3")]


def colour(label):
    return STYLE[label]["color"] if label in STYLE else EXTRA[label]


def main():
    comm = json.load(open(COMM))["methods"]
    steps = json.load(open(STEPS))

    def split(row):
        """(gradient sync MB, optimizer MB) per step"""
        by = comm[row]["by_source"]
        g = sum(v["MB_per_step"] for k, v in by.items() if k.startswith("grad_sync"))
        o = sum(v["MB_per_step"] for k, v in by.items() if k.startswith("optimizer"))
        return g, o

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    # ---- left: bytes per step, gradient sync vs the optimizer's own all-reduce
    # SignMuon P=2 sends exactly what LionMuon P=2 sends, so it is not a separate bar.
    bars = [r for r in ROWS if r[0] != "SignMuon P=2"]
    for i, (label, row, _) in enumerate(bars):
        g, o = split(row)
        c = colour(label)
        ax1.bar(i, g, color=c, alpha=0.35, edgecolor=c, linewidth=1.2)
        ax1.bar(i, o, bottom=g, color=c, edgecolor="black", linewidth=0.5, hatch="//")
        ax1.text(i, g + o + 12, f"{g + o:.0f}", ha="center", fontsize=9)
        if o:
            ax1.text(i, g + o / 2, f"{o:.0f}", ha="center", va="center", fontsize=8,
                     color="white", fontweight="bold")
    ax1.set_xticks(range(len(bars)))
    ax1.set_xticklabels([l.replace(" (P=1)", "").replace(" (P=5)", " P=5") for l, _, _ in bars],
                        rotation=20, ha="right", fontsize=9)
    ax1.set_ylabel("MB sent per step (4 GPUs)", fontsize=11)
    ax1.set_title("What one step sends: gradients (plain) and\nthe optimizer's own all-reduce (hatched)",
                  fontsize=11)
    ax1.grid(alpha=0.3, axis="y")
    ax1.set_axisbelow(True)

    # ---- right: bytes to reach Muon's final loss
    target = steps["target"]
    pts = []
    for label, row, key in ROWS:
        g, o = split(row)
        n = steps["steps"].get(key)
        pts.append((label, None if n is None else n * (g + o) / 1e6))
    reached = [(l, tb) for l, tb in pts if tb is not None]
    reached.sort(key=lambda t: t[1])
    for i, (label, tb) in enumerate(reached):
        ax2.barh(i, tb, color=colour(label), edgecolor="black", linewidth=0.5)
        ax2.text(tb + 1.5, i, f"{tb:.0f} TB", va="center", fontsize=9)
    miss = [l for l, tb in pts if tb is None]
    for j, label in enumerate(miss):
        i = len(reached) + j
        ax2.barh(i, max(t for _, t in reached), color="white", edgecolor=colour(label),
                 linewidth=1.5, hatch="xx")
        ax2.text(2, i, "never reaches this loss", va="center", fontsize=9, style="italic")
    ax2.set_yticks(range(len(reached) + len(miss)))
    ax2.set_yticklabels([l for l, _ in reached] + miss, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("TB sent to reach Muon's final loss", fontsize=11)
    ax2.set_title(f"Cost of getting to loss {target}\n(steps measured, bytes counted)", fontsize=11)
    ax2.margins(x=0.22)
    ax2.grid(alpha=0.3, axis="x")
    ax2.set_axisbelow(True)

    fig.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out = os.environ.get("OUT_PNG", "results/figures/comms_ddp_124m.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"{out} written")
    for label, tb in pts:
        g, o = split(dict((l, r) for l, r, _ in ROWS)[label])
        print(f"  {label:<14} {g + o:6.1f} MB/step ({o:5.1f} exposed)   "
              + (f"{tb:5.1f} TB to {target}" if tb else "never reaches the target"))


if __name__ == "__main__":
    main()
