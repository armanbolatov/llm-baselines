"""Plot WSM (arXiv:2507.17634) runs against the existing cos-scheduled
baselines for `adamw` and `lionmuon_k2` on FineWeb / 124M GPT-base.

For each optimizer, draws:
  - cos baseline training curve (from exps/fw_base_<opt>/summary.json)
  - WSM training curve            (from exps/fw_base_<opt>_wsm/summary.json)
  - final WSM val_loss for each merge method, plotted as a marker on the
    right edge (from exps/wsm_fw_summary.json, if present)

Writes one PNG per optimizer to results/wsm_<opt>.png plus a combined
results/wsm_all.png.

Usage:
    python scripts/plotting/plot_wsm.py            # both opts
    python scripts/plotting/plot_wsm.py adamw      # one opt
"""

import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPS_DIR = os.path.join(REPO_ROOT, "exps")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

OPTS = ["adamw", "lionmuon_k2"]
LABELS = {"adamw": "AdamW", "lionmuon_k2": "LionMuon (P=2)"}

MERGE_STYLE = {
    "mean":    {"color": "#2c8a4f", "marker": "v", "name": "mean"},
    "ema":     {"color": "#7a3fa0", "marker": "D", "name": "EMA (α=0.5)"},
    "theorem": {"color": "#d4a017", "marker": "*", "name": "theorem (1−√t)"},
}


def load_curve(exp_dir):
    summary = os.path.join(exp_dir, "summary.json")
    if not os.path.isfile(summary):
        return None
    with open(summary) as f:
        d = json.load(f)
    val_loss = d.get("val_loss") or []
    if not val_loss:
        return None
    eval_interval = (d.get("args") or {}).get("eval_interval", 500)
    return {
        "iters": [i * eval_interval for i in range(len(val_loss))],
        "losses": val_loss,
    }


def load_merge_eval(opt):
    """Return {method: val_loss} for the given opt from the eval summary."""
    path = os.path.join(EXPS_DIR, "wsm_fw_summary.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        rows = json.load(f)
    out = {}
    for row in rows:
        if f"fw_base_{opt}_wsm" not in row.get("ckpt", ""):
            continue
        meta = row.get("merge_meta")
        if not meta:
            continue
        method = meta.get("method") or meta.get("merge_method")
        if method in MERGE_STYLE:
            out[method] = row.get("val_loss")
    return out


def plot_one(opt, ax=None):
    cos = load_curve(os.path.join(EXPS_DIR, f"fw_base_{opt}"))
    wsm = load_curve(os.path.join(EXPS_DIR, f"fw_base_{opt}_wsm"))
    if cos is None and wsm is None:
        print(f"[skip] {opt}: no summary.json under exps/fw_base_{opt}[_wsm]")
        return False
    merges = load_merge_eval(opt)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8.0, 5.0))

    if cos is not None:
        ax.plot(cos["iters"], cos["losses"],
                color="#1f4e8a", lw=2.0,
                label=f"cos baseline (final={cos['losses'][-1]:.4f})")
    if wsm is not None:
        ax.plot(wsm["iters"], wsm["losses"],
                color="#b8410e", lw=2.0,
                label=f"WSM (final={wsm['losses'][-1]:.4f})")

    if merges and (wsm or cos):
        x_end = (wsm or cos)["iters"][-1]
        for method, loss in merges.items():
            if loss is None:
                continue
            s = MERGE_STYLE[method]
            ax.scatter([x_end], [loss],
                       color=s["color"], marker=s["marker"], s=140,
                       zorder=5, edgecolor="black", linewidths=0.6,
                       label=f"WSM-merge {s['name']} = {loss:.4f}")

    y_min = min(min(c["losses"]) for c in (cos, wsm) if c is not None)
    ax.set_ylim(max(1e-6, y_min * 0.97), 4.5)
    ax.set_yscale("log")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Val loss", fontsize=12)
    ax.set_title(f"{LABELS[opt]} — cos vs WSM (FineWeb, 124M GPT-base)", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")

    if own_fig:
        plt.tight_layout()
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out = os.path.join(RESULTS_DIR, f"wsm_{opt}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to {out}")

        # Numerical summary.
        print(f"\n{LABELS[opt]} — final val_loss:")
        if cos is not None:
            print(f"  {'cos baseline':<22s} {cos['losses'][-1]:.4f}  "
                  f"(pp={math.exp(min(cos['losses'][-1],80)):.2f})")
        if wsm is not None:
            print(f"  {'WSM (no merge)':<22s} {wsm['losses'][-1]:.4f}  "
                  f"(pp={math.exp(min(wsm['losses'][-1],80)):.2f})")
        for method, loss in merges.items():
            if loss is None:
                continue
            print(f"  {'WSM-' + MERGE_STYLE[method]['name']:<22s} {loss:.4f}  "
                  f"(pp={math.exp(min(loss,80)):.2f})")
    return True


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else OPTS
    if any(t not in OPTS for t in targets):
        print(f"Unknown opt. Choose from: {OPTS}")
        sys.exit(1)

    plotted = [opt for opt in targets if plot_one(opt)]

    if len(plotted) > 1:
        fig, axes = plt.subplots(1, len(plotted), figsize=(8.0 * len(plotted), 5.0))
        if len(plotted) == 1:
            axes = [axes]
        for ax, opt in zip(axes, plotted):
            plot_one(opt, ax=ax)
        plt.tight_layout()
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out = os.path.join(RESULTS_DIR, "wsm_all.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nCombined plot saved to {out}")


if __name__ == "__main__":
    main()
