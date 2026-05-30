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
    "mean":       {"color": "#2c8a4f", "marker": "v", "name": "mean"},
    "ema":        {"color": "#7a3fa0", "marker": "D", "name": "EMA (α=0.5)"},
    "theorem":    {"color": "#d4a017", "marker": "*", "name": "theorem (1−√t)"},
    "online":     {"color": "#9c5d10", "marker": "X", "name": "online theorem (postproc)"},
    "online_live":{"color": "#5f3a08", "marker": "P", "name": "online theorem (LIVE in training)"},
    "sampled_k1": {"color": "#e98a6a", "marker": "o", "name": "sampled K=1"},
    "sampled_k2": {"color": "#dd6644", "marker": "s", "name": "sampled K=2"},
    "sampled_k3": {"color": "#bb3322", "marker": "p", "name": "sampled K=3"},
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
    args = d.get("args") or {}
    eval_interval = args.get("eval_interval", 500)
    n_iter = args.get("iterations", len(val_loss) * eval_interval)
    # Detect resumed runs (WSD finetune): val_loss starts at the first
    # eval-aligned iter AFTER the resume point, not at iter 0.
    resume_from = args.get("resume_from")
    start = 0
    if resume_from:
        leaf = os.path.basename(str(resume_from).rstrip("/"))
        if leaf.isdigit():
            r = int(leaf)
            start = ((r + eval_interval - 1) // eval_interval) * eval_interval
    iters = list(range(start, start + len(val_loss) * eval_interval, eval_interval))
    # Clamp the last point to n_iterations if eval-at-final triggered.
    if iters and iters[-1] != n_iter and abs(iters[-1] - n_iter) < eval_interval:
        iters[-1] = n_iter
    return {"iters": iters, "losses": val_loss}


def load_full_eval(opt):
    """All full-val-set eval results for an opt from wsm_fw_summary.json.

    Returns a dict with keys among {'wsm_final', 'wsd_final', 'mean', 'ema',
    'theorem'} mapped to val_loss.
    """
    path = os.path.join(EXPS_DIR, "wsm_fw_summary.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        rows = json.load(f)
    out = {}
    for row in rows:
        ckpt = (row.get("ckpt") or "").replace("\\", "/")
        loss = row.get("val_loss")
        meta = row.get("merge_meta")
        if meta:
            method = meta.get("method") or meta.get("merge_method")
            key = method
            if method == "sampled":
                key = f"sampled_k{meta.get('k')}"
            if key in MERGE_STYLE and f"fw_base_{opt}_wsm" in ckpt:
                out[key] = loss
        elif f"fw_base_{opt}_wsm/ckpts/" in ckpt and "/merged_" not in ckpt:
            out["wsm_final"] = loss
        elif f"fw_base_{opt}_wsd/ckpts/" in ckpt:
            out["wsd_final"] = loss
    return out


def plot_one(opt, ax=None):
    cos = load_curve(os.path.join(EXPS_DIR, f"fw_base_{opt}"))
    wsm = load_curve(os.path.join(EXPS_DIR, f"fw_base_{opt}_wsm"))
    wsd = load_curve(os.path.join(EXPS_DIR, f"fw_base_{opt}_wsd"))
    if cos is None and wsm is None and wsd is None:
        print(f"[skip] {opt}: no summary.json under exps/fw_base_{opt}[_wsm|_wsd]")
        return False
    eval_full = load_full_eval(opt)
    merges = {m: eval_full[m] for m in MERGE_STYLE if m in eval_full}

    # Prefer full-val eval results for the printed/legend final losses
    # (cos baseline still uses training-time eval since we don't save its
    # final ckpt). The trajectory curves themselves are always training-time
    # evals — they show the trajectory shape, not the final number.
    cos_final = cos["losses"][-1] if cos is not None else None
    wsm_final = eval_full.get("wsm_final",
                              wsm["losses"][-1] if wsm is not None else None)
    wsd_final = eval_full.get("wsd_final",
                              wsd["losses"][-1] if wsd is not None else None)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8.0, 5.0))

    if cos is not None:
        ax.plot(cos["iters"], cos["losses"],
                color="#7aa6d6", lw=1.8, alpha=0.8,
                label=f"cos baseline (final={cos_final:.4f}*)")
    if wsm is not None:
        ax.plot(wsm["iters"], wsm["losses"],
                color="#b8410e", lw=2.0,
                label=f"WSM no merge (final={wsm_final:.4f})")
    if wsd is not None:
        ax.plot(wsd["iters"], wsd["losses"],
                color="#1f4e8a", lw=2.2,
                label=f"WSD decay (final={wsd_final:.4f})")

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

    y_min = min(min(c["losses"]) for c in (cos, wsm, wsd) if c is not None)
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
        n_full = "full val" if eval_full else "training-time eval (32 batches)"
        print(f"\n{LABELS[opt]} -- final val_loss "
              f"(* = training-time 32-batch eval; rest = full val set):")
        if cos_final is not None:
            print(f"  {'cos baseline*':<22s} {cos_final:.4f}  "
                  f"(pp={math.exp(min(cos_final,80)):.2f})")
        if wsd_final is not None:
            tag = "WSD (decay)" if "wsd_final" in eval_full else "WSD (decay)*"
            print(f"  {tag:<22s} {wsd_final:.4f}  "
                  f"(pp={math.exp(min(wsd_final,80)):.2f})")
        if wsm_final is not None:
            tag = "WSM (no merge)" if "wsm_final" in eval_full else "WSM (no merge)*"
            print(f"  {tag:<22s} {wsm_final:.4f}  "
                  f"(pp={math.exp(min(wsm_final,80)):.2f})")
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
