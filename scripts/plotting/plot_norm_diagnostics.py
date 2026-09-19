"""The constants of the bound during training, the same five the first version of this paper
showed, plus the update ratio that says how tight the dense approximation is.

Panels, all against training step and for both datasets:
  alpha = |G|_1/|G|_nuc, rho_nuc = |G-M|_nuc/|G-M|_F, rho_1 = |G-M|_1/|G-M|_F,
  L_2 = |dG|_nuc/|dW|_2, L_inf = |dG|_1/|dW|_inf, and |U|_2/|U|_inf of the applied update
  against the worst case sqrt(mn).
The legend of each panel carries the mean over training, which is the number the text quotes.

    python scripts/plotting/plot_norm_diagnostics.py fw.json:FineWeb wt.json:WikiText-103
"""
import json
import os
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_results import finish_axis  # noqa: E402

COLORS = ["#1144cc", "#cc1111"]          # the blue and red of the two optimizer families
WORST_CASE = 768.0                       # sqrt(mn) for the 768 x 768 matrices of the 124M model

# key in the record -> (panel title, invert the stored ratio?, log scale?)
PANELS = [
    ("grad_nuc_over_1", r"$\alpha=\|G\|_1/\|G\|_{\rm nuc}$", True, True),
    ("errmom_F_over_nuc", r"$\rho_{\rm nuc}=\|G-M\|_{\rm nuc}/\|G-M\|_F$", True, True),
    ("errmom_F_over_1", r"$\rho_1=\|G-M\|_1/\|G-M\|_F$", True, False),
    ("smooth_nuc_over_2", r"$L_2=\|\Delta G\|_{\rm nuc}/\|\Delta W\|_2$", False, True),
    ("smooth_1_over_inf", r"$L_\infty=\|\Delta G\|_1/\|\Delta W\|_\infty$", False, True),
    ("update_2_over_inf", r"$\|U\|_2/\|U\|_\infty$ of the update", False, False),
]


def load(path):
    d = json.load(open(path))
    return d["records"] if isinstance(d, dict) and "records" in d else d


def main(args):
    runs = [(p.split(":")[1], load(p.split(":")[0])) for p in args]
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.6))

    for ax, (key, title, invert, logy) in zip(axes.ravel(), PANELS):
        for (name, rows), c in zip(runs, COLORS):
            # the two smoothness proxies are single-batch estimates and have gaps
            pts = [(r["step"], (1 / r[key]) if invert else r[key])
                   for r in rows if r.get(key) is not None and r[key] == r[key]]
            step, y = [p[0] for p in pts], [p[1] for p in pts]
            ax.plot(step, y, color=c, lw=1.5, label=f"{name} ($\\mu={st.mean(y):.1f}$)")
        if key == "update_2_over_inf":
            ax.axhline(WORST_CASE, color="0.4", ls=":", lw=1.4)
            ax.text(0.97, 0.93, f"worst case $\\sqrt{{mn}}={WORST_CASE:.0f}$", fontsize=8,
                    color="0.3", ha="right", va="top", transform=ax.transAxes)
            ax.set_ylim(0, WORST_CASE * 1.18)
        if logy:
            ax.set_yscale("log")
        finish_axis(ax, "step", title)
        ax.margins(x=0.02)
        ax.legend(fontsize=7.5, frameon=False)

    fig.tight_layout()
    out = os.environ.get("OUT_PNG", "results/figures/norm_diagnostics_124m.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"{out} written")
    for name, rows in runs:
        vals = {}
        for key, _, invert, _ in PANELS:
            series = [(1 / r[key]) if invert else r[key] for r in rows
                      if r.get(key) and r[key] == r[key]]
            vals[key] = st.mean(series)
        a, rn, r1 = vals["grad_nuc_over_1"], vals["errmom_F_over_nuc"], vals["errmom_F_over_1"]
        L2, Li = vals["smooth_nuc_over_2"], vals["smooth_1_over_inf"]
        print(f"  {name:<14} alpha {a:5.1f}  rho_nuc {rn:5.1f}  rho_1 {r1:6.1f}  L2 {L2:5.2f}  "
              f"Linf {Li:7.1f}  Linf/(a^2 L2) {Li/(a*a*L2):.2f}  rho1/(a rho_nuc) {r1/(a*rn):.2f}  "
              f"|U|2/|U|inf {vals['update_2_over_inf']:.0f}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["fw.json:FineWeb", "wt.json:WikiText-103"])
