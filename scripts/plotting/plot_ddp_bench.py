"""Wall-clock per step under real DDP (scripts/ddp_bench.sh) -> table + figure.

Parses results/benchmarks/ddp_bench*/<opt>_g<N>_ebs<B>.log, takes the median
iter_dt over iterations >= 100 per run, and averages over repeats.

    python scripts/plotting/plot_ddp_bench.py
"""
import glob
import json
import os
import re
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["adamw", "muon", "lionmuon_p2", "lionmuon_p5", "muonbp_p5", "dion"]
LABEL = {"adamw": "AdamW", "muon": "Muon (P=1)", "lionmuon_p2": "LionMuon P=2",
         "lionmuon_p5": "LionMuon P=5", "muonbp_p5": "MuonBP (P=5)", "dion": "Dion (r=1/4)"}
COLOR = {"adamw": "#888888", "muon": "#8b0000", "lionmuon_p2": "#1144cc",
         "lionmuon_p5": "#3377ee", "muonbp_p5": "#2e8b57", "dion": "#b8860b"}
PAT = re.compile(r"Iter=(\d+) .*?wall=([0-9.]+)")


def median_dt(log):
    """Median synchronized wall-clock per iteration over iterations >= 100."""
    pts = [(int(it), float(w)) for it, w in PAT.findall(open(log).read())]
    vals = [(w - pw) / (it - pit) for (pit, pw), (it, w) in zip(pts, pts[1:]) if it >= 100]
    return st.median(vals) if vals else None


runs = {}  # (opt, ngpu, ebs) -> [median per repeat]
for log in glob.glob("results/benchmarks/ddp_bench*/*_g*_ebs*.log"):
    m = re.match(r"(.+)_g(\d+)_ebs(\d+)\.log$", os.path.basename(log))
    if not m:
        continue
    v = median_dt(log)
    if v is not None:
        runs.setdefault((m.group(1), int(m.group(2)), int(m.group(3))), []).append(v)

rows = []
for (opt, ngpu, ebs), vals in sorted(runs.items(), key=lambda kv: (kv[0][1], kv[0][2], ORDER.index(kv[0][0]) if kv[0][0] in ORDER else 99)):
    rows.append(dict(opt=opt, ngpu=ngpu, ebs=ebs, s_per_step=st.mean(vals),
                     spread=(max(vals) - min(vals)) if len(vals) > 1 else 0.0, n=len(vals),
                     tok_per_s=ebs * 512 / st.mean(vals)))

print(f"{'optimizer':<14}{'gpus':>5}{'ebs':>5}{'s/step':>9}{'±':>7}{'tok/s':>9}{'vs AdamW':>10}{'vs Muon':>9}")
for ebs in sorted({r['ebs'] for r in rows}):
    base = {r['opt']: r['s_per_step'] for r in rows if r['ebs'] == ebs}
    for r in rows:
        if r['ebs'] != ebs:
            continue
        print(f"{LABEL.get(r['opt'], r['opt']):<14}{r['ngpu']:>5}{r['ebs']:>5}{r['s_per_step']:>9.3f}"
              f"{r['spread']/2:>7.3f}{r['tok_per_s']:>9.0f}"
              f"{r['s_per_step']/base.get('adamw', float('nan')):>10.2f}x{r['s_per_step']/base.get('muon', float('nan')):>8.2f}x")

os.makedirs("results/benchmarks", exist_ok=True)
json.dump(rows, open("results/benchmarks/ddp_bench_124m.json", "w"), indent=2)

ebss = sorted({r['ebs'] for r in rows})
fig, axes = plt.subplots(1, len(ebss), figsize=(6 * len(ebss), 4.5), sharey=False)
axes = [axes] if len(ebss) == 1 else list(axes)
for ax, ebs in zip(axes, ebss):
    sub = {r['opt']: r for r in rows if r['ebs'] == ebs}
    opts = [o for o in ORDER if o in sub]
    ys = list(range(len(opts) - 1, -1, -1))
    for y, o in zip(ys, opts):
        r = sub[o]
        ax.barh(y, r['s_per_step'], height=0.62, color=COLOR[o], edgecolor="black", linewidth=0.5,
                xerr=r['spread'] / 2 if r['spread'] else None, error_kw=dict(lw=1.2, capsize=3))
        ax.text(r['s_per_step'] + 0.01, y, f"{r['s_per_step']:.2f} s", va="center", fontsize=11)
    ax.set_yticks(ys)
    ax.set_yticklabels([LABEL[o] for o in opts], fontsize=12)
    ngpu = next(iter(sub.values()))['ngpu']
    ax.set_title(f"{ngpu}xA100 DDP, batch {ebs} ({ebs // ngpu}/GPU)", fontsize=13)
    ax.set_xlabel("Wall-clock per training step (s)", fontsize=13)
    ax.set_xlim(0, max(r['s_per_step'] for r in sub.values()) * 1.25)
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_axisbelow(True)
plt.tight_layout()
os.makedirs("results/figures", exist_ok=True)
plt.savefig("results/figures/step_time_ddp_124m.png", dpi=150, bbox_inches="tight")
print("saved results/figures/step_time_ddp_124m.png")
