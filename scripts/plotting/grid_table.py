"""Compact LaTeX table of the 124M tuning sweep: range covered, chosen cell, and the neighbours
that make the optimum interior. A table handles a sparse sweep better than a heatmap does.
    python grid_table.py            # FineWeb
    TAG=wt python grid_table.py
"""
import json
import os

TAG = os.environ.get("TAG", "fw")
FAMILY = {"09-09": "SignMuon", "09-099": "LionMuon"}
PERIODS = [1, 2, 5, 20]
MIN_EVALS = 100

full, cut = {}, {}
for line in open("results/tuning/grid_124m.jsonl"):
    r = json.loads(line)
    if r["tag"] != TAG or r["betas"] not in FAMILY or r["p"] not in PERIODS:
        continue
    try:
        a = float(r["alpha"])
    except ValueError:
        a = 1.0
    key = (FAMILY[r["betas"]], r["p"])
    (full if r["evals"] >= MIN_EVALS else cut).setdefault(key, {})[(float(r["lr"]), a)] = r["loss"]


def g(x):
    return f"{x:g}"


print(r"\begin{tabular}{llcccc}")
print(r"\toprule")
print(r"method & $\eta_M$ swept & $\alpha$ swept & cells & chosen $(\eta_M,\alpha)$ & "
      r"loss (next $\eta_M$ down / up) \\")
print(r"\midrule")
for fam in ("SignMuon", "LionMuon"):
    for p in PERIODS:
        d = full.get((fam, p))
        if not d:
            continue
        c = cut.get((fam, p), {})
        lrs = sorted({lr for lr, _ in d} | {lr for lr, _ in c})
        als = sorted({a for _, a in d} | {a for _, a in c})
        (blr, bal), bl = min(d.items(), key=lambda kv: kv[1])
        # neighbours along eta_M at the chosen alpha; "cut" means stopped early, i.e. clearly worse
        nb = []
        for step in (-1, 1):
            j = lrs.index(blr) + step
            if 0 <= j < len(lrs):
                k = (lrs[j], bal)
                nb.append(f"{d[k]:.3f}" if k in d else ("cut" if k in c else "--"))
            else:
                nb.append("--")
        name = {("SignMuon", 1): "Muon ($P{=}1$)", ("LionMuon", 1): "LionMuon $P{=}1$"}.get(
            (fam, p), f"{fam} $P{{=}}{p}$")
        alpha_rng = "--" if len(als) == 1 and als[0] == 1.0 else f"{g(min(als))}--{g(max(als))}"
        chosen = g(blr) if alpha_rng == "--" else f"({g(blr)}, {g(bal)})"
        print(f"{name} & {g(min(lrs))}--{g(max(lrs))} & {alpha_rng} & {len(d)}+{len(c)} & "
              f"{chosen} & \\textbf{{{bl:.3f}}} ({nb[0]} / {nb[1]}) \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
