# Per-iteration norm-ratio diagnostics

This pipeline records 7 norm ratios at each optimizer step and plots their
trajectories across training. It is wired into `LionMuon` (and used as-is for
SignMuon, since SignMuon is the `β1=β2` special case of LionMuon).

The 7 ratios per iteration:

| key                  | meaning                                                                |
|----------------------|------------------------------------------------------------------------|
| `grad_nuc_over_1`    | $\|G\|_{\mathrm{nuc}} / \|G\|_{1}$ — gradient nuclear vs.\ $\ell_1$    |
| `param_2_over_inf`   | $\|W\|_{2}    / \|W\|_{\infty}$ — parameter spectral vs.\ $\ell_\infty$|
| `update_2_over_inf`  | $\|U\|_{2}    / \|U\|_{\infty}$ — post-LMO update (sign or NS)         |
| `errmom_F_over_nuc`  | $\|G-M\|_{F} / \|G-M\|_{\mathrm{nuc}}$ — momentum-residual            |
| `errmom_F_over_1`    | $\|G-M\|_{F} / \|G-M\|_{1}$ — momentum-residual                       |
| `smooth_nuc_over_2`  | $\|\nabla f(W_{t+1})-\nabla f(W_{t})\|_{\mathrm{nuc}} / \|W_{t+1}-W_{t}\|_{2}$ — smoothness proxy in $L_2$ pairing |
| `smooth_1_over_inf`  | $\|\nabla f(W_{t+1})-\nabla f(W_{t})\|_{1}   / \|W_{t+1}-W_{t}\|_{\infty}$ — smoothness proxy in $L_\infty$ pairing |

Each entry stores the **median across all 2D matrices** in the model at that
iteration. Smoothness is a noisy single-batch estimate (different mini-batches
at $t$ and $t{+}1$); plot smoothing in the plotter mitigates this.

---

## How to record

Add `--norm_diag` to the training command. The diagnostic JSON lands at
`<exp_dir>/norm_diag.json` (atomically rewritten every 10 records ≈ every 500
steps with default settings).

Optional flags:
- `--norm_diag_every_k 50` — record every 50 optimizer steps (default).
- `--norm_diag_path /custom/path.json` — override location.

### The runs of the paper

`scripts/run_norm_diagnostics.sh` runs the 124M model (12 layers, width 768) for 10,000 steps
with a record every 100 steps, on FineWeb and WikiText-103:

```bash
bash scripts/run_norm_diagnostics.sh 0                       # LionMuon P=2: the constants table and the figure
PERIODS="1 2 5 inf" bash scripts/run_norm_diagnostics.sh 0   # the comparison across periods (inf is pure Lion)
```

The two smoothness entries of a record are differences between this record and the previous
one, so with `--norm_diag_every_k 100` they compare weights and gradients 100 steps apart.
The records of the runs in the paper are in `results/additional/constants_124m/`.

---

## How to plot and tabulate

```bash
D=results/additional/constants_124m
python scripts/plotting/plot_norm_diagnostics.py \
  $D/ndL12_fw_p2_seed0.norm_diag.json:FineWeb $D/ndL12_wt_p2_seed0.norm_diag.json:WikiText-103
python scripts/plotting/constants_table.py --ref P=1 \
  P=1=$D/ndL12_fw_p1_seed0.norm_diag.json P=2=$D/ndL12_fw_p2_seed0.norm_diag.json \
  P=5=$D/ndL12_fw_p5_seed0.norm_diag.json Lion=$D/ndL12_fw_pinf_seed0.norm_diag.json
```

The figure has one panel per constant with both datasets in it and is written to
`results/figures/norm_diagnostics_124m.png` (set `OUT_PNG` for another path). The table has
one row per run: the constants as means over the records, the two ratios of the same run, and
with `--ref` the ratios against the reference run.

---

## Cost notes

- Diagnostic recording adds an extra forward-free SVD per 2D matrix per logged
  step. At 124M with ~50 matrices and `every_k=50`, that's ~50 SVDs every 50
  steps — negligible against forward+backward.
- Cloning `(p, g, m, u)` per logged matrix on logged steps only — also small.
- Disabling: omit `--norm_diag` (zero overhead path; hook is gated by
  `norm_diag_active`).
