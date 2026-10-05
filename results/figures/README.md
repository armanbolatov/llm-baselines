# Figures

`<metric>_<setting>.png`, all lowercase, underscores only.

| file | what it shows | script |
|---|---|---|
| `loss_vs_flops_124m.png` | 124M tuned grid, FineWeb and WikiText-103, three seeds (Figure 2) | `plot_124m.py` |
| `loss_vs_flops_355m.png` | 355M with transferred hyperparameters, mean and standard deviation over three seeds (Figure 3) | `plot_loss_vs_cost.py` |
| `loss_vs_cost_ddp124.png` | 124M under 4-GPU data parallelism against the bytes one step sends (Figure 4) | `plot_loss_vs_cost.py` |
| `loss_vs_cost_355m_ddp.png` | the two panels above side by side | `plot_loss_vs_cost.py` |
| `norm_diagnostics_124m.png` | the constants of the bound during training (appendix) | `plot_norm_diagnostics.py` |

The plotting scripts read run summaries from `exps/` and `exps_124m/`, the measured costs from
`results/benchmarks/cost/`, the DDP results from `results/ddp124/` and the 355M results from
`results/scaling/`. The tuning tables of the appendix come from `grid_table.py`.
