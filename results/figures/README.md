# Figure naming

`<metric>_<setting>.png`, all lowercase, underscores only.

| file | what it shows | script |
|---|---|---|
| `loss_vs_flops_124m.png` | 124M tuned grid, FineWeb and WikiText-103, 3 seeds | `plot_124m.py` |
| `loss_vs_cost_355m_ddp.png` | 355M vs FLOPs, and 124M under 4-GPU DDP vs measured communication | `plot_loss_vs_cost.py` |
| `loss_vs_flops_355m_ddp.png` | same, right panel against counted FLOPs (`DDP_X=flops`) | `plot_loss_vs_cost.py` |
| `loss_vs_time_ddp_124m.png` | 124M DDP training curves vs tokens and wall-clock | `plot_ddp124.py` |
| `tuning_heatmap_124m.png`, `_wt.png` | learning-rate / ratio grid, regenerated from the 64k runs — **not used in the paper**, the sweep is a table instead (`grid_table.py`) | `plot_tuning_heatmap.py` |
| `norm_diagnostics_124m.png`, `norm_ratio_124m.png` | norm diagnostics during training | `plot_norm_diagnostics.py` |

In the paper build, the per-dataset training curves follow the same scheme as
`curves_<dataset>_<arch>_<size>.png`, and the estimated theory constants are `constants_124m.png`.

`plot_comms.py` still prints the per-step byte table and the bytes-to-target numbers, but its
figure is no longer used: the byte comparison now lives in the right panel of
`loss_vs_cost_355m_ddp.png` (MB per step; `DDP_X=comm_total` gives the same axis in TB for the
whole run, `DDP_X=flops` gives counted FLOPs).
