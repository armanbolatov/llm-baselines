# LionMuon

Code for the paper *LionMuon: Alternating Spectral and Sign Descent for Efficient Training*.
LionMuon takes one Muon step every `P` iterations and Lion steps in between, with one momentum
buffer shared by both. SignMuon is the single-EMA variant. The code is a trimmed fork of the
`llm-baselines` benchmark (GPT-2-style decoders on FineWeb and WikiText-103), kept under its
MIT license.

## Setup

```
pip install -r requirements.txt
```

Datasets are tokenized on first use into `--datasets_dir` (WikiText-103, and the FineWeb
sample chosen by `FINEWEB_SAMPLE`, default `sample-10BT`).

## The optimizer

`src/optim/lion_muon.py` implements Algorithm 1 of the paper. The flags that matter:

```
--opt lion_muon --muon_every_k P --muon_lr_factor ETA_M --sign_lr ETA_L --beta1 B1 --beta2 B2
```

`--beta1 == --beta2` gives SignMuon, `--muon_every_k 1` gives Muon, a very large period gives
Lion or Signum. Under DDP the sign steps are local and only the Muon steps all-reduce their
updates. Two ablation flags, `--muon_random_period` and `--one_d_lion`, are described in the
appendix. Baselines: `--opt adamw`, `--opt dion` (`src/optim/dion.py`), `--opt muonbp`
(`src/optim/muonbp.py`).

## Reproducing the paper

| what | how |
|---|---|
| 124M tuning grid, baselines, seeds, ablations | `scripts/run_experiments.sh {grid,base,seeds,ablate,extra} {fineweb,wikitext} 124m`, pruning rule in `scripts/prune.sh` |
| 355M with transferred hyperparameters | `scripts/run_experiments.sh seeds fineweb 355m` |
| 4-GPU data-parallel runs | `scripts/ddp_train_124m.sh` (round-robin over methods, resumes from checkpoints) |
| FLOPs, bytes and step time per step | `scripts/cost_bench.py --mode {flops,comm,time}` |
| norm constants and the tail index | `scripts/run_norm_diagnostics.sh`, `scripts/estimate_kappa.py` |
| full-split re-scoring of finished runs | `scripts/eval_val_anchor.py` |
| figures and tables | `scripts/plotting/plot_124m.py`, `plot_loss_vs_cost.py`, `plot_norm_diagnostics.py`, `grid_table.py` |
| Muon with Nesterov momentum, the beta1 of LionMuon | `scripts/run_experiments.sh more {fineweb,wikitext} 124m` |
| periods 3 and 10 | the `grid` mode with `GRID_ITERS=64000 GRID_WU=3000 PS=3 LRS=2e-3 ALPHAS=20 BETAS=0.9:0.99` (example in the script) |
| Qwen3-style decoder (`src/models/qwen.py`) | `scripts/run_experiments.sh qwen fineweb 124m` |
| where the time of a step goes | `scripts/cost_bench.py --mode breakdown` |

The run summaries the plots read are in `exps/` and `exps_124m/` (124M, one `summary.json`
per run), and the measured costs, DDP results, 355M results and ablations are in `results/`.

## Additional experiments

`results/additional/` holds the runs of the appendix "Additional Experiments":

| file | what |
|---|---|
| `runs_124m.jsonl` | best and final validation loss of every 124M run: periods 3 and 10 (`*g_p3_*`, `*g_p10_*`), Muon with Nesterov momentum (`*n_nest_m<momentum>_*`), Muon and LionMuon P=2 rerun next to it (`*n_muon_*`, `*n_lmp2_ctrl_*`), the beta1 of LionMuon (`*n_lmp2_b1-*`), and the Qwen3-style decoder (`qw124*`). `fw` is FineWeb, `wt` is WikiText-103, and the name carries the rates: `lr` is the Muon rate, `a` the ratio of the Muon rate to the sign-step rate, `sl` an explicit sign-step rate. |
| `long_horizon_150k/` | 150,000 steps on one GPU with the batch and schedule of the 4-GPU runs: LionMuon P=2, Muon, Muon with Nesterov momentum 0.99 and AdamW. Each summary has the full configuration under `args`. |
| `355m_higher_rate/` | 355M at a Muon rate of 3e-3, two seeds each for LionMuon P=2 and Muon, and Muon with Nesterov momentum at the copied rate. |
| `step_breakdown/` | the optimizer step split into Newton-Schulz and the update all-reduce, without and with 16-fold gradient accumulation, and the step times of the same node without the instrumentation. |

The time to Muon's loss with bootstrap intervals is in `results/ddp124/time_to_target_ci.json`.
