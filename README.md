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

The run summaries the plots read are in `exps/` and `exps_124m/` (124M, one `summary.json`
per run), and the measured costs, DDP results, 355M results and ablations are in `results/`.
