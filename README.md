# LionMuon

Code accompanying the paper [LionMuon: Alternating Spectral and Sign Descent for Efficient Training](https://arxiv.org/abs/2605.19811).

This repository is a fork of Andrei Semenov's [`llm-baselines`](https://github.com/epfml/llm-baselines), extended with the `LionMuon` and `SignMuon` optimizers and the experiments reported in the paper. Many thanks to the original contributors for maintaining such a clean and reproducible codebase.

## Algorithm

![LionMuon algorithm](imgs/algo.png)

`SignMuon` is the special case `beta1 = beta2`: the dual-EMA collapses to a single Signum-style buffer, but the alternation between Newton-Schulz and elementwise sign is preserved. Pure `Signum`, `Lion`, and `Muon` are recovered as further special cases of the same algorithm:

| Optimizer | Momentum | Period |
| --- | --- | --- |
| `Signum` (Bernstein et al., 2018) | `beta1 = beta2` | `P = inf` |
| `Lion` (Chen et al., 2024) | `beta1 != beta2` (dual-EMA) | `P = inf` |
| `Muon` (Jordan et al., 2024) | `beta1 = beta2` | `P = 1` |
| `SignMuon` **(this work)** | `beta1 = beta2` | any `P` |
| `LionMuon` **(this work)** | `beta1 != beta2` (dual-EMA) | any `P` |

1D parameters (biases, LayerNorm/RMSNorm gains) fall back to AdamW with a small fixed learning rate, following the standard Muon-hybrid convention. The optimizer state therefore matches `Lion` / `Muon` and is exactly half of `AdamW`.

Reference implementations live in [src/optim/lion_muon.py](src/optim/lion_muon.py) and [src/optim/sign_muon.py](src/optim/sign_muon.py); Newton-Schulz orthogonalization is in [src/optim/muon.py](src/optim/muon.py).

## Install

```bash
pip install -r requirements.txt
```

## Running experiments

A single training run is launched through `src/main.py`:

```bash
python ./src/main.py --config_format base --opt lion_muon --dataset fineweb --model base
```

The batched scripts under `scripts/` reproduce the paper's experiments. They write checkpoints and logs to `./exps/<experiment_name>/`.

```bash
# All baselines on FineWeb (GPT-base + LLaMA)
bash scripts/train_fineweb.sh 0

# A single dataset/model combination
bash scripts/train_baselines.sh 0 --dataset fineweb --model base --algo-set regular

# Scaling runs
bash scripts/train_fineweb_355m.sh 0
bash scripts/train_fineweb_720m.sh 0

# Hyperparameter tuning sweeps
bash scripts/train_tuning.sh 0
bash scripts/train_tuning_720m.sh 0
bash scripts/train_tuning_llama.sh 0
```

Shared hyperparameters (architecture, batch size, schedule, weight decay) live in [scripts/common_config.sh](scripts/common_config.sh). Per-experiment learning rates are at the top of each `train_*.sh` script.

### Hyperparameter study (124M)

One protocol for every optimizer in the paper. Both learning rates live on the
same `1e-x / 3e-x` ladder: the grid sweeps `eta_M` (Muon step) and the ratio
`alpha = eta_M / eta_L`, so the sign learning rate is a ladder value too.
`P = 1` is pure Muon, `P = 10000000` the pure-sign end (Lion / Signum), and the
two beta settings select the family (`0.9:0.9` SignMuon, `0.9:0.99` LionMuon).
Every run is resume-safe: cells with a `summary.json` are skipped.

```bash
D=fineweb                                   # or wikitext
export GRID_ITERS=64000 GRID_WU=3000        # sweep at the full training budget

# 1. the family, from pure Muon to pure sign
PS="1 2 5 20 10000000" LRS="1e-3 3e-3" ALPHAS="10 30 100" \
  bash scripts/run_experiments.sh grid $D 124m

# 2. extend the grid until every selected value has worse neighbours on both sides
bash scripts/close_edges.sh $D 124m

# 3. AdamW, the one baseline outside the family
LRS="3e-4 1e-3 3e-3" bash scripts/run_experiments.sh base $D 124m

# 4. the tuned winners over three seeds (the grid cell is seed 0)
bash scripts/run_experiments.sh seeds $D 124m "1 2"

# 5. the figure
python scripts/plotting/plot_124m.py
```

`scripts/prune.sh <sweep log>` can be run alongside any sweep: it stops cells
that diverge or fall far behind the best cell of their own group, which halves
the cost of the grid.

`GSEEDS="1 2"` repeats any grid cell over seeds, which is how two cells that sit
within seed noise of each other are separated.

### Scaling and efficiency runs

355M transfers the tuned `(eta_M, alpha)` from 124M rather than re-tuning:

```bash
bash scripts/run_experiments.sh seeds fineweb 355m "0 1 2"   # BS/ACC override the 16x32 default
bash scripts/run_experiments.sh extra fineweb 124m           # Nesterov, Dion, MuonBP
python scripts/microbenchmark.py                             # per-step wall clock
torchrun --nproc_per_node=8 scripts/comms_bench.py           # optimizer all-reduce
```

On consumer GPUs without NVLink, export `NCCL_P2P_DISABLE=1` before any
distributed run. FineWeb defaults to the 10BT sample; `FINEWEB_SAMPLE=sample-100BT`
selects the full one (about 450GB).

Multi-GPU runs use `torchrun`:

```bash
torchrun --nproc_per_node=4 ./src/main.py --config_format base --distributed_backend nccl \
    --opt lion_muon --dataset fineweb --model base
```

### Key flags

| Flag | Meaning |
| --- | --- |
| `--opt {lion_muon, sign_muon}` | Select the alternating optimizer |
| `--lr` | AdamW backup LR for 1D parameters |
| `--muon_lr_factor` | Muon learning rate for 2D parameters |
| `--sign_lr` | Lion / sign learning rate for 2D parameters |
| `--muon_every_k K` | Period `P` (`K=1`: pure Muon; `K=inf`: pure Lion/Signum) |
| `--beta1`, `--beta2` | Set equal => SignMuon, unequal => LionMuon |
| `--muon_ns_steps` | Newton-Schulz iterations |
| `--weight_decay` | Decoupled weight decay |
| `--srank_alpha` | Adaptive Muon trigger via stable-rank ratio (overrides `--muon_every_k`) |
| `--nesterov` | Nesterov lookahead inside the Muon momentum EMA |
| `--opt {dion, muonbp}` | Efficiency baselines: Dion (arXiv:2504.05295), MuonBP (arXiv:2510.16981) |

## Directory layout

```
llm-baselines/
  src/
    main.py              # entry point: picks dataset, model, optimizer, training loop
    config/base.py       # all CLI flags
    data/                # dataset loaders (FineWeb, SlimPajama, WikiText, ...)
    models/              # base GPT and LLaMA architectures
    optim/
      lion_muon.py       # LionMuon  (this paper)
      sign_muon.py       # SignMuon  (this paper)
      muon.py            # Muon baseline + Newton-Schulz primitive
      lion.py            # Lion baseline
      ...                # AdamW, SOAP, Shampoo, Sophia, AdEMAMix, MARS, ...
    distributed/         # DDP wrappers
  scripts/               # bash scripts to reproduce paper experiments
  exps/                  # output checkpoints + logs (created on run)
  requirements.txt
```

## Logging with WandB

Pass `--wandb --wandb_project <name>` to log to your account. Set `WANDB_API_KEY` in the environment for headless runs.

## License

See [LICENSE](LICENSE). The original `llm-baselines` license is preserved.
