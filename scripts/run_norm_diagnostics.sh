#!/bin/bash
# The constants of the bound, recorded during training: 10,000 steps of the 124M model (12 layers, width 768)
# with LionMuon, a record every 100 steps. One run per dataset and period.
#   bash scripts/run_norm_diagnostics.sh [gpu]                       # P=2, the runs of the constants table and figure
#   PERIODS="1 2 5 inf" bash scripts/run_norm_diagnostics.sh [gpu]   # the comparison across periods (inf is pure Lion)
set -e

export OMP_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON=${PYTHON:-python}

GPU_ID=${1:-0}
DEVICE="cuda:${GPU_ID}"
ITERATIONS=10000
WARMUP=500
PERIODS=${PERIODS:-2}

DATASETS_DIR=${DATASETS_DIR:-~/datasets/}

echo "Running norm diagnostics experiments on $DEVICE, periods: $PERIODS"

for DATASET in fineweb wikitext; do
  if [ "$DATASET" = fineweb ]; then TAG=fw; SIGN_LR="--sign_lr 5e-5"; else TAG=wt; SIGN_LR=""; fi
  for P in $PERIODS; do
    if [ "$P" = inf ]; then EVERY_K=10000000; else EVERY_K=$P; fi
    NAME=ndL12_${TAG}_p${P}_seed0
    echo
    echo "[RUN] $DATASET - LionMuon P=$P ($NAME)"
    $PYTHON ./src/main.py \
      --opt lion_muon \
      --dataset $DATASET \
      --model base \
      --n_layer 12 --n_head 12 --n_embd 768 \
      --device $DEVICE \
      --datasets_dir $DATASETS_DIR \
      --muon_every_k $EVERY_K \
      --muon_lr_factor 0.001 \
      $SIGN_LR \
      --lr 0.001 \
      --beta1 0.9 --beta2 0.99 \
      --batch_size 32 \
      --sequence_length 512 \
      --iterations $ITERATIONS \
      --warmup_steps $WARMUP \
      --scheduler wsd \
      --wsd_fract_decay 0 \
      --wsd_final_lr_scale 1 \
      --muon_ns_steps 5 \
      --weight_decay 0.1 \
      --grad_clip 0.5 \
      --eval_interval 200 \
      --seed 0 \
      --experiment_name $NAME \
      --norm_diag \
      --norm_diag_every_k 100
  done
done

echo
echo "Norm diagnostics experiments completed. Each run leaves norm_diag.json in its folder under exps/."
echo "The figure (the two P=2 runs) and the table:"
echo "  python scripts/plotting/plot_norm_diagnostics.py <fineweb P=2 norm_diag.json>:FineWeb <wikitext P=2 norm_diag.json>:WikiText-103"
echo "  python scripts/plotting/constants_table.py --ref P=1 P=1=<...p1...norm_diag.json> P=2=<...> P=5=<...> Lion=<...pinf...>"
echo "The records of the runs in the paper are in results/additional/constants_124m/."
