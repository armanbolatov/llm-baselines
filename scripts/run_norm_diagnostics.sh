#!/bin/bash
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

DATASETS_DIR=${DATASETS_DIR:-~/datasets/}

echo "Running norm diagnostics experiments on $DEVICE"
echo

# FineWeb - LionMuon K=2 - FAST
echo "[RUN] FineWeb - LionMuon K=2"
$PYTHON ./src/main.py \
  --opt lion_muon \
  --dataset fineweb \
  --model base \
  --device $DEVICE \
  --datasets_dir $DATASETS_DIR \
  --muon_every_k 2 \
  --muon_lr_factor 0.001 \
  --sign_lr 5e-5 \
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
  --experiment_name fw_base_lionmuon_k2_normdiag_fast \
  --norm_diag \
  --norm_diag_every_k 100

echo
echo "[RUN] WikiText-103 - LionMuon K=2"
$PYTHON ./src/main.py \
  --opt lion_muon \
  --dataset wikitext \
  --model base \
  --device $DEVICE \
  --datasets_dir $DATASETS_DIR \
  --muon_every_k 2 \
  --muon_lr_factor 0.001 \
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
  --experiment_name wt_base_lionmuon_k2_normdiag_fast \
  --norm_diag \
  --norm_diag_every_k 100

echo
echo "Norm diagnostics experiments completed!"
echo "Plots can be generated with:"
echo "  python scripts/plotting/plot_norm_diagnostics.py \\"
echo "    exps/fw_base_lionmuon_k2_normdiag_fast/norm_diag.json:FineWeb \\"
echo "    exps/wt_base_lionmuon_k2_normdiag_fast/norm_diag.json:WikiText-103 \\"
echo "    --out results/norm_diag_fast.png \\"
echo "    --smooth_window 3"
