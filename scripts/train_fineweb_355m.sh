#!/bin/bash
set -e

export OMP_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON=${PYTHON:-python}

# Setup for 355M model on FineWeb
# ~8.2B tokens, Sequence Length 1024
DATASETS_DIR=${DATASETS_DIR:-~/datasets/}
DATASET="fineweb"
MODEL="base" # GPT architecture

GPU_ID=${1:-0}
DEVICE="cuda:${GPU_ID}"

BATCH_SIZE=64
ACC_STEPS=8
ITERATIONS=15650
WARMUP=1500
EVAL_INTERVAL=500

N_LAYER=24
N_HEAD=16
N_EMBD=1024
SEQ_LEN=1024

EXPS_DIR="./exps_355m"
EXP_PREFIX="fw_base_355m_"

run() {
  local name=$1; shift
  local exp_name="${EXP_PREFIX}${name}"

  if [ -f "${EXPS_DIR}/${exp_name}/summary.json" ]; then
    echo "[SKIP] $exp_name already completed."
    return
  fi

  echo "[RUN] $exp_name"
  $PYTHON ./src/main.py \
    --dataset $DATASET \
    --datasets_dir $DATASETS_DIR \
    --model $MODEL \
    --device $DEVICE \
    --batch_size $BATCH_SIZE \
    --acc_steps $ACC_STEPS \
    --iterations $ITERATIONS \
    --eval_interval $EVAL_INTERVAL \
    --sequence_length $SEQ_LEN \
    --n_layer $N_LAYER \
    --n_head $N_HEAD \
    --n_embd $N_EMBD \
    --scheduler cos \
    --warmup_steps $WARMUP \
    --weight_decay 0.1 \
    --grad_clip 0.5 \
    --muon_ns_steps 5 \
    --results_base_folder $EXPS_DIR \
    --tensorboard \
    --experiment_name "$exp_name" "$@"
}

# --- Algorithms ---
# LRs scaled down by 0.75 from 124M based on muP heuristic (768/1024)

run "adamw" \
  --opt adamw --lr 0.005 --beta1 0.8 --beta2 0.999

run "muon" \
  --opt sign_muon --lr 7.5e-4 --muon_lr_factor 7.5e-4 \
  --muon_every_k 1 --cheap_mode sign --momentum 0.95 --nesterov True

run "signum" \
  --opt sign_muon --lr 7.5e-4 --sign_lr 3.75e-5 \
  --muon_every_k 10000000 --cheap_mode sign --sign_scaling none \
  --momentum 0.95 --nesterov True

run "lion" \
  --opt lion_muon --lr 7.5e-4 --muon_lr_factor 5.25e-3 \
  --sign_lr 3.75e-5 --muon_every_k 10000000 --beta1 0.9 --beta2 0.99

run "lionmuon_k1" \
  --opt lion_muon --lr 7.5e-4 --muon_lr_factor 5.25e-4 \
  --muon_every_k 1 --beta1 0.9 --beta2 0.99

run "lionmuon_k2" \
  --opt lion_muon --lr 7.5e-4 --muon_lr_factor 7.5e-4 \
  --sign_lr 3.75e-5 --muon_every_k 2 --beta1 0.9 --beta2 0.99

run "lionmuon_k5" \
  --opt lion_muon --lr 7.5e-4 --muon_lr_factor 1.5e-3 \
  --sign_lr 3.75e-5 --muon_every_k 5 --beta1 0.9 --beta2 0.99

run "signmuon_fixed_k2" \
  --opt lion_muon --lr 7.5e-4 --muon_lr_factor 1.5e-3 \
  --sign_lr 3.75e-5 --muon_every_k 2 --beta1 0.9 --beta2 0.9

run "signmuon_fixed_k5" \
  --opt lion_muon --lr 7.5e-4 --muon_lr_factor 2.25e-3 \
  --sign_lr 3.75e-5 --muon_every_k 5 --beta1 0.9 --beta2 0.9

echo "Script finished!"
