#!/bin/bash
set -e

export OMP_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON=${PYTHON:-python}

# Setup for 720M model on FineWeb
# Architecture: n_layer=12, n_embd=2048, n_head=16, seq_len=512
# Param count: ~709M; trained at ~20 tokens/param ~= 14.4B tokens
DATASETS_DIR=${DATASETS_DIR:-~/datasets/}
DATASET="fineweb"
MODEL="base" # GPT architecture

GPU_ID=${1:-0}
DEVICE="cuda:${GPU_ID}"

BATCH_SIZE=64
ACC_STEPS=31
ITERATIONS=14000
WARMUP=100
EVAL_INTERVAL=250

N_LAYER=12
N_HEAD=16
N_EMBD=2048
SEQ_LEN=512

EXPS_DIR="./exps_720m"
EXP_PREFIX="fw_base_720m_"

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
    --grad_clip 0.1 \
    --muon_ns_steps 5 \
    --results_base_folder $EXPS_DIR \
    --tensorboard \
    --experiment_name "$exp_name" "$@"
}

# --- Algorithms ---

run "adamw" \
  --opt adamw --lr 0.001 --beta1 0.9 --beta2 0.999

run "muon_fixed" \
  --opt lion_muon --lr 3e-3 --muon_lr_factor 3e-3 \
  --muon_every_k 1 --beta1 0.9 --beta2 0.9

run "signum_fixed" \
  --opt lion_muon --lr 3e-3 --muon_lr_factor 3e-3 --sign_lr 3e-5 \
  --muon_every_k 10000000 --beta1 0.9 --beta2 0.9

run "lion" \
  --opt lion_muon --lr 3.75e-4 --muon_lr_factor 2.625e-3 \
  --sign_lr 1.875e-5 --muon_every_k 10000000 --beta1 0.9 --beta2 0.99

run "lionmuon_k1" \
  --opt lion_muon --lr 3e-3 --muon_lr_factor 3e-3 \
  --muon_every_k 1 --beta1 0.9 --beta2 0.99

run "lionmuon_k2" \
  --opt lion_muon --lr 3e-3 --muon_lr_factor 3e-3 \
  --sign_lr 1e-5 --muon_every_k 2 --beta1 0.9 --beta2 0.99

run "lionmuon_k5" \
  --opt lion_muon --lr 3e-3 --muon_lr_factor 3e-3 \
  --sign_lr 3e-5 --muon_every_k 5 --beta1 0.9 --beta2 0.99

run "signmuon_fixed_k2" \
  --opt lion_muon --lr 3e-3 --muon_lr_factor 3e-3 \
  --sign_lr 3e-5 --muon_every_k 2 --beta1 0.9 --beta2 0.9

run "signmuon_fixed_k5" \
  --opt lion_muon --lr 1e-2 --muon_lr_factor 1e-2 \
  --sign_lr 1e-4 --muon_every_k 5 --beta1 0.9 --beta2 0.9

echo "Script finished!"
