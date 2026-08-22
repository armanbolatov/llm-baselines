#!/bin/bash
set -e

export OMP_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON=${PYTHON:-python}

# Fast tuning for 720M model on FineWeb
# Tune LR only for signmuon_fixed and lionmuon (k=1,2,5)

DATASETS_DIR=${DATASETS_DIR:-~/datasets/}
DATASET="fineweb"
MODEL="base"

GPU_ID=${1:-0}
DEVICE="cuda:${GPU_ID}"

BATCH_SIZE=64
ACC_STEPS=31
ITERATIONS=500
WARMUP=100
EVAL_INTERVAL=100

N_LAYER=12
N_HEAD=16
N_EMBD=2048
SEQ_LEN=512

EXPS_DIR="./exps_tuning_720m"
EXP_PREFIX="fw_base_720m_tune_"

run() {
  local name=$1; shift
  local exp_name="${EXP_PREFIX}${name}"

  if [ -f "${EXPS_DIR}/${exp_name}/summary.json" ]; then
    echo "[SKIP] $exp_name"
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
    --experiment_name "$exp_name" "$@"
}

LRS="1e-4 3e-4 1e-3 3e-3 1e-2 3e-2"
SIGN_LRS="1e-5 3e-5 1e-4 3e-4"

# # SignMuon-Fixed K=1 (muon_fixed) - LR sweep
echo "=== SignMuon Fixed K=1 (muon_fixed) ==="
for lr in $LRS; do
  run "signmuon_fixed_k1_lr${lr}" \
    --opt lion_muon --lr $lr --muon_lr_factor $lr \
    --muon_every_k 1 --beta1 0.9 --beta2 0.9
done

# # SignMuon-Fixed K=2 - LR and sign_lr sweep
echo "=== SignMuon Fixed K=2 ==="
for lr in $LRS; do
  for sign_lr in $SIGN_LRS; do
    run "signmuon_fixed_k2_lr${lr}_slr${sign_lr}" \
      --opt lion_muon --lr $lr --muon_lr_factor $lr \
      --sign_lr $sign_lr --muon_every_k 2 --beta1 0.9 --beta2 0.9
  done
done

# # SignMuon-Fixed K=5 - LR and sign_lr sweep
echo "=== SignMuon Fixed K=5 ==="
for lr in $LRS; do
  for sign_lr in $SIGN_LRS; do
    run "signmuon_fixed_k5_lr${lr}_slr${sign_lr}" \
      --opt lion_muon --lr $lr --muon_lr_factor $lr \
      --sign_lr $sign_lr --muon_every_k 5 --beta1 0.9 --beta2 0.9
  done
done

# SignMuon-Fixed K=infty (signum_fixed) - LR and sign_lr sweep
echo "=== SignMuon Fixed K=infty (signum_fixed) ==="
for lr in $LRS; do
  for sign_lr in $SIGN_LRS; do
    run "signum_fixed_lr${lr}_slr${sign_lr}" \
      --opt lion_muon --lr $lr --muon_lr_factor $lr \
      --sign_lr $sign_lr --muon_every_k 10000000 --beta1 0.9 --beta2 0.9
  done
done

# LionMuon K=1 - LR sweep
echo "=== LionMuon K=1 ==="
for lr in $LRS; do
  run "lionmuon_k1_lr${lr}" \
    --opt lion_muon --lr $lr --muon_lr_factor $lr \
    --muon_every_k 1 --beta1 0.9 --beta2 0.99
done

# LionMuon K=2 - LR and sign_lr sweep
echo "=== LionMuon K=2 ==="
for lr in $LRS; do
  for sign_lr in $SIGN_LRS; do
    run "lionmuon_k2_lr${lr}_slr${sign_lr}" \
      --opt lion_muon --lr $lr --muon_lr_factor $lr \
      --sign_lr $sign_lr --muon_every_k 2 --beta1 0.9 --beta2 0.99
  done
done

# LionMuon K=5 - LR and sign_lr sweep
echo "=== LionMuon K=5 ==="
for lr in $LRS; do
  for sign_lr in $SIGN_LRS; do
    run "lionmuon_k5_lr${lr}_slr${sign_lr}" \
      --opt lion_muon --lr $lr --muon_lr_factor $lr \
      --sign_lr $sign_lr --muon_every_k 5 --beta1 0.9 --beta2 0.99
  done
done

echo "Tuning script finished!"
