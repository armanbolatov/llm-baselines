#!/bin/bash
# Train WSM (arXiv:2507.17634) variants of `adamw` and `lionmuon_k2` on
# FineWeb / 124M GPT-base, mirroring the existing cos-scheduled baselines
# in exps/fw_base_adamw and exps/fw_base_lionmuon_k2.
#
# Only the LR schedule and the checkpointing interval differ from the
# baselines: --scheduler wsm replaces the cosine decay with a constant
# peak LR held to the end, and we save every 1600 iters so the last 4
# permanent checkpoints land inside the final 10% of training (the same
# window the cos/wsd schedulers would have decayed through).
#
# After training, the last 4 ckpts are merged three ways (mean / EMA /
# theorem-derived 1-sqrt weights) and all checkpoints are evaluated on
# the full validation set.

set -e
export OMP_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/common_config.sh"

PYTHON=${PYTHON:-python}
GPU_ID="${1:-0}"
DEVICE="cuda:${GPU_ID}"
DATASET="${DATASET:-fineweb}"
N_MERGE="${N_MERGE:-4}"
WSM_CKPT_INTERVAL=1600

# Hyperparameters copied verbatim from scripts/train_baselines.sh so the
# WSM runs are directly comparable to the existing cos-scheduled runs.
ADAMW_LR=5e-4; ADAMW_BETA1=0.8; ADAMW_BETA2=0.999
LM_K2_LR=1e-3; LM_K2_SLR=5e-5; LM_ADAMW_LR=1e-3
LM_BETA1=0.9;  LM_BETA2=0.99

COMMON_ARGS="--dataset $DATASET --datasets_dir $DATASETS_DIR \
  --model base --batch_size $BATCH_SIZE --acc_steps $ACC_STEPS \
  --iterations $ITERATIONS --warmup_steps $WARMUP \
  --eval_interval $EVAL_INTERVAL --sequence_length $SEQ_LEN \
  --n_layer $N_LAYER --n_head $N_HEAD --n_embd $N_EMBD \
  --device $DEVICE --weight_decay $WEIGHT_DECAY --grad_clip $GRAD_CLIP \
  --scheduler wsm --permanent_ckpt_interval $WSM_CKPT_INTERVAL \
  --results_base_folder ./exps --tensorboard"

run_one() {
  local name="$1"; shift
  local exp="fw_base_${name}_wsm"
  if [ -f "./exps/${exp}/summary.json" ]; then
    echo "[SKIP] ${exp}"; return 0
  fi
  echo "[RUN ] ${exp}"
  $PYTHON ./src/main.py $COMMON_ARGS --experiment_name "$exp" "$@"
}

# 1) Train the two WSM runs.
run_one "adamw" \
  --opt adamw --lr $ADAMW_LR --beta1 $ADAMW_BETA1 --beta2 $ADAMW_BETA2

run_one "lionmuon_k2" \
  --opt lion_muon --lr $LM_ADAMW_LR --muon_lr_factor $LM_K2_LR \
  --sign_lr $LM_K2_SLR --muon_every_k 2 --beta1 $LM_BETA1 --beta2 $LM_BETA2

# 2) Merge the last N permanent checkpoints three ways.
for NAME in adamw lionmuon_k2; do
  EXP="exps/fw_base_${NAME}_wsm"
  for METHOD in mean ema theorem; do
    $PYTHON ./src/merge_wsm.py --exp_dir "$EXP" --n "$N_MERGE" --method "$METHOD"
  done
done

# 3) Evaluate every WSM checkpoint (+ merged variants) on full val set.
EVAL_ARGS="--config_format base --dataset $DATASET --datasets_dir $DATASETS_DIR \
  --model base --batch_size $BATCH_SIZE --sequence_length $SEQ_LEN \
  --n_layer $N_LAYER --n_head $N_HEAD --n_embd $N_EMBD --device $DEVICE --eval_full"

CKPTS=()
for NAME in adamw lionmuon_k2; do
  EXP="exps/fw_base_${NAME}_wsm"
  CKPTS+=("$EXP/ckpts/latest/main.pt")
  CKPTS+=("$EXP/ckpts/merged_mean_n${N_MERGE}/main.pt")
  CKPTS+=("$EXP/ckpts/merged_ema0.5_n${N_MERGE}/main.pt")
  CKPTS+=("$EXP/ckpts/merged_theorem_n${N_MERGE}/main.pt")
done
EXISTING=(); for c in "${CKPTS[@]}"; do [ -f "$c" ] && EXISTING+=("$c"); done

$PYTHON ./src/eval_wsm.py $EVAL_ARGS --ckpts "${EXISTING[@]}" \
  --output "exps/wsm_fw_summary.json"
echo "Wrote exps/wsm_fw_summary.json"
echo "Plot with: python scripts/plotting/plot_wsm.py"
