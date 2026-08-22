#!/bin/bash
set -e

export OMP_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

source "$SCRIPT_DIR/common_config.sh"

PYTHON=${PYTHON:-python}

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") [GPU_ID] [--dataset fw|fineweb|spj|slimpajama|wt|wikitext|wikitext103] [--model base|gpt|llama] [--max-jobs N] [--algo-set regular|all]

Examples:
  $(basename "$0") 0 --dataset fw --model base
  $(basename "$0") 1 --dataset spj --model llama
  $(basename "$0") --dataset fineweb --model llama
USAGE
}

GPU_ID="0"
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
  GPU_ID="$1"
  shift
fi

DATASET_RAW="fw"
MODEL_RAW="base"
MAX_JOBS_OVERRIDE=""
ALGO_SET="regular"

while [ $# -gt 0 ]; do
  case "$1" in
    --dataset)
      DATASET_RAW="$2"
      shift 2
      ;;
    --model)
      MODEL_RAW="$2"
      shift 2
      ;;
    --max-jobs)
      MAX_JOBS_OVERRIDE="$2"
      shift 2
      ;;
    --algo-set)
      ALGO_SET="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

case "$DATASET_RAW" in
  fw|fineweb)
    DATASET="fineweb"
    DATASET_PREFIX="fw"
    ;;
  spj|slimpajama)
    DATASET="slimpajama"
    DATASET_PREFIX="spj"
    ;;
  wt|wikitext|wikitext103|wikitext-103)
    DATASET="wikitext"
    DATASET_PREFIX="wt"
    ;;
  *)
    echo "Unsupported dataset: $DATASET_RAW"
    usage
    exit 1
    ;;
esac

case "$MODEL_RAW" in
  base|gpt)
    MODEL="base"
    MODEL_PREFIX="base"
    ;;
  llama)
    MODEL="llama"
    MODEL_PREFIX="llama"
    ;;
  *)
    echo "Unsupported model: $MODEL_RAW"
    usage
    exit 1
    ;;
esac

case "$ALGO_SET" in
  regular|all)
    ;;
  *)
    echo "Unsupported algo-set: $ALGO_SET (supported: regular, all)"
    usage
    exit 1
    ;;
esac

DEVICE="cuda:${GPU_ID}"
ITERATIONS=${ITERATIONS:-64000}
WARMUP=${WARMUP:-3000}
EVAL_INTERVAL=${EVAL_INTERVAL:-500}
MAX_JOBS_LOCAL=${MAX_JOBS_OVERRIDE:-${MAX_JOBS:-1}}

EXPS_DIR=./exps
EXP_PREFIX="${DATASET_PREFIX}_${MODEL_PREFIX}_"

ADAMW_LR=5e-4
ADAMW_BETA1=0.8
ADAMW_BETA2=0.999
SIGNUM_LR=5e-5

MUON_LR=1e-3
SM_K2_LR=2e-3;   SM_K2_SLR=2e-5
SM_K5_LR=3e-3;   SM_K5_SLR=2e-5
SM_K20_LR=1e-2;  SM_K20_SLR=5e-5
SM_K100_LR=1e-2; SM_K100_SLR=5e-5

# SignMuon-fixed (LiMuon dynamics with beta1=beta2=0.9) tuned defaults.
SMF_K2_LR=2e-3;   SMF_K2_SLR=5e-5
SMF_K5_LR=3e-3;   SMF_K5_SLR=5e-5
SMF_K20_LR=1e-2;  SMF_K20_SLR=5e-5
SMF_K100_LR=1e-2; SMF_K100_SLR=1e-4
SMF_MUON_LR=1e-3
SMF_SIGNUM_SLR=1e-4

LMK1_LR=7e-4
LM_K2_LR=1e-3;   LM_K2_SLR=5e-5
LM_K5_LR=2e-3;   LM_K5_SLR=5e-5
LM_K20_LR=5e-3;  LM_K20_SLR=5e-5
LM_K100_LR=7e-3; LM_K100_SLR=5e-5

DATASET_DIR_ARG="--datasets_dir $DATASETS_DIR"

COMMON_ARGS="--dataset $DATASET \
  $DATASET_DIR_ARG \
  --model $MODEL \
  --batch_size $BATCH_SIZE \
  --acc_steps $ACC_STEPS \
  --iterations $ITERATIONS \
  --eval_interval $EVAL_INTERVAL \
  --sequence_length $SEQ_LEN \
  --n_layer $N_LAYER \
  --n_head $N_HEAD \
  --n_embd $N_EMBD \
  --device $DEVICE \
  --scheduler $SCHEDULER \
  --warmup_steps $WARMUP \
  --weight_decay $WEIGHT_DECAY \
  --grad_clip $GRAD_CLIP \
  --muon_ns_steps $MUON_NS_STEPS \
  --results_base_folder $EXPS_DIR \
  --tensorboard"

PIDS=()

cleanup() {
  echo -e "\nCaught interrupt, killing all jobs..."
  kill "${PIDS[@]}" 2>/dev/null
  wait 2>/dev/null
  exit 1
}
trap cleanup INT TERM

reap() {
  local alive=()
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      alive+=("$pid")
    fi
  done
  PIDS=("${alive[@]}")
}

run() {
  local name=$1; shift
  local exp_name="${EXP_PREFIX}${name}"
  if [ -f "${EXPS_DIR}/${exp_name}/summary.json" ]; then
    echo "[SKIP] $exp_name"
    return
  fi
  reap
  while [ ${#PIDS[@]} -ge $MAX_JOBS_LOCAL ]; do
    sleep 1
    reap
  done
  echo "[RUN]  $exp_name"
  OMP_NUM_THREADS=1 $PYTHON ./src/main.py $COMMON_ARGS --experiment_name "$exp_name" "$@" &
  PIDS+=($!)
}

run "muon" \
  --opt sign_muon --lr $MUON_ADAMW_LR --muon_lr_factor $MUON_LR \
  --muon_every_k 1 --cheap_mode sign --momentum $MUON_MOM --nesterov True

run "adamw" \
  --opt adamw --lr $ADAMW_LR --beta1 $ADAMW_BETA1 --beta2 $ADAMW_BETA2

for K in 2 5 20 100; do
  eval MUON_K_LR=\$SM_K${K}_LR
  eval MUON_K_SLR=\$SM_K${K}_SLR
  run "signmuon_k${K}" \
    --opt sign_muon --lr $SM_ADAMW_LR --muon_lr_factor $MUON_K_LR \
    --sign_lr $MUON_K_SLR --muon_every_k $K --cheap_mode sign --sign_scaling none \
    --momentum $SM_MOM --nesterov True
done

# SignMuon-fixed variants implemented with LiMuon dynamics
# (beta1=beta2=0.9), using dedicated tuned SMF LRs.
for K in 2 5 20 100; do
  eval MUON_K_LR=\$SMF_K${K}_LR
  eval MUON_K_SLR=\$SMF_K${K}_SLR
  run "signmuon_fixed_k${K}" \
    --opt lion_muon --lr $SM_ADAMW_LR --muon_lr_factor $MUON_K_LR \
    --sign_lr $MUON_K_SLR --muon_every_k $K --beta1 0.9 --beta2 0.9
done

run "muon_fixed" \
  --opt lion_muon --lr $SM_ADAMW_LR --muon_lr_factor $SMF_MUON_LR \
  --muon_every_k 1 --beta1 0.9 --beta2 0.9

run "signum_fixed" \
  --opt lion_muon --lr $SM_ADAMW_LR --muon_lr_factor $SMF_MUON_LR --sign_lr $SMF_SIGNUM_SLR \
  --muon_every_k 10000000 --beta1 0.9 --beta2 0.9

run "signum" \
  --opt sign_muon --lr $SM_ADAMW_LR --sign_lr $SIGNUM_LR \
  --muon_every_k 10000000 --cheap_mode sign --sign_scaling none \
  --momentum $SM_MOM --nesterov True

run "lionmuon_k1" \
  --opt lion_muon --lr $LM_ADAMW_LR --muon_lr_factor $LMK1_LR \
  --muon_every_k 1 --beta1 $LM_BETA1 --beta2 $LM_BETA2

for K in 2 5 20 100; do
  eval LION_K_LR=\$LM_K${K}_LR
  eval LION_K_SLR=\$LM_K${K}_SLR
  run "lionmuon_k${K}" \
    --opt lion_muon --lr $LM_ADAMW_LR --muon_lr_factor $LION_K_LR \
    --sign_lr $LION_K_SLR --muon_every_k $K --beta1 $LM_BETA1 --beta2 $LM_BETA2
done

run "lion" \
  --opt lion_muon --lr $LM_ADAMW_LR --muon_lr_factor $LM_K100_LR \
  --sign_lr $LM_K100_SLR --muon_every_k 10000000 --beta1 $LM_BETA1 --beta2 $LM_BETA2

# Adaptive srank: best k5 LRs, sweep alpha
LM_SRANK_LR=1e-3
LM_SRANK_LR_0p002=3e-3
LM_SRANK_SLR=1e-4

run "lionmuon_srank_a0.002" \
  --opt lion_muon --lr $LM_ADAMW_LR --muon_lr_factor $LM_SRANK_LR_0p002 \
  --sign_lr $LM_SRANK_SLR --srank_alpha 0.002 \
  --beta1 $LM_BETA1 --beta2 $LM_BETA2

for ALPHA in 0.004 0.006 0.008 0.01; do
  run "lionmuon_srank_a${ALPHA}" \
    --opt lion_muon --lr $LM_ADAMW_LR --muon_lr_factor $LM_SRANK_LR \
    --sign_lr $LM_SRANK_SLR --srank_alpha $ALPHA \
    --beta1 $LM_BETA1 --beta2 $LM_BETA2

done

echo "Waiting for remaining jobs..."
wait
echo "Done!"
