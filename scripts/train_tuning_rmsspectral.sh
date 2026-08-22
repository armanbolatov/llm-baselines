#!/bin/bash
set -e

export OMP_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

source "$SCRIPT_DIR/common_config.sh"

PYTHON=${PYTHON:-python}

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [GPU_ID] [--dataset fw|fineweb|spj|slimpajama] [--model base|gpt|llama] [--max-jobs N] [--results-dir DIR]

Examples:
  $(basename "$0") 0 --dataset fw --model base
  $(basename "$0") 0 --dataset fw --model llama
EOF
}

GPU_ID="0"
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
  GPU_ID="$1"
  shift
fi

DATASET_RAW="fw"
MODEL_RAW="base"
MAX_JOBS_OVERRIDE=""
RESULTS_DIR_OVERRIDE=""

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
    --results-dir)
      RESULTS_DIR_OVERRIDE="$2"
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
  *)
    echo "Unsupported dataset: $DATASET_RAW"
    usage
    exit 1
    ;;
esac

case "$MODEL_RAW" in
  base|gpt)
    MODEL="base"
    MODEL_PREFIX="gpt"
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

DEVICE="cuda:${GPU_ID}"
MAX_JOBS_LOCAL=${MAX_JOBS_OVERRIDE:-2}

if [ "$MODEL" = "base" ]; then
  ITERATIONS=2000
  WARMUP=200
  EVAL_INTERVAL=100
else
  ITERATIONS=3000
  WARMUP=300
  EVAL_INTERVAL=500
fi

if [ -n "$RESULTS_DIR_OVERRIDE" ]; then
  EXPS_DIR="$RESULTS_DIR_OVERRIDE"
else
  EXPS_DIR="./exps_tuning_rmsspectral_${DATASET_PREFIX}_${MODEL_PREFIX}"
fi

COMMON_ARGS="--dataset $DATASET \
  --datasets_dir $DATASETS_DIR \
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
  --grad_clip $GRAD_CLIP \
  --results_base_folder $EXPS_DIR \
  --tensorboard"

RMS_SHARED="--opt rmsspectral --muon_ns_steps $MUON_NS_STEPS --weight_decay $WEIGHT_DECAY"

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
  if [ -f "${EXPS_DIR}/${name}/summary.json" ]; then
    echo "[SKIP] $name"
    return
  fi
  reap
  while [ ${#PIDS[@]} -ge $MAX_JOBS_LOCAL ]; do
    sleep 1
    reap
  done
  echo "[RUN]  $name"
  OMP_NUM_THREADS=1 $PYTHON ./src/main.py $COMMON_ARGS \
    --experiment_name "$name" "$@" &
  PIDS+=($!)
}

# Tune all 4 RMSspectral methods from rmsspectral.py docs.
RMS_VARIANTS="post pre post_orth split"
RMS_LRS="5e-4 1e-3 2e-3 3e-3 5e-3"
RMS_BETA1=${RMS_BETA1:-0.9}
RMS_BETA2=${RMS_BETA2:-0.95}

for VARIANT in $RMS_VARIANTS; do
  for LR in $RMS_LRS; do
    run "rmsspectral_${VARIANT}_lr${LR}" \
      --lr $LR --beta1 $RMS_BETA1 --beta2 $RMS_BETA2 \
      --rmsspectral_variant $VARIANT \
      $RMS_SHARED
  done
done

echo "Waiting for remaining jobs..."
wait
echo "Done!"
