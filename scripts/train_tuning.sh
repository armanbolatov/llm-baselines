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
  $(basename "$0") [GPU_ID] [--dataset fw|fineweb|spj|slimpajama|wt|wikitext] [--model base|gpt|llama] [--max-jobs N] [--results-dir DIR] [--algo-set regular|all]

Examples:
  $(basename "$0") 0 --dataset fw --model base
  $(basename "$0") 0 --dataset fw --model llama
  $(basename "$0") --dataset spj --model llama --max-jobs 2
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
RESULTS_DIR_OVERRIDE=""
ALGO_SET="regular"

while [ $# -gt 0 ]; do
  case "$1" in
    --dataset)
      DATASET_RAW="$2"; shift 2 ;;
    --model)
      MODEL_RAW="$2"; shift 2 ;;
    --max-jobs)
      MAX_JOBS_OVERRIDE="$2"; shift 2 ;;
    --results-dir)
      RESULTS_DIR_OVERRIDE="$2"; shift 2 ;;
    --algo-set)
      ALGO_SET="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

case "$DATASET_RAW" in
  fw|fineweb)
    DATASET="fineweb"; DATASET_PREFIX="fw" ;;
  spj|slimpajama)
    DATASET="slimpajama"; DATASET_PREFIX="spj" ;;
  wt|wikitext|wikitext103|wikitext-103)
    DATASET="wikitext"; DATASET_PREFIX="wt" ;;
  *)
    echo "Unsupported dataset: $DATASET_RAW"
    usage
    exit 1
    ;;
esac

case "$MODEL_RAW" in
  base|gpt)
    MODEL="base"; MODEL_PREFIX="gpt" ;;
  llama)
    MODEL="llama"; MODEL_PREFIX="llama" ;;
  *)
    echo "Unsupported model: $MODEL_RAW"
    usage
    exit 1
    ;;
esac

RUN_REGULAR=0
case "$ALGO_SET" in
  regular)
    RUN_REGULAR=1 ;;
  all)
    RUN_REGULAR=1
    ;;
  *)
    echo "Unsupported algo-set: $ALGO_SET (supported: regular, all)"
    usage
    exit 1
    ;;
esac

DEVICE="cuda:${GPU_ID}"
MAX_JOBS_LOCAL=${MAX_JOBS_OVERRIDE:-2}
SM_LR=${SM_LR:-1e-3}

# Tuning is standardized to the completed llama schedule.
ITERATIONS=3000
WARMUP=300
EVAL_INTERVAL=500

if [ -n "$RESULTS_DIR_OVERRIDE" ]; then
  EXPS_DIR="$RESULTS_DIR_OVERRIDE"
elif [ "$DATASET" = "fineweb" ] && [ "$MODEL" = "base" ]; then
  EXPS_DIR=./exps_tuning_gpt
elif [ "$DATASET" = "fineweb" ] && [ "$MODEL" = "llama" ]; then
  EXPS_DIR=./exps_tuning_llama
else
  EXPS_DIR="./exps_tuning_${DATASET_PREFIX}_${MODEL_PREFIX}"
fi

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
  --grad_clip $GRAD_CLIP \
  --results_base_folder $EXPS_DIR \
  --tensorboard"

MUON_SHARED="--momentum $MUON_MOM --nesterov True --muon_ns_steps $MUON_NS_STEPS --weight_decay $WEIGHT_DECAY"
SIGNMUON_SHARED="--momentum $SM_MOM --nesterov True --muon_ns_steps $MUON_NS_STEPS --weight_decay $WEIGHT_DECAY"
LIONMUON_SHARED="--beta1 $LM_BETA1 --beta2 $LM_BETA2 --muon_ns_steps $MUON_NS_STEPS --weight_decay $WEIGHT_DECAY"

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
  OMP_NUM_THREADS=1 $PYTHON ./src/main.py $COMMON_ARGS --experiment_name "$name" "$@" &
  PIDS+=($!)
}

sweep_1d() {
  local exp_prefix=$1
  local values=$2
  local arg_name=$3
  shift 3
  local fixed=("$@")
  local v
  for v in $values; do
    run "${exp_prefix}_${arg_name}${v}" "${fixed[@]}" --"$arg_name" "$v"
  done
}

sweep_2d() {
  local exp_prefix=$1
  local values1=$2
  local name1=$3
  local values2=$4
  local name2=$5
  shift 5
  local fixed=("$@")
  local v1 v2
  for v1 in $values1; do
    for v2 in $values2; do
      run "${exp_prefix}_${name1}${v1}_${name2}${v2}" "${fixed[@]}" --"$name1" "$v1" --"$name2" "$v2"
    done
  done
}

if [ "$RUN_REGULAR" = "1" ]; then
  # Keep legacy run names so existing exps_tuning_* results are detected by [SKIP].
  for v in 1e-4 2e-4 5e-4 1e-3 2e-3 5e-3 7e-3 1e-2 2e-2 3e-2; do
    run "muon_lr${v}" --opt sign_muon --lr "$MUON_ADAMW_LR" --muon_every_k 1 --cheap_mode sign $MUON_SHARED --muon_lr_factor "$v"
  done

  for v in 1e-5 2e-5 5e-5 1e-4 2e-4 5e-4 1e-3 2e-3 5e-3; do
    run "signum_slr${v}" --opt sign_muon --lr "$SM_ADAMW_LR" --muon_lr_factor "$SM_LR" --muon_every_k 10000000 --cheap_mode sign --sign_scaling none $SIGNMUON_SHARED --sign_lr "$v"
  done

  for v in 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2; do
    run "muon_fixed_lr${v}" --opt lion_muon --lr "$SM_ADAMW_LR" --muon_every_k 1 --beta1 0.9 --beta2 0.9 --muon_lr_factor "$v"
  done

  for v in 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2; do
    run "signum_fixed_slr${v}" --opt lion_muon --lr "$SM_ADAMW_LR" --muon_lr_factor "$SM_LR" --muon_every_k 10000000 --beta1 0.9 --beta2 0.9 --sign_lr "$v"
  done

  for lr in 1e-3 2e-3 3e-3 5e-3 7e-3; do
    for slr in 1e-6 2e-6 5e-6 1e-5 2e-5 5e-5 1e-4; do
      run "signmuon_k2_lr${lr}_slr${slr}" --opt sign_muon --lr "$SM_ADAMW_LR" --muon_every_k 2 --cheap_mode sign --sign_scaling none $SIGNMUON_SHARED --muon_lr_factor "$lr" --sign_lr "$slr"
    done
  done

  for lr in 1e-3 2e-3 3e-3 5e-3 7e-3; do
    for slr in 5e-6 1e-5 2e-5 5e-5 1e-4; do
      run "signmuon_k5_lr${lr}_slr${slr}" --opt sign_muon --lr "$SM_ADAMW_LR" --muon_every_k 5 --cheap_mode sign --sign_scaling none $SIGNMUON_SHARED --muon_lr_factor "$lr" --sign_lr "$slr"
    done
  done

  for lr in 3e-3 5e-3 7e-3 1e-2 2e-2; do
    for slr in 5e-6 1e-5 2e-5 5e-5 1e-4; do
      run "signmuon_k20_lr${lr}_slr${slr}" --opt sign_muon --lr "$SM_ADAMW_LR" --muon_every_k 20 --cheap_mode sign --sign_scaling none $SIGNMUON_SHARED --muon_lr_factor "$lr" --sign_lr "$slr"
    done
  done

  for lr in 3e-3 5e-3 7e-3 1e-2 2e-2; do
    for slr in 5e-6 1e-5 2e-5 5e-5 1e-4; do
      run "signmuon_k100_lr${lr}_slr${slr}" --opt sign_muon --lr "$SM_ADAMW_LR" --muon_every_k 100 --cheap_mode sign --sign_scaling none $SIGNMUON_SHARED --muon_lr_factor "$lr" --sign_lr "$slr"
    done
  done

  # SignMuon-fixed: implemented through LiMuon dynamics with beta1=beta2=0.9
  for lr in 1e-3 2e-3 3e-3 5e-3 7e-3; do
    for slr in 1e-6 2e-6 5e-6 1e-5 2e-5 5e-5 1e-4; do
      run "signmuon_fixed_k2_lr${lr}_slr${slr}" --opt lion_muon --lr "$SM_ADAMW_LR" --muon_every_k 2 --beta1 0.9 --beta2 0.9 --muon_lr_factor "$lr" --sign_lr "$slr"
    done
  done

  for lr in 1e-3 2e-3 3e-3 5e-3 7e-3; do
    for slr in 5e-6 1e-5 2e-5 5e-5 1e-4; do
      run "signmuon_fixed_k5_lr${lr}_slr${slr}" --opt lion_muon --lr "$SM_ADAMW_LR" --muon_every_k 5 --beta1 0.9 --beta2 0.9 --muon_lr_factor "$lr" --sign_lr "$slr"
    done
  done

  for lr in 3e-3 5e-3 7e-3 1e-2 2e-2; do
    for slr in 5e-6 1e-5 2e-5 5e-5 1e-4; do
      run "signmuon_fixed_k20_lr${lr}_slr${slr}" --opt lion_muon --lr "$SM_ADAMW_LR" --muon_every_k 20 --beta1 0.9 --beta2 0.9 --muon_lr_factor "$lr" --sign_lr "$slr"
    done
  done

  for lr in 3e-3 5e-3 7e-3 1e-2 2e-2; do
    for slr in 5e-6 1e-5 2e-5 5e-5 1e-4; do
      run "signmuon_fixed_k100_lr${lr}_slr${slr}" --opt lion_muon --lr "$SM_ADAMW_LR" --muon_every_k 100 --beta1 0.9 --beta2 0.9 --muon_lr_factor "$lr" --sign_lr "$slr"
    done
  done

  for lr in 5e-4 7e-4 1e-3 2e-3 3e-3; do
    run "lionmuon_k1_lr${lr}" --opt lion_muon --lr "$LM_ADAMW_LR" --muon_every_k 1 $LIONMUON_SHARED --muon_lr_factor "$lr"
  done

  LM_K2_LRS="5e-4 7e-4 1e-3 2e-3 3e-3 5e-3 7e-3"

  for lr in $LM_K2_LRS; do
    for slr in 5e-6 1e-5 2e-5 5e-5 1e-4; do
      run "lionmuon_k2_lr${lr}_slr${slr}" --opt lion_muon --lr "$LM_ADAMW_LR" --muon_every_k 2 $LIONMUON_SHARED --muon_lr_factor "$lr" --sign_lr "$slr"
    done
  done

  for lr in 1e-3 2e-3 3e-3 5e-3 7e-3; do
    for slr in 5e-6 1e-5 2e-5 5e-5 1e-4; do
      run "lionmuon_k5_lr${lr}_slr${slr}" --opt lion_muon --lr "$LM_ADAMW_LR" --muon_every_k 5 $LIONMUON_SHARED --muon_lr_factor "$lr" --sign_lr "$slr"
    done
  done

  for lr in 3e-3 5e-3 7e-3 1e-2 2e-2; do
    for slr in 2e-5 5e-5 1e-4; do
      run "lionmuon_k20_lr${lr}_slr${slr}" --opt lion_muon --lr "$LM_ADAMW_LR" --muon_every_k 20 $LIONMUON_SHARED --muon_lr_factor "$lr" --sign_lr "$slr"
    done
  done

  for lr in 3e-3 5e-3 7e-3 1e-2 2e-2; do
    for slr in 2e-5 5e-5 1e-4; do
      run "lionmuon_k100_lr${lr}_slr${slr}" --opt lion_muon --lr "$LM_ADAMW_LR" --muon_every_k 100 $LIONMUON_SHARED --muon_lr_factor "$lr" --sign_lr "$slr"
    done
  done

  # --- Adaptive srank: lion_muon with srank_alpha, 3 LRs x 4 alphas ---
  # When srank_alpha > 0, muon_every_k is ignored; per-layer stable rank decides muon vs sign.
  # sign_lr sets the Lion step lr; muon_lr_factor sets the Muon step lr.
  LM_SRANK_MUON_LRS="1e-3 3e-3 5e-3"
  LM_SRANK_SIGN_LRS="1e-5 5e-5 1e-4"

  for ALPHA in 0.002 0.004 0.006 0.008 0.01; do
    sweep_2d "lionmuon_srank_a${ALPHA}" "$LM_SRANK_MUON_LRS" "muon_lr_factor" "$LM_SRANK_SIGN_LRS" "sign_lr" \
      --opt lion_muon --lr "$LM_ADAMW_LR" --srank_alpha "$ALPHA" $LIONMUON_SHARED
  done
fi

echo "Waiting for remaining jobs..."
wait
echo "Done!"
