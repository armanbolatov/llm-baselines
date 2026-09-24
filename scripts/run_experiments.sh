#!/bin/bash
# Experiment driver. One GPU (DEV), resume-safe.
#   grid   <ds> <scale>            sweep LRS x ALPHAS x PS x BETAS
#   base   <ds> <scale>            AdamW over LRS
#   seeds  <ds> <scale> "<seeds>"  grid winners, repeated
#   extra  <ds> <scale>            Nesterov / Dion / MuonBP
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
DEV=${DEV:-cuda:0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FINEWEB_SAMPLE=${FINEWEB_SAMPLE:-sample-10BT}

MODE=${1:?mode: grid|seeds|extra}
DS=${2:?dataset: fineweb|wikitext}
SCALE=${3:-124m}
SEEDS=${4:-"0 1 2"}
TAG=$([ "$DS" = fineweb ] && echo fw || echo wt)

if [ "$SCALE" = 355m ]; then
  # 16x32 keeps the paper's effective batch 512 and fits a 32GB card
  ARCH="--batch_size ${BS:-16} --acc_steps ${ACC:-32} --sequence_length 1024 --n_layer 24 --n_head 16 --n_embd 1024"
  SCHED="--iterations 15650 --warmup_steps 1500"
elif [ "$SCALE" = 720m ]; then
  # effective batch 1984 as in train_fineweb_720m.sh; 1x Chinchilla (14.4B tokens)
  ARCH="--batch_size ${BS:-16} --acc_steps ${ACC:-124} --sequence_length 512 --n_layer 12 --n_head 16 --n_embd 2048"
  SCHED="--iterations 14000 --warmup_steps 700"
else
  ARCH="--batch_size ${BS:-32} --acc_steps ${ACC:-1} --sequence_length 512 --n_layer 12 --n_head 12 --n_embd 768"
  SCHED="--iterations 64000 --warmup_steps 3000"
fi
[ "$MODE" = grid ] && [ "$SCALE" = 124m ] && SCHED="--iterations ${GRID_ITERS:-20000} --warmup_steps ${GRID_WU:-1000}"

C="--config_format base --dataset $DS --datasets_dir ${DATASETS_DIR:-~/datasets/} --model base --device $DEV $ARCH $SCHED
  --scheduler cos --muon_ns_steps 5 --weight_decay 0.1 --grad_clip 0.5
  --eval_interval 500 --lr 1e-3 --latest_ckpt_interval ${CKPT_EVERY:-0}"

# name k muon_lr sign_lr b1 b2 seed [extra flags...]
run(){ local name=$1 k=$2 mlr=$3 slr=$4 b1=$5 b2=$6 seed=$7; shift 7
  local a=""; [ "$slr" != "-" ] && a="--sign_lr $slr"
  echo "[$(date +%m-%d\ %H:%M)] ${name}_seed${seed}"
  $PY src/main.py $C --opt lion_muon --muon_every_k $k --muon_lr_factor $mlr $a \
    --beta1 $b1 --beta2 $b2 --seed $seed --experiment_name "${name}_seed${seed}" "$@" \
    || echo "FAILED ${name}_seed${seed}"; }

P=${TAG}${SCALE%m}   # e.g. fw124
GB=${GRID_ITERS:-20000}   # budget goes in the name so sweeps never collide
SUF=""; [ "$MODE" = grid ] && [ "$GB" != 20000 ] && SUF="_i$((GB/1000))k"

case $MODE in
base)    # AdamW, the one baseline outside the family.
         # betas are the benchmark's tuned pair, which beats (0.9, 0.95) here.
  for s in $SEEDS; do for lr in ${LRS:?set LRS}; do
    echo "[$(date +%m-%d\ %H:%M)] ${P}b_adamw_lr${lr}_seed${s}"
    $PY src/main.py $C --opt adamw --lr $lr --beta1 0.8 --beta2 0.999 --seed $s \
      --experiment_name "${P}b_adamw_lr${lr}_seed${s}" || echo "FAILED adamw $lr"
  done; done ;;

grid)    # eta_L = eta_M / alpha. P=1 has no sign step, so alpha is skipped.
         # P=10000000 is the pure-sign end: Lion / Signum.
  LRS=${LRS:-"3e-4 1e-3 3e-3 1e-2"}
  ALPHAS=${ALPHAS:-"3 10 30 100 300"}
  PS=${PS:-"2"}
  BETAS=${BETAS:-"0.9:0.9 0.9:0.99"}
  for sd in ${GSEEDS:-0}; do
  for p in $PS; do for lr in $LRS; do for b in $BETAS; do
    b1=${b%%:*}; b2=${b##*:}; bt=$(echo "$b" | tr -d '.' | tr ':' '-')
    if [ "$p" = 1 ]; then
      run ${P}g_p1_lr${lr}_b${bt}${SUF} 1 $lr - $b1 $b2 $sd
    else
      for al in $ALPHAS; do
        sl=$(awk -v m=$lr -v a=$al 'BEGIN{printf "%.2e", m/a}')
        run ${P}g_p${p}_lr${lr}_a${al}_b${bt}${SUF} $p $lr $sl $b1 $b2 $sd
      done
    fi
  done; done; done; done ;;

seeds)   # columns: P eta_M alpha beta1 beta2 label
  case $DS in
  fineweb) W="1  1e-3 -   0.9 0.9  signmuon
1  1e-3 -   0.9 0.99 lionmuon
2  3e-3 30  0.9 0.9  signmuon
2  1e-3 10  0.9 0.99 lionmuon
5  1e-2 300 0.9 0.9  signmuon
5  3e-3 30  0.9 0.99 lionmuon
20 1e-2 100 0.9 0.9  signmuon
20 1e-2 100 0.9 0.99 lionmuon
10000000 3e-4 1   0.9 0.9  signmuon
10000000 3e-4 1   0.9 0.99 lionmuon" ;;
  wikitext) W="1  3e-3 -   0.9 0.9  signmuon
1  3e-3 -   0.9 0.99 lionmuon
2  3e-3 3   0.9 0.9  signmuon
2  3e-3 10  0.9 0.99 lionmuon
5  1e-2 30  0.9 0.9  signmuon
5  1e-2 30  0.9 0.99 lionmuon
20 3e-2 100 0.9 0.9  signmuon
20 3e-2 100 0.9 0.99 lionmuon" ;;
  *) echo "no tuned winners recorded for $DS yet"; exit 1 ;;
  esac
  # the grid cell is seed 0, so run "1 2" here
  for s in $SEEDS; do
    echo "$W" | while read p mlr al b1 b2 lbl; do
      [ -n "${PS:-}" ] && ! echo " $PS " | grep -q " $p " && continue
      [ -n "${LBLS:-}" ] && ! echo " $LBLS " | grep -q " $lbl " && continue
      sl=-; [ "$al" != "-" ] && sl=$(awk -v m=$mlr -v a=$al 'BEGIN{printf "%.2e", m/a}')
      run ${P}t_${lbl}_p${p} $p $mlr $sl $b1 $b2 $s
    done
  done ;;
ablate)  # two reviewer questions at the tuned FineWeb rates: a random period with the same
         # mean as the fixed one, and Lion instead of AdamW on the 1D parameters
  for s in $SEEDS; do
    run ${P}x_lionmuon_p2_randP  2 1e-3 1.00e-04 0.9 0.99 $s --muon_random_period
    run ${P}x_lionmuon_p5_randP  5 3e-3 1.00e-04 0.9 0.99 $s --muon_random_period
    run ${P}x_lionmuon_p2_lion1d 2 1e-3 1.00e-04 0.9 0.99 $s --one_d_lion
    run ${P}x_lionmuon_p5_lion1d 5 3e-3 1.00e-04 0.9 0.99 $s --one_d_lion
  done ;;
extra)
  run ${P}_muon_nesterov 1 1e-3 - 0.9 0.9 0 --nesterov True
  echo "[$(date +%H:%M)] dion"
  $PY src/main.py $C --opt dion --muon_lr_factor 1e-3 --dion_rank_frac 0.25 \
    --seed 0 --experiment_name ${P}_dion_seed0 || echo "FAILED dion"
  echo "[$(date +%H:%M)] muonbp"
  $PY src/main.py $C --opt muonbp --muon_lr_factor 1e-3 --muonbp_blocks 4 \
    --muonbp_period 5 --muonbp_block_lr_ratio 1.0 \
    --seed 0 --experiment_name ${P}_muonbp_seed0 || echo "FAILED muonbp" ;;
esac
echo "[$(date +%m-%d\ %H:%M)] $MODE $DS $SCALE done"
