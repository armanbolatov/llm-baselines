#!/bin/bash
# 124M at 1x Chinchilla under real 4-GPU DDP, all methods round-robin on the same GPUs
# so that shared-node contention averages out. Each slice runs one method for SLICE
# seconds, then the next; runs resume from ckpts/latest (auto_resume) until done.
#
#   bash scripts/ddp_train_124m.sh            # loop until every method has summary.json
#   SLICE=7200 METHODS="adamw muon lionmuon_p5" bash scripts/ddp_train_124m.sh
set -u
cd "$(dirname "$0")/.."
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export FINEWEB_SAMPLE=${FINEWEB_SAMPLE:-sample-100BT}
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-python}
NGPU=${NGPU:-4}
ITERS=${ITERS:-150000}          # 150k x 32 x 512 = 2.46B tokens = 20 TPP
WARMUP=${WARMUP:-7500}
SLICE=${SLICE:-7200}            # seconds per method per round
OUT=${OUT:-exps_ddp124}
SEED=${SEED:-0}
METHODS=${METHODS:-"adamw muon lionmuon_p5 lionmuon_p2"}
LOGDIR=${LOGDIR:-logs_ddp124}        # per-method logs (round-robin appends)
mkdir -p "$OUT" "$LOGDIR"

C="--config_format base --dataset fineweb --datasets_dir ${DATASETS_DIR:-/home/shared/datasets} --model base
 --distributed_backend nccl --batch_size 32 --acc_steps 1 --sequence_length 512
 --n_layer 12 --n_head 12 --n_embd 768 --iterations $ITERS --warmup_steps $WARMUP --scheduler cos
 --muon_ns_steps 5 --weight_decay 0.1 --grad_clip 0.5 --eval_interval 1000 --eval_batches 32
 --lr 1e-3 --seed $SEED --log_interval 50 --latest_ckpt_interval ${CKPT:-250} --results_base_folder $OUT"

flags() { case $1 in
  adamw)       echo "--opt adamw --beta1 0.8 --beta2 0.999";;
  muon)        echo "--opt lion_muon --muon_every_k 1 --muon_lr_factor 1e-3 --beta1 0.9 --beta2 0.9";;
  muon_*)      echo "--opt lion_muon --muon_every_k 1 --muon_lr_factor ${1#muon_} --beta1 0.9 --beta2 0.9";;   # LR check at this horizon
  lionmuon_p2) echo "--opt lion_muon --muon_every_k 2 --muon_lr_factor 1e-3 --sign_lr 1e-4 --beta1 0.9 --beta2 0.99";;
  lionmuon_p5) echo "--opt lion_muon --muon_every_k 5 --muon_lr_factor 3e-3 --sign_lr 1e-4 --beta1 0.9 --beta2 0.99";;
  signmuon_p2) echo "--opt lion_muon --muon_every_k 2 --muon_lr_factor 3e-3 --sign_lr 1e-4 --beta1 0.9 --beta2 0.9";;
  # Baselines from the papers, untuned at this horizon, so a small LR sweep around Muon's optimum.
  # Dion: rank r = min(m,n)/4, error feedback 0.05 (mu = 0.95), local momentum + factor all-reduce.
  # MuonBP: 4 column blocks (= 4-way TP), full step every 5, block updates RMS-scaled by block dims.
  dion_*)      echo "--opt dion --muon_lr_factor ${1#dion_} --dion_rank_frac 0.25";;
  muonbp_*)    echo "--opt muonbp --muon_lr_factor ${1#muonbp_} --muonbp_blocks 4 --muonbp_period 5 --muonbp_block_lr_ratio 1.0 --momentum 0.9";;
  *) echo "unknown method $1" >&2; exit 1;; esac; }

for round in $(seq 1 200); do
  left=0
  for m in $METHODS; do
    name="ddp124_${m}_seed${SEED}"
    [ -f "$OUT/$name/summary.json" ] && continue
    left=$((left+1))
    echo "[$(date -u +%m-%d\ %H:%M)] round $round: $m for ${SLICE}s"
    t0=$(date +%s)
    timeout -s TERM -k 60 "$SLICE" $PY -m torch.distributed.run --nproc_per_node="$NGPU" --master_port="${PORT:-29517}" \
      src/main.py $C --experiment_name "$name" $(flags "$m") >> "$LOGDIR/$m.log" 2>&1
    # a slice that died within 2 min (OOM on a shared GPU, port clash) -> back off before the next method
    [ $(( $(date +%s) - t0 )) -lt 120 ] && sleep 300
    sleep 10
  done
  [ "$left" -eq 0 ] && { echo "[$(date -u +%m-%d\ %H:%M)] all methods finished"; break; }
done
