#!/bin/bash
# Wall-clock per training step under real multi-GPU DDP at the 124M config.
# Runs each optimizer for a short horizon with torchrun and records the
# per-iteration time printed by the training loop. Losses are not the point
# here (use the single-GPU runs for that); this measures compute + exposed
# communication per step for AdamW / Muon / LionMuon / MuonBP / Dion.
#
#   NGPU=4 EBS=32  bash scripts/ddp_bench.sh   # paper batch: 32 seqs split 4x8
#   NGPU=4 EBS=128 bash scripts/ddp_bench.sh   # 32 seqs per GPU
#
# Requires: export NCCL_P2P_DISABLE=1 on boxes without NVLink.
set -u
cd "$(dirname "$0")/.."
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export FINEWEB_SAMPLE=${FINEWEB_SAMPLE:-sample-100BT}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-python}
NGPU=${NGPU:-4}
ITERS=${ITERS:-400}
EBS=${EBS:-32}
OUT=${OUT:-results/benchmarks/ddp_bench}
mkdir -p "$OUT"

C="--config_format base --dataset fineweb --datasets_dir ${DATASETS_DIR:-/home/shared/datasets} --model base
 --distributed_backend nccl --batch_size $EBS --acc_steps 1 --sequence_length 512
 --n_layer 12 --n_head 12 --n_embd 768 --iterations $ITERS --warmup_steps 100 --scheduler cos
 --muon_ns_steps 5 --weight_decay 0.1 --grad_clip 0.5 --eval_interval 100000 --eval_batches 4
 --lr 1e-3 --seed 0 --log_interval 10 --results_base_folder $OUT/exps"

run() { local name=$1; shift
  local tag="${name}_g${NGPU}_ebs${EBS}"
  echo "[$(date +%H:%M:%S)] $tag"
  rm -rf "$OUT/exps/$tag"
  $PY -m torch.distributed.run --nproc_per_node="$NGPU" --master_port="${PORT:-29511}" \
    src/main.py $C --experiment_name "$tag" "$@" > "$OUT/$tag.log" 2>&1 \
    || echo "FAILED $tag (see $OUT/$tag.log)"
  # median synchronized wall-clock per iteration after warmup (iterations >= 100):
  # (wall_i - wall_{i-1}) / log_interval between consecutive log lines.
  grep -oE "Iter=[0-9]+ .*wall=[0-9.]+" "$OUT/$tag.log" \
    | awk -F'[= ]' '{for(i=1;i<=NF;i++){if($i=="Iter")it=$(i+1); if($i=="wall")w=$(i+1)}; if(pit!="" && it+0>=100) print (w-pw)/(it-pit); pit=it; pw=w}' \
    | sort -g | awk '{a[NR]=$1} END{if(NR) printf "  median s/iter = %.4f over %d samples\n", a[int((NR+1)/2)], NR; else print "  no samples"}'
}

for rep in $(seq 1 "${REPS:-1}"); do
  run adamw        --opt adamw --beta1 0.8 --beta2 0.999
  run muon         --opt lion_muon --muon_every_k 1 --muon_lr_factor 1e-3 --beta1 0.9 --beta2 0.9
  run lionmuon_p2  --opt lion_muon --muon_every_k 2 --muon_lr_factor 1e-3 --sign_lr 1e-4 --beta1 0.9 --beta2 0.99
  run lionmuon_p5  --opt lion_muon --muon_every_k 5 --muon_lr_factor 1e-3 --sign_lr 1e-4 --beta1 0.9 --beta2 0.99
  run muonbp_p5    --opt muonbp --muon_lr_factor 1e-3 --muonbp_blocks 4 --muonbp_period 5 --muonbp_block_lr_ratio 1.0
  run dion         --opt dion --muon_lr_factor 1e-3 --dion_rank_frac 0.25
done
echo "[$(date +%H:%M:%S)] done"
