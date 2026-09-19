#!/bin/bash
# Print results/ddp124/final_losses.json for the 124M DDP campaign, from the runs on this node.
# Finished runs (summary.json) are marked done; runs still training report where they are.
# usage (on the training node):  bash scripts/collect_ddp124.sh [exps_dir] [log_dir] > final_losses.json
set -u
OUT=${1:-exps_ddp124}
LOGS=${2:-logs_ddp124}
PY=${PY:-python}
ITERS=${ITERS:-150000}

printf '{\n  "_source": "%s on %s, %s; 124M GPT-2, FineWeb, 4-GPU DDP, batch 32x512, %s iterations = 2.46B tokens (1x Chinchilla), hyperparameters transferred from the tuned 124M grid",\n' \
  "$OUT/<run>/summary.json" "$(hostname)" "$(date -u +%Y-%m-%d)" "$ITERS"
printf '  "_step_time": "wall-clock uses the controlled 4-GPU benchmark in results/benchmarks/ddp_bench_124m.json (A100 PCIe, 8 sequences per GPU): the training node is shared, so its own step times measure the neighbours",\n'
printf '  "runs": [\n'

first=1
for d in "$OUT"/ddp124_*_seed*/; do
  [ -d "$d" ] || continue
  run=$(basename "$d")
  method=${run#ddp124_}; method=${method%_seed*}
  seed=${run##*_seed}
  if [ -f "$d/summary.json" ]; then
    loss=$("$PY" -c "import json,sys;print(round(min(json.load(open(sys.argv[1]))['val_loss']),4))" "$d/summary.json")
    iters=$ITERS; done=true
  else
    log="$LOGS/$method.log"
    [ -f "$log" ] || continue
    # one log holds several launches of the same run; read only this run's section
    read -r iters loss <<<"$(awk -v n="$run" '/Starting Experiment:/{cur=$3}
      cur==n && /Eval: Iter=/ {match($0,/Iter=[0-9]+/); it=substr($0,RSTART+5,RLENGTH-5)
        match($0,/val_loss=[0-9.]+/); vl=substr($0,RSTART+9,RLENGTH-9)
        if (best=="" || vl+0 < best+0) {best=vl; bit=it} }
      END{if (best!="") print bit, best}' "$log")"
    [ -n "${loss:-}" ] || continue
    done=false
  fi
  [ $first -eq 1 ] || printf ',\n'
  first=0
  printf '    {"method": "%s", "seed": %s, "iters": %s, "loss": %s, "done": %s}' \
    "$method" "$seed" "$iters" "$loss" "$done"
done
printf '\n  ]\n}\n'
