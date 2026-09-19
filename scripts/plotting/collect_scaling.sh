#!/bin/bash
# Best validation loss of every 355M / 720M run on this machine, as JSON lines.
# Finished runs come from summary.json; running ones from the log (marked done=false).
# usage: collect_scaling.sh <machine-tag> <repo> <python>
set -u
TAG=$1; R=$2; PY=$3
cd "$R" || exit 1
for d in exps/fw355[tb]_* exps/fw355g_* exps/fw720[tb]_* exps/fw720g_*; do
  [ -d "$d" ] || continue
  name=$(basename "$d")
  scale=$(echo "$name" | grep -oE '^fw[0-9]+' | tr -d fw)m
  if [ -f "$d/summary.json" ]; then
    best=$("$PY" -c "import json,sys;print(round(min(json.load(open(sys.argv[1]))['val_loss']),4))" "$d/summary.json")
    done=true
  else
    log=$(grep -l "Starting Experiment: $name\$" logs355/*.log logs720/*.log logsq/*.log 2>/dev/null | head -1)
    [ -n "$log" ] || continue
    # a queue writes several runs into one log, so read only this run's section
    best=$(awk -v n="$name" '/Starting Experiment:/{cur=$3} cur==n && /val_loss=/{
             match($0, /val_loss=[0-9.]+/); print substr($0, RSTART+9, RLENGTH-9)}' "$log" \
           | sort -g | head -1)
    [ -n "$best" ] || continue
    done=false
  fi
  printf '{"machine":"%s","scale":"%s","name":"%s","best":%s,"done":%s}\n' "$TAG" "$scale" "$name" "$best" "$done"
done
