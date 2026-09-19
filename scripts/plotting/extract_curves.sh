#!/bin/bash
# Print every run's validation curve as JSON lines: {"machine","scale","run","iters","loss"}
# Scans all logs and splits them on "Starting Experiment:", so runs launched by a
# queue (several per log file) are separated correctly.
# usage: extract_curves.sh <machine-tag> <repo>
set -u
TAG=$1; R=$2
cd "$R" || exit 1
cat logs355/*.log logs720/*.log logsq/*.log 2>/dev/null | awk -v tag="$TAG" '
/Starting Experiment:/ { name = $3; next }
/Eval: Iter=/ {
  if (name == "") next
  match($0, /Iter=[0-9]+/);      it = substr($0, RSTART+5, RLENGTH-5)
  match($0, /val_loss=[0-9.]+/); vl = substr($0, RSTART+9, RLENGTH-9)
  its[name] = its[name] (its[name] == "" ? "" : ",") it
  lss[name] = lss[name] (lss[name] == "" ? "" : ",") vl
}
END {
  for (n in its) {
    scale = (n ~ /^fw720/) ? "720m" : "355m"
    printf "{\"machine\":\"%s\",\"scale\":\"%s\",\"run\":\"%s\",\"iters\":[%s],\"loss\":[%s]}\n", \
           tag, scale, n, its[n], lss[n]
  }
}'
