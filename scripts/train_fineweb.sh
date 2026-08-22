#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# FineWeb is stored in the shared datasets directory.
FW_DATASETS_DIR="${DATASETS_DIR:-~/datasets}"

GPU_ID="0"
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
  GPU_ID="$1"
  shift
fi

if [[ " $* " == *" --model "* ]]; then
  exec env DATASETS_DIR="$FW_DATASETS_DIR" "$SCRIPT_DIR/train_baselines.sh" "$GPU_ID" --dataset fineweb --algo-set regular "$@"
fi

env DATASETS_DIR="$FW_DATASETS_DIR" "$SCRIPT_DIR/train_baselines.sh" "$GPU_ID" --dataset fineweb --model base --algo-set regular "$@"
env DATASETS_DIR="$FW_DATASETS_DIR" "$SCRIPT_DIR/train_baselines.sh" "$GPU_ID" --dataset fineweb --model llama --algo-set regular "$@"
