#!/usr/bin/env bash
# Prepare GoEmotions and run an AutoIntent multilabel optimization.
#
# Usage: ./run.bash [MIN_SAMPLES_PER_CLASS] [PRESET] [DEVICE]
#   ./run.bash                          # full data, classic-light
#   ./run.bash 50                       # balanced stratified train subsample, classic-light
#   ./run.bash 50 classic-medium mps    # subsample, classic-medium, run embedder on mps
set -euo pipefail
cd "$(dirname "$0")"

MIN_PER_CLASS="${1:-}"
PRESET="${2:-classic-light}"
DEVICE="${3:-}"

if [[ -n "$MIN_PER_CLASS" ]]; then
  uv run prepare_data.py --min-samples-per-class "$MIN_PER_CLASS"
  SUFFIX="min${MIN_PER_CLASS}"
else
  uv run prepare_data.py
  SUFFIX="full"
fi

RUN_ARGS=(--preset "$PRESET" --run-name "goemotions-${PRESET}-${SUFFIX}")
if [[ -n "$DEVICE" ]]; then
  RUN_ARGS+=(--device "$DEVICE")
fi

uv run run.py "${RUN_ARGS[@]}"
