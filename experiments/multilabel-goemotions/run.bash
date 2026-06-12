#!/usr/bin/env bash
# Prepare GoEmotions and run an AutoIntent multilabel optimization.
#
# Usage: ./run.bash [MIN_SAMPLES_PER_CLASS] [PRESET] [DEVICE] [BALANCE]
#   ./run.bash                              # full data, classic-light
#   ./run.bash 50                           # stratified subsample (floor 50/class), classic-light
#   ./run.bash 50 classic-medium mps        # subsample, classic-medium, embedder on mps
#   ./run.bash 50 classic-light cpu classwise   # classwise-flattened subsample (cap 50/class)
#
# For target-metric overrides, call run.py directly with --scoring-metric / --decision-metric.
set -euo pipefail
cd "$(dirname "$0")"

MIN_PER_CLASS="${1:-}"
PRESET="${2:-classic-light}"
DEVICE="${3:-}"
BALANCE="${4:-stratified}"

if [[ -n "$MIN_PER_CLASS" ]]; then
  uv run prepare_data.py --min-samples-per-class "$MIN_PER_CLASS" --balance "$BALANCE"
  SUFFIX="${BALANCE}${MIN_PER_CLASS}"
else
  uv run prepare_data.py
  SUFFIX="full"
fi

RUN_ARGS=(--preset "$PRESET" --run-name "goemotions-${PRESET}-${SUFFIX}")
if [[ -n "$DEVICE" ]]; then
  RUN_ARGS+=(--device "$DEVICE")
fi

uv run run.py "${RUN_ARGS[@]}"
