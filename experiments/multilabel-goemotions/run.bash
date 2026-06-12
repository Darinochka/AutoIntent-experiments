#!/usr/bin/env bash
# Prepare GoEmotions and run an AutoIntent multilabel optimization.
#
# Usage: ./run.bash [MAX_TRAIN] [PRESET]
#   ./run.bash                    # full data, classic-light
#   ./run.bash 2000               # 2000-sample train subsample, classic-light
#   ./run.bash 2000 classic-medium
set -euo pipefail
cd "$(dirname "$0")"

MAX_TRAIN="${1:-}"
PRESET="${2:-classic-light}"

if [[ -n "$MAX_TRAIN" ]]; then
  uv run prepare_data.py --max-train "$MAX_TRAIN"
  SUFFIX="$MAX_TRAIN"
else
  uv run prepare_data.py
  SUFFIX="full"
fi

uv run run.py --preset "$PRESET" --run-name "goemotions-${PRESET}-${SUFFIX}"
