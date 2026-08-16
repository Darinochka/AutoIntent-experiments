#!/usr/bin/env bash
# End-to-end reproduction of the issue #39 advisor calibration.
#
# The advisor and its calibration harness live in a *different* repo
# (deeppavlov/AutoIntent), so this script's job is to pin that repo to the
# right ref, build an environment that can actually load the deberta presets,
# run all four phases, and collect the JSONs back into ./results/.
#
# Usage:
#   ./reproduce.sh                 # everything (~2-3 h, mostly Phase 2)
#   ./reproduce.sh 1 1b 3          # only the cheap phases (~10 min, no training)
#   ./reproduce.sh --check         # verify environment + hardware, run nothing
#
# Environment overrides:
#   AUTOINTENT_DIR   where to find/clone deeppavlov/AutoIntent
#                    (default: ../../../AutoIntent relative to this script)
#   AUTOINTENT_REF   git ref holding the calibration scripts
#                    (default: feat/issue39-calibration-scripts)
#   OUT_DIR          where phase JSONs are collected (default: ./results)
#   SKIP_SETUP       non-empty to skip clone/checkout/uv-sync
#
# Phases:
#   1   advisor verdicts only, no training          (~2 min)
#   1b  metadata counterfactual, CPU only           (~2 min)
#   2   real fits, one preset per process           (~1-2 h)
#   3   strict gate + reduce-to-fit + a real fit    (~6 min)
#   tables  re-render Tables 1 and 2 from the JSONs (instant)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTOINTENT_DIR="${AUTOINTENT_DIR:-$(cd "$HERE/../../.." && pwd)/AutoIntent}"
AUTOINTENT_REF="${AUTOINTENT_REF:-feat/issue39-calibration-scripts}"
OUT_DIR="${OUT_DIR:-$HERE/results}"
RUNS_DIR=""   # set after AUTOINTENT_DIR is known

log() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# --- argument parsing -------------------------------------------------------
CHECK_ONLY=""
PHASES=()
for arg in "$@"; do
    case "$arg" in
        --check) CHECK_ONLY=1 ;;
        -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
        1|1b|2|3|tables) PHASES+=("$arg") ;;
        *) die "unknown argument '$arg' (want: 1 | 1b | 2 | 3 | tables | --check)" ;;
    esac
done
[[ ${#PHASES[@]} -eq 0 ]] && PHASES=(1 1b 2 3 tables)

# --- 0. the AutoIntent checkout --------------------------------------------
if [[ -z "${SKIP_SETUP:-}" ]]; then
    log "Setting up AutoIntent at $AUTOINTENT_DIR (ref: $AUTOINTENT_REF)"
    if [[ ! -d "$AUTOINTENT_DIR/.git" ]]; then
        git clone https://github.com/deeppavlov/AutoIntent.git "$AUTOINTENT_DIR"
    fi
    git -C "$AUTOINTENT_DIR" fetch --all --quiet
    git -C "$AUTOINTENT_DIR" checkout "$AUTOINTENT_REF"

    # protobuf is NOT pulled in by any AutoIntent extra, but deberta-v3's
    # SentencePiece -> fast-tokenizer conversion needs it. Without it all three
    # transformers-* presets die at trial 0 with an ImportError that looks
    # nothing like a missing dependency. See README, "Packaging gap".
    log "Syncing environment (+ protobuf)"
    ( cd "$AUTOINTENT_DIR" \
        && uv sync --extra transformers --extra sentence-transformers --group sentencepiece \
        && uv pip install protobuf )
fi

[[ -d "$AUTOINTENT_DIR" ]] || die "AUTOINTENT_DIR '$AUTOINTENT_DIR' does not exist"
[[ -f "$AUTOINTENT_DIR/scripts/calibrate_advisor.py" ]] \
    || die "no scripts/calibrate_advisor.py in '$AUTOINTENT_DIR' — wrong ref?"
RUNS_DIR="$AUTOINTENT_DIR/calibration_runs"
mkdir -p "$OUT_DIR"

run_py() { ( cd "$AUTOINTENT_DIR" && uv run --no-sync python "$@" ); }

# --- environment + hardware report -----------------------------------------
log "Environment"
( cd "$AUTOINTENT_DIR" && uv run --no-sync python - <<'PY'
import torch
from autointent._advisor import detect_hardware

hw = detect_hardware()
print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
print(f"accelerator={hw.accelerator}  device={hw.device_name}")
print(f"VRAM={hw.vram_gb:.2f} GB  RAM={hw.ram_gb:.2f} GB  class={hw.device_class}")
if hw.accelerator != "cuda":
    raise SystemExit(
        "\nThis calibration needs CUDA. The whole point is a real constrained GPU;\n"
        "on CPU every verdict degenerates and the OOM rows cannot be reproduced."
    )
if not 4.5 <= hw.vram_gb <= 8.0:
    print(
        f"\nNOTE: the published results came from a 5.67 GiB card. Yours reports "
        f"{hw.vram_gb:.2f} GiB,\nso the per-preset verdicts will legitimately differ "
        f"from the tables in README.md.\nThresholds: ample < 0.9x budget <= tight < 1.0x budget <= over."
    )
PY
) || die "environment check failed"

if [[ -n "$CHECK_ONLY" ]]; then
    log "--check requested; stopping before any run."
    exit 0
fi

collect() {  # collect <glob-dir> <destination-name>
    local src dest
    dest="$OUT_DIR/$2"
    # shellcheck disable=SC2086  # deliberate glob
    src="$(ls -t $1 2>/dev/null | head -1 || true)"
    if [[ -n "$src" ]]; then cp "$src" "$dest" && echo "  -> $dest"; else
        echo "  !! nothing matched $1 (phase may have failed)"; fi
}

for phase in "${PHASES[@]}"; do
case "$phase" in

1)
    log "Phase 1 — advisor verdicts only (no training)"
    ( cd "$AUTOINTENT_DIR" && SKIP_FIT=1 REQUIRE_CUDA=1 RUN_NAME=laptop6gb_preflight \
        OUTPUT_DIR="$RUNS_DIR" scripts/run_calibration_banking77.sh )
    collect "$RUNS_DIR/banking77_*.json" "phase1_preflight_all_presets.json"
    ;;

1b)
    log "Phase 1b — metadata counterfactual (CPU only, no GPU context)"
    # Deliberately hides the GPU: this only needs preflight arithmetic, and
    # creating a CUDA context costs ~300 MB we may not be able to spare.
    ( cd "$AUTOINTENT_DIR" && CUDA_VISIBLE_DEVICES="" uv run --no-sync python \
        scripts/phase1b_metadata_counterfactual.py \
        --assume-hardware 5.67,15.03 \
        --output "$RUNS_DIR/phase1b_counterfactual.json" )
    collect "$RUNS_DIR/phase1b_counterfactual.json" "phase1b_metadata_counterfactual.json"
    ;;

2)
    log "Phase 2 — real fits, ONE PRESET PER PROCESS"
    # Process isolation is not a style choice. In a single process a preset
    # that OOMs leaves several GB allocated and the *next* preset OOMs on an
    # allocation it would never have made on a clean device.
    ( cd "$AUTOINTENT_DIR" && PRESETS="classic-light transformers-no-hpo transformers-heavy" \
        SUBSAMPLE_PER_CLASS=30 OUTPUT_DIR="$RUNS_DIR/phase2_isolated" \
        scripts/run_phase2_isolated.sh )
    for p in classic-light transformers-no-hpo transformers-heavy; do
        collect "$RUNS_DIR/phase2_isolated/$p/banking77_*.json" "phase2_$p.json"
    done

    # zero-shot-encoders is subsampled harder: description_cross scores every
    # (utterance x intent) pair, so 30/class = 178k pairs/trial and a single
    # trial ran ~100 min. Feasibility is driven by model + batch size, not
    # dataset size, and predictions are recomputed from the same stats, so the
    # ratios stay comparable. See README.
    ( cd "$AUTOINTENT_DIR" && PRESETS="zero-shot-encoders" SUBSAMPLE_PER_CLASS=5 \
        RUN_NAME=laptop6gb_fit_sub5 OUTPUT_DIR="$RUNS_DIR/phase2_isolated_sub5" \
        scripts/run_phase2_isolated.sh )
    collect "$RUNS_DIR/phase2_isolated_sub5/zero-shot-encoders/banking77_*.json" \
            "phase2_zero-shot-encoders_sub5.json"
    ;;

3)
    log "Phase 3 — strict gate + reduce-to-fit (+ a real fit of the pruned config)"
    run_py scripts/phase3_reduce_to_fit.py --subsample-per-class 30 \
        --output "$RUNS_DIR/phase3.json"
    collect "$RUNS_DIR/phase3.json" "phase3_reduce_to_fit.json"
    ;;

tables)
    log "Tables 1 and 2, rendered from the collected JSONs"
    # phase2_CONTAMINATED_single_process.json is kept in results/ as evidence
    # for the isolation finding — it holds a *false* classic-light OOM and must
    # never be fed to the renderer, which keys rows by preset name.
    fits=()
    for f in "$OUT_DIR"/phase2_*.json; do
        [[ "$(basename "$f")" == *CONTAMINATED* ]] && continue
        [[ -e "$f" ]] && fits+=("$f")
    done
    [[ ${#fits[@]} -gt 0 ]] || die "no phase 2 JSONs in $OUT_DIR — run phase 2 first"
    run_py scripts/render_issue39_tables.py \
        --preflight "$OUT_DIR/phase1_preflight_all_presets.json" \
        --fits "${fits[@]}" \
        --counterfactual "$OUT_DIR/phase1b_metadata_counterfactual.json"
    ;;
esac
done

log "Done. JSONs in $OUT_DIR"
