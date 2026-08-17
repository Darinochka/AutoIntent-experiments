# Calibration harness

The instruments that produced `../results/`. Archived here so the experiment
stays reproducible independently of the AutoIntent repo's branch lifecycle.

## Why they moved

These scripts lived in `deeppavlov/AutoIntent` under `scripts/`. They were
removed from that repo while preparing [AutoIntent#291][pr291] for merge, for
two reasons:

1. They are validation instruments, not library code. AutoIntent's `scripts/`
   is load-bearing — `check-schema.yaml` runs `python -m
   scripts.generate_json_schema_config` — so it is a real package directory,
   not a scratch drawer.
2. `test_calibration_tracker.py` imported `calibrate_advisor` from `scripts/`,
   which is not a Python package. That import was one of the `mypy` failures
   blocking #291.

[pr291]: https://github.com/deeppavlov/AutoIntent/pull/291

## Provenance

Copied verbatim from `deeppavlov/AutoIntent`, branch
`feat/issue39-calibration-scripts` at commit `b38f3c3a` — the head of
AutoIntent PR #348, which has now been closed in favour of this archive.

| File | Role |
|---|---|
| `calibrate_advisor.py` | Runs presets under instrumentation; records predicted vs. actual VRAM/RAM/time per module, plus `headroom` / `is_feasible` / `severity_by_metric` / resolved model names per row. |
| `run_calibration_banking77.sh` | Driver for the banking77 sweep. |
| `run_phase2_isolated.sh` | One preset per process, to avoid the inter-preset GPU leak that produced a false `classic-light` OOM in the first sweep. |
| `phase1b_metadata_counterfactual.py` | Re-prices presets with true parameter counts, for the safetensors-missing metadata fallback. |
| `phase3_reduce_to_fit.py` | Exercises `reduce_to_fit` end to end. |
| `render_issue39_tables.py` | Renders the two definition-of-done tables in `../README.md`. |
| `coverage_preset.yaml` | Search space covering every module class the advisor estimates. |
| `test_calibration_tracker.py` | Unit tests for the harness's own peak-sampling and tracking logic. |

## ⚠️ `../reproduce.sh` does not use this copy yet

`../reproduce.sh` still clones AutoIntent and runs the harness from *there*:

```sh
AUTOINTENT_REF="${AUTOINTENT_REF:-feat/issue39-calibration-scripts}"
...
[[ -f "$AUTOINTENT_DIR/scripts/calibrate_advisor.py" ]] \
    || die "no scripts/calibrate_advisor.py in '$AUTOINTENT_DIR' — wrong ref?"
```

That still works today, because closing PR #348 did not delete its branch. Two
things to know:

**1. Do not delete the `feat/issue39-calibration-scripts` branch** in
`deeppavlov/AutoIntent` while `reproduce.sh` points at it. Deleting it is the
one action that breaks reproduction. This directory is the insurance against
that, but `reproduce.sh` has to be repointed to actually use it.

**2. Repointing `reproduce.sh` at a newer AutoIntent ref needs code changes**,
not just a new ref. AutoIntent #291 renames the package from the private
`autointent._advisor` to the public `autointent.advisor` and narrows its
exports. At minimum:

- `reproduce.sh`'s inline environment check does
  `from autointent._advisor import detect_hardware` → becomes
  `from autointent.advisor import detect_hardware`.
- Anything in these scripts importing `autointent._advisor` needs the same
  treatment; two public names were also renamed, `inspect` → `estimate` and
  `stats_from_dataset_obj` → `dataset_stats`.
- `Pipeline.fit(preflight=...)` now defaults to `"off"` rather than `"warn"`,
  so a script relying on preflight running implicitly must pass it explicitly.

Deliberately not done here: rewriting `reproduce.sh` to run from this directory
is a judgement call about how the experiment should work, and belongs to
whoever owns it rather than to the cleanup that displaced these files.
