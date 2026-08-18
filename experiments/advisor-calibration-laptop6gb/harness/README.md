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

Copied from `deeppavlov/AutoIntent` at commit
[`b38f3c3a`](https://github.com/deeppavlov/AutoIntent/commit/b38f3c3a8612e177e76fe94552180ef43555665c),
the head of [AutoIntent#348][pr348], which was closed in favour of this archive.

The commit is named, not a branch. `feat/issue39-calibration-scripts` — the
branch that held it — **has already been deleted**:

```console
$ git ls-remote https://github.com/deeppavlov/AutoIntent 'refs/heads/*issue39*'
$ git ls-remote https://github.com/deeppavlov/AutoIntent 'refs/pull/348/head'
b38f3c3a8612e177e76fe94552180ef43555665c        refs/pull/348/head
```

That is the whole argument for pinning commits: the branch is gone and the
commit is not, because GitHub keeps every commit that belonged to a PR under
`refs/pull/<N>/head` whether the PR merged, closed, or neither.

[pr348]: https://github.com/deeppavlov/AutoIntent/pull/348

### Changes from the AutoIntent copy

The Python is otherwise unmodified — same advisor API, same measurements. Only
paths moved, because these files are no longer inside `<autointent>/scripts/`:

| File | Change |
|---|---|
| `test_calibration_tracker.py` | `sys.path` bootstrap pointed at `parents[2]/"scripts"`, which does not exist here. Now points at this directory, where `calibrate_advisor.py` actually sits. Without this the test could not import its subject at all. |
| `run_calibration_banking77.sh` | Split `REPO_ROOT` into `HARNESS_DIR` (here) and `AUTOINTENT_DIR` (the checkout supplying the venv); invokes `calibrate_advisor.py` by absolute path; resolves relative `*.yaml` preset paths against `HARNESS_DIR` first, since the cwd is now someone else's repo. |
| `run_phase2_isolated.sh` | Same split; calls the sweep driver by absolute path and forwards `AUTOINTENT_DIR`. |
| `calibrate_advisor.py` | Two `--presets` help strings said `scripts/coverage_preset.yaml`; now `harness/coverage_preset.yaml`. Text only. |

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

## `../reproduce.sh` runs this copy

`../reproduce.sh` executes the scripts in this directory. It uses
`deeppavlov/AutoIntent` for one thing only — the advisor library and the venv
around it — pinned to a commit:

```sh
AUTOINTENT_COMMIT="${AUTOINTENT_COMMIT:-b38f3c3a8612e177e76fe94552180ef43555665c}"
AUTOINTENT_PR="${AUTOINTENT_PR:-348}"
```

If the commit is not already in the clone, it is fetched by name, then via
`refs/pull/$AUTOINTENT_PR/head`, then by full fetch, and checked out detached.
No branch name appears anywhere in the path from a fresh clone to a run.

Pinning the library at `b38f3c3a` is exact, not approximate. `../README.md`
credits `85848f2` with producing `../results/`, and the two commits differ only
under `scripts/` — which this experiment no longer reads:

```console
$ git diff --stat 85848f27 b38f3c3a
 scripts/calibrate_advisor.py               |  90 ++++-
 scripts/phase1b_metadata_counterfactual.py | 191 +++++++++
 scripts/phase3_reduce_to_fit.py            | 284 +++++++++++++
 scripts/render_issue39_tables.py           | 116 ++++++
 scripts/run_calibration_banking77.sh       |   4 +-
 scripts/run_phase2_isolated.sh             |  38 ++
```

`src/autointent/` is identical between them, so the pinned library is the one
that produced the archived numbers.

## Moving the pin forward

The pin is deliberately behind `dev`. [AutoIntent#291][pr291] renames the
package from the private `autointent._advisor` to the public
`autointent.advisor` and narrows its exports, so bumping `AUTOINTENT_COMMIT` to
anything at or after that merge needs code changes here, not just a new SHA:

- `../reproduce.sh`'s inline environment check does
  `from autointent._advisor import detect_hardware` → becomes
  `from autointent.advisor import detect_hardware`.
- Every `autointent._advisor` import in these scripts needs the same treatment;
  two public names were also renamed, `inspect` → `estimate` and
  `stats_from_dataset_obj` → `dataset_stats`, and `PreflightError` moves from
  `autointent._pipeline` to `autointent.advisor`.
- `Pipeline.fit(preflight=...)` now defaults to `"off"` rather than `"warn"`,
  so a script relying on preflight running implicitly must pass it explicitly.

Worth knowing before you do: #291 also changes advisor *behaviour* (among other
things, `reduce_to_fit` no longer prunes by VRAM when VRAM is not the binding
constraint). Re-running against a post-#291 pin is a new measurement, not a
reproduction of `../results/`. Bump the pin and re-record, or leave it and keep
the tables meaningful — but do not do the first while claiming the second.
