Ran all three phases on a real 6 GB laptop GPU. **The advisor clears the MVP bar: 4/4 verdicts matched reality, and reduce-to-fit produced a pipeline that actually ran.**

Hardware was the target profile exactly — RTX 3060 **Laptop**, 6144 MiB (advisor sees 5.67 GiB), 15.03 GB RAM, `device_class=low-gpu`. **No VRAM simulation or `--budget-vram-gb` override.** AutoIntent @ `feat/feasibility-check` `85848f2`.

Full write-up + all JSONs: `experiments/advisor-calibration-laptop6gb/`.

### Table 1 — feasibility

| preset | model | pred VRAM | advisor verdict | actual | match? |
| --- | --- | ---: | --- | --- | :---: |
| `transformers-heavy` | deberta-v3-large | 15.69 GB | **over** | OOM | ✅ |
| `transformers-no-hpo` | deberta-v3-small | 10.19 GB | **over** | OOM | ✅ |
| `zero-shot-encoders` | bge-reranker-v2-m3 | 3.36 GB | **ample** | fit | ✅ |
| `classic-light` | multilingual-e5-large-instruct | 1.83 GB | **ample** | fit | ✅ |
| `transformers-light` | deberta-v3-small | 10.91 GB | **over** | not run | — |
| `classic-medium` / `nn-heavy` / `nn-medium` | — | 1.83 / 1.13 / 1.04 GB | **ample** | not run | — |

`transformers-no-hpo` — the borderline case you flagged — **does not fit**: deberta-v3-small at `batch_size=96` OOMed with 5.39 GiB allocated against a 5.67 GiB card. The `over` call is a true positive, not a false alarm.

Model check: `transformers-heavy` → `microsoft/deberta-v3-large`. No preset swap. The calibrator now records the resolved model name per row so this can't be ambiguous again.

### Table 2 — accuracy on presets that fit

| preset | actual/pred VRAM | actual/pred RAM | actual/pred time | actual VRAM | pred VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| `classic-light` | **1.22×** | 0.41× | 0.59× | 2.23 GB | 1.83 GB |
| `zero-shot-encoders` | 0.67× | 0.39× | **52.4×** | 2.26 GB | 3.36 GB |

- VRAM is the trustworthy axis but **is not always an upper bound** — `classic-light` used 1.22× its prediction. Harmless at 5.67 GB, but it breaks the conservative-upper-bound contract, and on a 3–4 GB card that ratio flips `ample` into an OOM.
- RAM over-predicts ~2.5× on both, matching your ~2× note.
- Time isn't just "ordering-only", it's **dangerously optimistic for cross-encoders**: 0.009 h predicted vs 0.470 h real. `description_cross` scores every (utterance × intent) pair, so cost scales with `n_samples × n_classes` while the estimate appears to scale with `n_samples` alone — the error grows linearly with class count.

### Phase 3 — reduce-to-fit ✅

- **Strict gate:** `PreflightError` in **1.34 s with `vram_peak = 0.0`** — before any allocation or download. vs. the ungated path's 1.63 GiB download then `OutOfMemoryError`.
- **`reduce_to_fit(transformers-heavy)`:** single scoring module, so it correctly raises rather than returning a do-nothing pipeline. *But it does not point at a lighter preset* — the message is generic. `recommend()` already exists and would turn this dead end into "try classic-light (1.8 GB, ample)".
- **`reduce_to_fit` on a prunable space** (deberta-v3-large + knn + linear): pruned `bert`, `15.69 GB over → 1.83 GB ample`, and **the pruned config really fit** — 219 s, 2.23 GB peak.

---

## Findings that change the open-issues list

**1. The low-confidence fallback is structural — "stay online" does not avoid it.** This box was online and every Hub lookup succeeded. `microsoft/deberta-v3-*` ships no safetensors, so `info.safetensors is None` → `total_params == 0` → flat 350 M default for *every* deberta. Re-predicting with true param counts:

| preset | assumed | true | as-run | corrected | verdict changes? |
| --- | ---: | ---: | ---: | ---: | --- |
| `transformers-heavy` | 350 M | 435 M | 15.69 GB (over) | 17.40 GB (over) | no |
| `transformers-light` | 350 M | 142 M | 10.91 GB (over) | 6.72 GB (over) | no |
| `transformers-no-hpo` | 350 M | 142 M | 10.19 GB (over) | 6.01 GB (over) | no |

Every OVER survives the correction — the verdicts are not fallback artifacts. But the fallback **under**-counts deberta-v3-large (350 M vs 435 M) while over-counting the small ones ~1.7×, the opposite of what `_hub.py` claims it's for.

**2. `_effective_trials` is not the cause of "heavy = no-hpo".** It *is* wired through (`_resource.py:674,694`). The identical 172.52 h is because `_time_for_transformer` derives FLOPs from `params_millions` — pinned at 350 M for both models by the fallback — batch size cancels out, and all three presets ship epochs=30/n_trials=40 with one `bert` entry. **Fixing metadata resolution fixes the time ranking for free.**

**3. The disk axis is dead, and it's the advisor's own doing.** `transformers-heavy` predicted `disk_download_gb = 0.0`, "3.9 GB already cached", on a machine whose deberta-v3-large cache held **28 KB**. `_is_warm_cached()` falls back to `scan_cache_dir()` and returns True if the repo dir exists — and `_shape_from_config()` created that dir by downloading `config.json` moments earlier **in the same call**. The fit then downloaded **1.63 GiB**. Separately, `total_file_bytes` counts `tf_model.h5` (1.7 GB) and the discarded ELECTRA generator: 2.97 GiB booked vs ~0.82 GiB really needed.

**4. `cli_smoke` "diverges ~10×" is stale.** The paths agree **exactly** (`divergence == {}`) for all eight non-skipped presets. The one divergence in the whole run is `classic-light` under `--max-trials 2` (0.330 h vs 0.055 h): `_run_cli_smoke()` takes no trials argument and `autointent-advisor inspect` has no n_trials flag, so the CLI costs the bundled `n_trials=20`. Apples to oranges — and only `time_hours` scales with `n_trials`, which is exactly the key that diverged.

**5. New latent bug — `reduce_to_fit` always prunes by VRAM.** `_pick_module_to_drop()` matches `Finding.metric` (`"vram"`, `"time"`, `"ram"`, `"disk"`) against `priority = ["vram_gb", "time_hours", "ram_gb", "disk_download_gb"]`. The vocabularies never intersect, so `driver_key` always falls through to the `"vram_gb"` default and the documented "drop along whichever budget breached" logic is unreachable. Deterministic repro in the write-up: with a time-only OVER it drops the VRAM hog instead of the time hog. Correct by accident on VRAM-bound machines, which is why no test caught it.

**6. Calibration on a small GPU needs process isolation.** `run_calibration_banking77.sh` sweeps every preset in one process. `transformers-no-hpo` OOMed leaving **5.4 GiB allocated**, so `classic-light` "OOMed" next — on a 978 MiB allocation, against a 1.83 GB `ample` prediction. In its own process it fits at 2.23 GB. Both JSONs are attached; reported as-is this would have looked like an advisor false-`ample`.

**7. The deberta presets can't run from a clean install.** They need `protobuf` for the SentencePiece→fast tokenizer conversion, and no extra pulls it in — so all three `transformers-*` presets die at trial 0 with an error that looks nothing like a missing dependency.

**8. `headroom` isn't a "will it fit" signal.** `classic-light` / `classic-medium` / `zero-shot-encoders` report `headroom=tight` while all four *resource* budgets are `ample` — the `tight` is a config-phase duplicate-trials warning. Use `severity_by_metric["vram"]`.

**9. Measurement caveat.** `actual vram_gb` is `torch.cuda.memory_allocated()` — tensor bytes only, excluding the CUDA context (~250–400 MB) and allocator reserve. Driver-level peaks ran consistently higher (`classic-light`: 2.23 GB recorded vs 2549 MiB observed). Noise on an A100; 5–10 % of the budget here, always in the unsafe direction.

**10. The protection is opt-in.** `preflight="warn"` is the `fit()` default: it logs the OVER at ERROR level and fits anyway. Worth considering `strict` as the default on `low-gpu`.

### Suggested priority

1. Resolve param counts without safetensors — one fix, four symptoms (low-confidence flag, small-model inflation, non-conservative large-model estimate, time ranking).
2. Fix `_is_warm_cached()` to require a weight blob.
3. Fix `_pick_module_to_drop()`'s metric→driver-key mapping.
4. Make cross-encoder time scale with `n_samples × n_classes`.
5. Point `ReduceToFitError` at a feasible preset via `recommend()`.
6. Add `protobuf` to the `transformers` extra.

### Calibrator changes made along the way

In the local `deeppavlov/AutoIntent` clone (uncommitted): `calibrate_advisor.py` now records `headroom` / `is_feasible` / `severity_by_metric` / resolved `models` per row (previously the JSON couldn't answer the advisor's own question), releases accelerator memory between presets and flags leaks, and labels the expected `--max-trials` cli-smoke divergence. `run_calibration_banking77.sh`'s preset auto-discovery ran bare `python` and so the no-`PRESETS` invocation always failed — now uses `uv run`. New: `run_phase2_isolated.sh`, `phase1b_metadata_counterfactual.py`, `phase3_reduce_to_fit.py`, `render_issue39_tables.py`.
