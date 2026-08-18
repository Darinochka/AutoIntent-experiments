# Advisor calibration on constrained hardware (6 GB GPU / 16 GB RAM)

Validation run for [issue #39](https://github.com/Darinochka/AutoIntent-experiments/issues/39).

Every previous calibration ran on an A100-80 GB / 486 GB box where all four
budgets came back `ample` with `findings_over=0`, so the advisor's core promise
— catching an infeasible preset *before* the OOM — and the reduce-to-fit path
had never been exercised. This run exercises both on real constrained hardware.

## Verdict

**The advisor clears the MVP bar.** Every preset that was fitted matched its
prediction (4/4), both `over` verdicts really OOMed, both `ample` verdicts
really fit, the strict gate aborted before allocating a single byte of VRAM,
and reduce-to-fit produced a pipeline that genuinely ran.

The verdicts also survive correcting the advisor's biggest metadata defect
(below), so they are not accidents of an inflated fallback.

What is *not* healthy is everything downstream of the verdict: the disk axis is
broken in two independent ways, transformer time is under-predicted by up to
52×, and `reduce_to_fit`'s budget-priority selector is dead code that happens
to do the right thing only on VRAM-bound machines.

---

## Table 1 — feasibility (the key one)

| preset | model | pred VRAM | advisor verdict | actual | match? |
| --- | --- | ---: | --- | --- | :---: |
| `transformers-heavy` | microsoft/deberta-v3-large | 15.69 GB | **over** | OOM | ✅ |
| `transformers-no-hpo` | microsoft/deberta-v3-small | 10.19 GB | **over** | OOM | ✅ |
| `zero-shot-encoders` | BAAI/bge-reranker-v2-m3 | 3.36 GB | **ample** | fit | ✅ |
| `classic-light` | intfloat/multilingual-e5-large-instruct | 1.83 GB | **ample** | fit | ✅ |
| `transformers-light` | microsoft/deberta-v3-small | 10.91 GB | **over** | not run | — |
| `classic-medium` | intfloat/multilingual-e5-large-instruct | 1.83 GB | **ample** | not run | — |
| `nn-heavy` | (cnn/rnn from scratch) | 1.13 GB | **ample** | not run | — |
| `nn-medium` | (cnn/rnn from scratch) | 1.04 GB | **ample** | not run | — |

`transformers-no-hpo` was the case the issue called out as *"the borderline one
— may actually fit ~5 GB"*. It does not: deberta-v3-small at `batch_size=96`
OOMed with 5.39 GiB allocated by PyTorch against a 5.67 GiB card. The advisor's
`over` call is a **true positive**, not a false alarm.

The `advisor verdict` column is `severity_by_metric["vram"]`, not `headroom` —
see the note under Phase 1 for why those differ.

## Table 2 — accuracy on presets that fit

| preset | actual/pred VRAM | actual/pred RAM | actual/pred time | actual VRAM | pred VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| `classic-light` | **1.22×** | 0.41× | 0.59× | 2.23 GB | 1.83 GB |
| `zero-shot-encoders` | 0.67× | 0.39× | **52.4×** | 2.26 GB | 3.36 GB |

Reading these (want ≤ 1 but close — the advisor's contract is a conservative
upper bound):

* **VRAM is the trustworthy axis, but is not always an upper bound.**
  `classic-light` really used **1.22× its prediction**. Harmless here (2.23 GB
  against a 5.67 GB card) but it breaks the "conservative upper bound" contract
  on the one metric the advisor is judged by. On a 3–4 GB card that ratio flips
  an `ample` verdict into an OOM.
* **RAM over-predicts ~2.5× on both**, consistent with the issue's known ~2×
  note. Safe at 16 GB, would wrongly flag RED at 8 GB.
* **Time is not merely "ordering-only", it is dangerously optimistic for
  cross-encoders.** `zero-shot-encoders` predicted 0.009 h and took 0.470 h — a
  **52× under-prediction**. `description_cross` scores every
  (utterance × intent) pair, so its true cost scales with `n_samples ×
  n_classes`; the estimate appears to scale with `n_samples` alone, so the error
  grows linearly with class count. On banking77's 77 classes that is already ~50×.

### A measurement caveat that matters at 6 GB

`actual vram_gb` is `torch.cuda.memory_allocated()` — **allocated tensor bytes
only**. It excludes the CUDA context (~250–400 MB here), the caching
allocator's reserved-but-unallocated blocks, and any non-torch GPU memory.
Driver-level `nvidia-smi` peaks ran consistently higher than the recorded
figures (e.g. `classic-light`: 2.23 GB recorded vs **2 549 MiB** observed).
That gap is rounding error on an 80 GB A100 and roughly 5–10 % of the budget on
a 6 GB card, always in the unsafe direction. Both signals are in the JSON as
`vram_gb_tracker` / `vram_gb_sampler`.

---

## Hardware — matches the issue's target profile exactly

| | |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3060 **Laptop** GPU, 6144 MiB (advisor sees 5.67 GiB usable) |
| Driver / CUDA | 595.58.03 / 13.2 |
| RAM | 15.03 GB |
| CPU | 16 cores |
| Free disk | 166 GB |
| `device_class` | `low-gpu` |

**No VRAM simulation or `--budget-vram-gb` override was used.** This is a real
6 GB card; every number comes from the machine's own `detect_hardware()`.

| | |
| --- | --- |
| AutoIntent | `deeppavlov/AutoIntent` @ commit [`85848f2`](https://github.com/deeppavlov/AutoIntent/commit/85848f27), owned by [AutoIntent#291](https://github.com/deeppavlov/AutoIntent/pull/291) |
| Python / torch / transformers | 3.14.4 / 2.11.0+cu130 / 4.57.6 |
| Dataset | `DeepPavlov/banking77` — 10 003 train / 77 classes; Phase 2 subsampled |

### Reproducing

Everything runs from [`reproduce.sh`](reproduce.sh) in this directory, against
the harness in [`harness/`](harness/) — both live in this repo. It clones
`deeppavlov/AutoIntent` for the advisor library alone, pinned to a commit,
builds an environment that can actually load the deberta presets (including the
undocumented `protobuf` dependency), runs the phases, and collects the JSONs
back into [`results/`](results/).

```bash
./reproduce.sh --check      # verify CUDA + report the detected profile, run nothing
./reproduce.sh              # all phases (~2-3 h, almost entirely Phase 2)
./reproduce.sh 1 1b 3       # the cheap phases only (~10 min, no training)
./reproduce.sh tables       # re-render Tables 1 and 2 from the committed JSONs
```

| phase | what it does | cost |
| --- | --- | --- |
| `1` | advisor verdicts across all 10 presets, no training | ~2 min |
| `1b` | metadata counterfactual (CPU only, no CUDA context) | ~2 min |
| `2` | real fits, one preset per process | ~1–2 h |
| `3` | strict gate + reduce-to-fit + a real fit of the pruned config | ~6 min |
| `tables` | re-render Tables 1 and 2 from the JSONs | instant |

Overrides: `AUTOINTENT_DIR` (where to clone/find AutoIntent),
`AUTOINTENT_COMMIT` (default `b38f3c3a`), `AUTOINTENT_PR` (default `348` — the
PR whose `refs/pull/<N>/head` keeps that commit fetchable), `OUT_DIR`,
`SKIP_SETUP=1` to reuse an existing checkout and venv.

`./reproduce.sh tables` against the committed `results/` regenerates Tables 1
and 2 below exactly — the tables in this README are not hand-transcribed.

**The calibration harness lives in [`harness/`](harness/)**, in this repo. It
was written in `deeppavlov/AutoIntent` under `scripts/` and removed from there
while preparing [AutoIntent#291](https://github.com/deeppavlov/AutoIntent/pull/291)
for merge; this directory is the copy of record, taken at commit `b38f3c3a`
(head of the now-closed [AutoIntent#348](https://github.com/deeppavlov/AutoIntent/pull/348)).
See [`harness/README.md`](harness/README.md) for what each file does and what
changed in the move.

Nothing here depends on an AutoIntent *branch*. The only dependency is the
advisor library — which belongs in AutoIntent and nowhere else — pinned to one
commit that stays fetchable via `refs/pull/348/head` no matter what happens to
the branches that once held it. (The branch this experiment used to name,
`feat/issue39-calibration-scripts`, has since been deleted; the commit is
still there.)

Raw JSONs: [`results/`](results/).

---

## Phase 1 — advisor verdicts (no training)

All 10 bundled presets with `SKIP_FIT=1`. `zero-shot-llm` and `classic-heavy`
skip cleanly for missing optional extras (`openai`, `catboost`) — expected.

Per-metric severities were **not recorded by the calibrator** before this run:
`is_feasible` and `headroom` exist on `PreflightReport` but were dropped on the
floor, so a calibration JSON could not answer the question the advisor exists to
answer. `calibrate_advisor.py` now records `headroom`, `is_feasible`,
`severity_by_metric` and the resolved `models` per row.

**`headroom` is not a "will it fit" signal.** `classic-light`, `classic-medium`
and `zero-shot-encoders` all report `headroom=tight` while every one of their
four *resource* budgets is `ample`. The `tight` comes from a config-phase
warning:

```
'linear' entry has 1 unique configuration but hpo_config.n_trials=20
  — expect ~19 duplicate trials unless the sampler dedupes.
```

Use `severity_by_metric["vram"]` for feasibility; `headroom` mixes in
search-space hygiene warnings.

**Preset-swap check (issue caveat):** `transformers-heavy` resolves to
`microsoft/deberta-v3-large`, not `bert-base-uncased`. The calibrator now
records the resolved model name per row so this can never be ambiguous again.

Severity thresholds, for reading the tables: `ample` < 0.9 × budget ≤ `tight`
< 1.0 × budget ≤ `over`. On 5.67 GB: ample < 5.10 GB, tight 5.10–5.67,
over ≥ 5.67.

---

## Phase 1b — the low-confidence fallback is structural, not an offline artifact

The issue's caveat says *"stay online so transformer estimates don't hit the
inflating low-confidence fallback."* **Staying online does not help.** This box
was online, every Hub lookup succeeded, and all three `transformers-*` presets
still returned `low_confidence=True`.

Root cause: `_hub_metadata()` reads parameter counts **only** from
`HfApi().model_info().safetensors`. `microsoft/deberta-v3-*` predates
safetensors and ships `pytorch_model.bin` only, so `info.safetensors is None`,
`total_params == 0`, and a flat 350 M-param default is substituted — for *every*
deberta checkpoint, large and small alike.

Re-running preflight with parameter counts taken from the architecture
(meta-device init, nothing downloaded):

| preset | params assumed | params true | VRAM as-run | VRAM corrected | verdict changes? |
| --- | ---: | ---: | ---: | ---: | --- |
| `transformers-heavy` | 350.0 M | **435.1 M** | 15.69 GB (over) | 17.40 GB (over) | no |
| `transformers-light` | 350.0 M | **142.0 M** | 10.91 GB (over) | 6.72 GB (over) | no |
| `transformers-no-hpo` | 350.0 M | **142.0 M** | 10.19 GB (over) | 6.01 GB (over) | no |

Two consequences, pointing opposite ways:

1. **The OVER verdicts are real.** Every one survives the correction — the
   single most important result for the MVP question. And the corrected
   `transformers-no-hpo` figure (6.01 GB against a 5.67 GB budget) is a
   strikingly good match for a run that OOMed with 5.64 GiB in use; the
   fallback's 10.19 GB was right for the wrong reason.
2. **The fallback is not conservative where it matters.** It *under*-counts
   deberta-v3-large (350 M assumed vs 435 M real) while *over*-counting the
   small models ~1.7×. `_hub.py` documents the default as existing "so cost
   estimates upper-bound rather than under-predict"; for the flagship heavy
   preset it does the opposite.

**Fix:** when `info.safetensors` is absent, derive the parameter count from the
torch weight-file size (`pytorch_model.bin` bytes ÷ bytes-per-dtype) or from the
`config.json` shapes — the advisor already downloads `config.json` for
`hidden_size`/`num_hidden_layers`, so the data is one step away.

### The same root cause explains the "heavy = no-hpo" time bug

The issue lists *"transformer time not n_trials-aware end-to-end
(`_effective_trials` exists but delivered predictions don't reflect it: heavy =
no-hpo)"* as an open code issue. **`_effective_trials` is fine** — it is wired
through to `_estimate_transformer_model(n_trials=...)` at `_resource.py:674,694`.
The identical 172.52 h across heavy / light / no-hpo has a different cause:

* `_time_for_transformer` computes FLOPs from `params_millions`, which the
  fallback pins at 350 M for both deberta-large and deberta-small;
* batch size cancels out (`steps_per_epoch = n_samples // batch` multiplied by
  `step_flops ∝ batch`);
* all three presets ship `num_train_epochs: [30]`, `n_trials: 40` and a single
  `bert` entry, so `_effective_trials` returns 40 for each.

Same params × same epochs × same trials = same time. **Fixing metadata
resolution fixes the time ranking for free**; no change to `_effective_trials`
is needed.

---

## Phase 2 — real fits (cold embedding cache, `MAX_TRIALS=2`)

One preset per process. `classic-light`, `transformers-no-hpo` and
`transformers-heavy` use `SUBSAMPLE_PER_CLASS=30` (2 310 train samples) as the
issue specifies.

### Measurement isolation — read this before trusting any Phase 2 number

`run_calibration_banking77.sh` walks every preset **inside one Python process**.
On a big box that is harmless. On a 6 GB card it silently invalidates the sweep:

> `transformers-no-hpo` OOMed and left **5.4 GiB still allocated**. The next
> preset, `classic-light`, then "OOMed" too — on a 978 MiB allocation, against a
> 1.83 GB prediction and an `ample` verdict.

Run in its own process, `classic-light` **fits comfortably at 2.23 GB peak**.
Its OOM in the shared process was pure contamination — compare
[`results/phase2_CONTAMINATED_single_process.json`](results/phase2_CONTAMINATED_single_process.json)
with [`results/phase2_classic-light.json`](results/phase2_classic-light.json).

Reported as-is, this would have been written up as an advisor false-`ample` —
blaming the component that was right. Two changes came out of it:

* `calibrate_advisor.py` now releases accelerator memory between presets and
  **annotates the row when memory survives collection**, so a contaminated sweep
  announces itself.
* `harness/run_phase2_isolated.sh` runs one preset per process — the only
  airtight fix, since the kernel reclaims the GPU on exit regardless of what the
  Python object graph still holds.

**Recommendation:** any calibration on a card the workload can actually fill
must use process isolation.

### `zero-shot-encoders` deviates: `SUBSAMPLE_PER_CLASS=5`

`description_cross` scores every (utterance × intent) pair, so at 30 per class
that is 2 310 × 77 = 178 k cross-encoder pairs per trial; a trial that sampled
`batch_size=40` needed 4 447 batches at ~1.4 s (~100 min) and the preset had
already run 53 minutes without finishing. The issue's own reasoning applies —
*"feasibility is driven by model + batch size, not dataset size"* — so the
subsample was cut rather than the trial count. Predictions are recomputed from
the same subsampled stats, so the **ratios stay apples-to-apples**; only the
absolute RAM/time figures aren't comparable with the other rows.

The aborted 30-per-class attempt is itself a feasibility datapoint: **53 minutes
of cross-encoder scoring, driver-level VRAM peaking at 2 853 MiB, no OOM** —
consistent with the `ample` verdict and with the 5-per-class run.

---

## Phase 3 — reduce-to-fit and the strict gate

Run via `harness/phase3_reduce_to_fit.py` against the live 5.67 GB profile. Full
output: [`results/phase3_reduce_to_fit.json`](results/phase3_reduce_to_fit.json).

### A. `Pipeline.fit(preflight="strict")` on `transformers-heavy` — PASS

```
outcome              : PreflightError
raised_before_alloc  : true
vram_peak_gb         : 0.0
elapsed_s            : 1.34
message              : Preflight check failed with 1 OVER finding(s):
                         [resource] VRAM ~15.7 GB vs available 5.7 GB
```

Aborts in 1.3 s with **zero VRAM allocated** and no model download, versus the
several-minute download-then-OOM of the ungated path. This is the advisor
delivering exactly what it was built for.

### B. `reduce_to_fit(transformers-heavy)` — PASS, with a caveat

`transformers-heavy` has exactly one scoring module, so pruning can only empty
the search space. It correctly refuses rather than returning a do-nothing
pipeline:

```
scoring modules: ['bert'] -> []
ReduceToFitError: All scoring modules were pruned to fit the budget; the
resulting pipeline would have nothing to run. Raise the budget or add cheaper
scoring modules.
```

**Caveat:** the issue's acceptance wording was *"raise `ReduceToFitError`
**pointing at a lighter preset**"*. It does not — the message is generic and
names no alternative. The advisor already ships `recommend()`, which ranks
presets by feasibility on the current hardware; wiring it into this error would
turn a dead end into `try classic-light (1.8 GB, ample)`. Small, high-value
follow-up.

### C. `reduce_to_fit` on a prunable search space + a real fit — PASS

The interesting case needs something expensive to drop and something cheap to
keep, so: deberta-v3-large + knn + linear in one scoring node.

```
before : ['bert', 'knn', 'linear']   vram 15.69 GB  headroom=over
after  : ['knn', 'linear']           vram  1.83 GB  headroom=ample
real fit of the pruned config: outcome=ok, 219.4 s, vram_peak 2.23 GB
```

**Reduce-to-fit produced a pipeline that actually ran** — the DoD criterion, and
the first time this path has executed against hardware that genuinely could not
hold the original config.

---

## Other findings

### The disk axis is broken in two independent ways

Both appear only on a cold machine.

**1. `_is_warm_cached()` is self-defeating — download is always predicted as 0.**
For `transformers-heavy` the advisor reported *"Disk ~0.0 GB to download, 3.9 GB
already cached"* on a machine whose deberta-v3-large cache held **28 KB** (one
`config.json`). Mechanism: the function probes for `model.safetensors` /
`pytorch_model.bin` / the shard index — all three miss — then falls back to
`scan_cache_dir()` and returns `True` if *any* repo directory with that id
exists. `_shape_from_config()`, called moments earlier **in the same
`_hub_metadata()` call**, created that directory by downloading `config.json`.
Verified:

```
probe model.safetensors            -> None
probe pytorch_model.bin            -> None
probe model.safetensors.index.json -> None
scan_cache_dir sees repo           -> True     # created by the config.json fetch
_is_warm_cached                    -> True
```

The fit then really downloaded **1.63 GiB** against that 0.0 GB prediction:

```
833.2 MB  pytorch_model.bin    (snapshot 64a8c8ea)
833.1 MB  model.safetensors    (snapshot 71e5b576 — transformers resolved a
                                second revision that does have safetensors)
  2.4 MB  spm.model
```

*Fix:* require an actual weight blob, not just repo presence.

**2. `total_file_bytes` counts files torch never loads.** It sums every sibling
with a size. For deberta-v3-large that is `pytorch_model.bin` (873.7 MB, used),
`pytorch_model.generator.bin` (571.3 MB, discarded ELECTRA generator),
`tf_model.h5` (1736.6 MB, TensorFlow mirror) and `spm.model` (2.5 MB, used).
Booked **2.97 GiB** where a torch fine-tune needs ~0.82 GiB.

### `cli_smoke` divergence — the "~10×" caveat is stale

The issue says to ignore `cli_smoke` because the paths "diverge ~10×". On this
branch they agree **exactly** — `divergence == {}` for all eight non-skipped
Phase 1 presets, VRAM and time matching to full float precision.

The only non-empty divergence in the entire run is `classic-light` under
`--max-trials 2`: `cli_time=0.330 h` vs `direct_time=0.055 h` (6×). Cause:
`_run_cli_smoke()` takes `(preset, stats, budget_vram_gb)` and no trials
argument, and `autointent-advisor inspect` has no n_trials flag — so the CLI
path costs the preset's bundled `n_trials=20` while the direct path costs 2.
Apples to oranges, not a wrapper regression; only `time_hours` scales with
`n_trials`, which is exactly the one key that diverged. `calibrate_advisor.py`
now labels this case instead of reporting "numeric drift".

### Latent bug: `reduce_to_fit`'s budget-priority lookup never matches

`_pick_module_to_drop()` is documented as *"drop the driver with the largest
cost along the first dimension that has at least one OVER finding"*
(VRAM > time > RAM > disk). That branch is unreachable:

```python
findings_by_metric = {f.metric for f in report.findings if f.severity == Severity.OVER}
priority = ["vram_gb", "time_hours", "ram_gb", "disk_download_gb"]
driver_key = next((k for k in priority if k in findings_by_metric), None) or "vram_gb"
```

The resource phase sets `metric=` to `"vram"` / `"ram"` / `"disk"` / `"time"`
(`_resource.py:501,508,514,526,533`) — never `"vram_gb"` / `"time_hours"`. The
vocabularies never intersect, so `driver_key` **always** falls through to the
`"vram_gb"` default. Deterministic repro (report OVER on *time* only, with a
cheap-VRAM/slow module and an expensive-VRAM/fast module):

```
finding metrics marked OVER    : {'time'}
keys _pick_module_to_drop tries: ['vram_gb', 'time_hours', 'ram_gb', 'disk_download_gb']
intersection                   : set()   <-- always empty

expected drop (time is over)   : ('scoring', 'slow_but_small')
actual drop                    : ('scoring', 'fast_but_big')
```

No test caught it because on a VRAM-bound machine — every case anyone has
exercised, including this one — the fallback gives the right answer. *Fix:* map
finding metrics to driver keys before the lookup.

### Packaging: the deberta presets can't run from a clean install

The first Phase 2 attempt died before allocating a byte of VRAM:

```
Trial 0 failed ... ImportError('requires the protobuf library but it was not
found in your environment')
```

`microsoft/deberta-v3-*` uses a SentencePiece-backed `DebertaV2Tokenizer` whose
fast-tokenizer conversion needs `protobuf`. No extra pulls it in —
`uv sync --extra transformers --extra sentence-transformers --group sentencepiece`
is not enough — so **all three `transformers-*` presets fail on a clean
environment**, with an error that looks nothing like a missing dependency.
`protobuf` belongs in the `transformers` extra.

### `run_calibration_banking77.sh` default invocation was broken

The preset auto-discovery heredoc ran bare `python`, which has no `autointent`
on its path, so invoking the script without `PRESETS` died with
`error: argument --presets: expected at least one argument`. Fixed to use
`uv run --no-sync python`.

### UX: the protection is opt-in

`Pipeline.fit(..., preflight="warn")` is the **default**. `warn` logs the OVER
finding at `ERROR` level and then fits anyway. Only `preflight="strict"` raises.
So the out-of-the-box experience for `transformers-heavy` on this machine is:
one red log line, a 1.63 GiB download, then `torch.OutOfMemoryError` — when the
advisor knew the answer in 1.3 s. Worth considering `strict` as the default on a
`low-gpu` device class.

---

## Recommended follow-ups, in priority order

1. **Resolve parameter counts without safetensors** (weight-file size or
   `config.json` shapes). Fixes the low-confidence flag on every deberta preset,
   the small-model VRAM inflation, the non-conservative large-model estimate,
   *and* the heavy = no-hpo time ranking — one fix, four symptoms.
2. **Fix `_is_warm_cached()`** to require a weight blob. The disk axis currently
   predicts 0 GB download for every model in every preset.
3. **Fix `_pick_module_to_drop()`'s metric→driver-key mapping.** Currently
   correct only by accident on VRAM-bound hardware.
4. **Make cross-encoder time scale with `n_samples × n_classes`.** 52×
   under-prediction on `zero-shot-encoders`, growing with class count.
5. **Point `ReduceToFitError` at a feasible preset** via the existing
   `recommend()`.
6. **Add `protobuf` to the `transformers` extra.**
7. **Exclude `tf_model.h5` / `*.generator.bin` from `total_file_bytes`.**
8. Consider `preflight="strict"` as the default on `low-gpu`.

## Changes made to `deeppavlov/AutoIntent` during this run

All in `scripts/`, made during this run and landed upstream as commit
[`85848f2`](https://github.com/deeppavlov/AutoIntent/commit/85848f27) on
[AutoIntent#291](https://github.com/deeppavlov/AutoIntent/pull/291). Those files
have since been removed from AutoIntent and archived in
[`harness/`](harness/) — see [`harness/README.md`](harness/README.md):

| file | change |
| --- | --- |
| `calibrate_advisor.py` | record `headroom` / `is_feasible` / `severity_by_metric` / resolved `models` per row; release accelerator memory between presets and flag leaks; label the expected `--max-trials` cli-smoke divergence |
| `run_calibration_banking77.sh` | preset auto-discovery now runs under `uv run` |
| `run_phase2_isolated.sh` | **new** — one preset per process |
| `phase1b_metadata_counterfactual.py` | **new** — re-predict with corrected model metadata |
| `phase3_reduce_to_fit.py` | **new** — strict gate + reduce-to-fit + real fit of the pruned config |
| `render_issue39_tables.py` | **new** — regenerate Tables 1 and 2 from the JSONs |
