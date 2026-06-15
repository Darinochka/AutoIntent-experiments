# GoEmotions Multilabel — AutoIntent Capability Evaluation

Design for fairly evaluating **AutoIntent's ability to fit and predict on multilabel text**, using
GoEmotions as the benchmark.

## 1. Goal

Measure how well AutoIntent's AutoML pipeline learns a genuinely multilabel intent task, and how that
ability scales with the amount of training data — under both controlled (balanced) and realistic
(imbalanced) conditions.

Two distinct questions, kept separate on purpose:

- **Capability / scaling** — given *N* examples per class, how good is the fitted classifier? (primary)
- **Best target metric** — which `target_metric` should drive scoring/decision module selection? (a
  prerequisite knob, settled per balance in Phase 1 on `validation`, then frozen)

## 2. Dataset

`google-research-datasets/go_emotions`, `simplified` config.

- **28 classes** (27 emotions + `neutral`), genuinely multilabel (an utterance can carry several labels).
- Heavily **imbalanced**: `neutral` dominates; the rarest class `grief` has only ~77 train examples.
- Splits used: GoEmotions `train` (43,410), `validation` (5,426), and `test` (5,427) — **all three** (see
  §3): `train` for fitting, `validation` for Phase-1 target-metric selection, `test` for Phase-2 reporting.
  All 28 classes appear in every split (rarest-class eval count: `validation` 13, `test` 6).

## 3. How data is fed to AutoIntent

Implemented in `src/data.py` (`assemble_mapping` / `prepare_mapping`); the eval source is selected per
phase by `--eval-split` (`src/sweep.py`, `prepare_data.py`).

| GoEmotions split | Fed to AutoIntent as | Role |
|---|---|---|
| `train` (subsampled) | `train` | model fitting **and** the source AutoIntent carves HPO-validation from |
| `validation` (full, 5,426) | `test`, when `--eval-split validation` | **Phase-1** selection eval (pick the target metric) |
| `test` (full, 5,427) | `test`, when `--eval-split test` | **Phase-2** reporting eval (the headline numbers) |

We give AutoIntent **only `train` + `test`** (no explicit validation split). AutoIntent then carves its
own HPO-validation out of the (subsampled) `train` via `DataConfig.validation_size` (default 0.2). This is
deliberate: the balance treatment shapes the **HPO-validation** too, so an imbalanced train yields an
imbalanced model-selection signal — model selection is affected by balance, not just model fitting.

**Two disjoint eval sets, on purpose.** The target-metric pair is *selected* on `validation` (Phase 1) and
the capability curve is *reported* on `test` (Phase 2). The two splits never overlap, so the reporting set
is never touched during any selection — `test` stays pristine for the headline numbers. `--eval-split`
defaults to `validation`, so a casual sweep never spends `test`; you opt into `test` explicitly for Phase 2.

**Consequence to remember:** at small N the carved HPO-val is tiny (e.g. classwise-10 → ~256 train →
~51 val ≈ 2/class), so model selection is noisy. This is real and is why seeds matter (§5.3).

## 4. Evaluation metric

AutoIntent's `decision_f1` / `decision_precision` / `decision_recall` are **macro-averaged**
(`sklearn ... average="macro"`); `decision_accuracy` is per-sample subset accuracy.

- **Headline: macro `decision_f1`** (report `decision_recall` alongside). Macro weights every class
  equally, so the rarest classes (`grief`, etc.) dominate — exactly the multilabel ability we want to
  probe.
- **Ignore `decision_accuracy`** for conclusions: on sparse multilabel it sits ~0.92 (the all-negative
  baseline) and is uninformative.

Because the metric is macro, **per-class data availability is the binding constraint** — which is what the
"caps" in §5.2 control.

## 5. Experimental factors

### 5.1 Balance modes (`src/data.py`, `src/sweep.py`)

- **`classwise` — primary capability probe.** Cap *N* samples/class → balanced *N*-shot. The standard,
  fair way to measure per-class learning ability: equal budget per class, so the score reflects the model,
  not the dataset's skew. (Rare classes saturate at availability — `grief` maxes at ~77.)
- **`stratified` — realism anchor.** Floor of *N*/class, natural proportions above the floor → imbalanced,
  like real deployment data. Collapses to the **full** train once *N* > the rarest class total (~77), so
  `100`- and `500`-shot are identical (the sweep deduplicates identical (dataset, seed) cells).
- **`natural` — excluded.** A size-matched natural sample drops rare classes at small N, and AutoIntent
  requires every class in every split (including the carved HPO-val) → it raises. Available as opt-in;
  infeasible cells are skipped with a warning, not crashed.

### 5.2 Sizes (caps)

A shared shot ladder for both modes: **`5 10 25 50 100`**.

- `classwise`: a clean balanced few-shot curve (rare classes plateau ~77 above the ladder's top — an
  honest ceiling, not a bug).
- `stratified`: floors 5–50 give increasing imbalanced sizes (~2.8k → ~28k rows); 100 = full train
  (43,410). A realistic learning curve.

### 5.3 Seeds

Run **3 seeds** per cell (`--seeds 1 2 3`). Each seed redraws the subsample **and** reseeds AutoIntent's
HPO/val-carve. At 5–10 shots the variance is large; a single seed is not a fair measurement. The summary
reports `f1_mean ± f1_std` per cell.

### 5.4 Held fixed (do not vary in the capability study)

- **Target metrics**: one `(scoring, decision)` pair **per balance mode**, chosen in Phase 1 (on
  `validation`) then frozen across sizes — after checking the winner is stable across sizes (§6).
- **Preset**: `classic-light` (KNN / linear / mlknn scorers; threshold / argmax / tunable / … deciders).
- **Embedder**: the preset default `intfloat/multilingual-e5-large-instruct`.
- **Device**: `mps`.
- **Eval set**: GoEmotions `validation` (fixed, §3).

## 6. The two phases

### Phase 1 — pick the target metrics (on `validation`)

Sweep the scoring × decision `target_metric` grid **per balance mode**, on two representative sizes, with
the eval fed from `validation` (the default). Pick the pair with the best mean macro-F1 per balance, and
confirm the winner is stable across the two sizes — so freezing it across the whole size ladder is
evidence-backed, not assumed. We select once *per balance* (not once globally) because the target metric
governs how model selection reacts to class structure, which the balance treatment changes far more than
size does.

```bash
# classwise, two sizes, full metric grid (9 scoring x 4 decision), 3 seeds; eval = validation (default)
uv run sweep.py --device mps --balances classwise --sizes 25 50 --seeds 1 2 3
# stratified, same
uv run sweep.py --device mps --balances stratified --sizes 25 50 --seeds 1 2 3
```

Inspect `logs/sweep_summary_val.csv` / the "best per cell" log line. Freeze the winning `(scoring,
decision)` per balance (it should agree across the two sizes; if not, prefer the larger size and note the
instability). A sensible default if you skip Phase 1: `scoring_f1` + `decision_f1` (decision target aligned
with the headline eval metric).

### Phase 2 — capability & scaling study (on `test`)

Fix the per-balance target metrics from Phase 1; vary size × seed; report on `test` via `--eval-split
test`. Run each balance with its own frozen pair:

```bash
uv run sweep.py --device mps --eval-split test \
  --balances classwise --sizes 5 10 25 50 100 \
  --scoring-metrics scoring_f1 --decision-metrics decision_f1 --seeds 1 2 3
# repeat for --balances stratified with its own Phase-1 winner
```

= 5 sizes × 1 balance × 1 × 1 × 3 seeds = **15 runs per balance**. Read `f1_mean ± f1_std` vs size from
`logs/sweep_summary_test.csv`, per balance.

## 7. Reading the results

- **Capability curve**: `classwise` `f1_mean` vs shots → how AutoIntent's per-class multilabel ability
  grows with a controlled budget (expect a plateau once rare classes saturate).
- **Realism gap**: `stratified` vs `classwise` at comparable points → how much the natural imbalance costs.
  Note these differ in total size, so this is a directional comparison, not a clean balance-only ablation
  (see §8).
- **Stability**: `f1_std` quantifies subsample + HPO noise; large std at small N is expected.

## 8. Threats to validity / confounds

- **Selection vs. reporting are disjoint splits.** The target metric is picked on `validation` (Phase 1)
  and the curve is reported on `test` (Phase 2), so the headline numbers are *not* selected-on. The only
  thing transferred is one categorical knob (the `(scoring, decision)` pair); per-run HPO (module choice +
  thresholds) is tuned on the carved HPO-val, never on `test`. An earlier design selected and reported both
  on `validation` and was optimistically biased; this split removes that.
- **Per-balance metric freeze is verified, not assumed.** We freeze the target metric across sizes *within*
  a balance only after checking it's stable across two Phase-1 sizes (§6); across balances we don't freeze
  at all (each gets its own pair). Residual assumption: stability is checked at two sizes, not all five.
- **`test`'s long tail is thin.** `test`'s rarest class has only **6** examples (`validation`'s has 13), so
  macro-F1 — which weights the rarest classes most — has high-variance per-class estimates there. Seeds
  average *model* noise but cannot add eval examples; treat the very-rarest-class contribution as noisy.
- **`test` is spent once, by design.** We deliberately never use `test` for any selection or tuning — it is
  read exactly once, for Phase-2 reporting. A future report wanting a second independent reporting pass has
  no untouched split left; budget for that up front.
- **Size ≠ balance.** `classwise-N` and `stratified-N` have different totals, so "balanced beats imbalanced"
  can be a data-volume effect. The capability claim lives **within** each balance curve; cross-balance
  comparisons are directional only. A clean balance-only ablation would need size-matched sets (the
  `natural` mode aimed at this but is infeasible at small N — §5.1).
- **Tiny HPO-val at small N.** classwise-5/10 leave ~1–2 val samples/class for model selection → noisy
  picks. Mitigated (not removed) by seeds.
- **Stratified collapse.** stratified ≥ ~77/class = full train; treat stratified-100 as "full data".
- **Single embedder / preset.** Results are for `classic-light` + e5-large-instruct on CPU/MPS, not a
  statement about transformer fine-tuning presets.
- **Macro on a long tail.** Macro-F1 is dominated by the rarest, lowest-data classes; this is intentional
  but means absolute numbers look low compared to micro/weighted scores.

## 9. Outputs

Per sweep (written to `logs/`, git-ignored). Output names carry the eval-split tag (`val` for Phase 1,
`test` for Phase 2) so the two phases never clobber each other:

- `sweep_runs_<tag>.csv` — one row per (size, balance, scoring, decision, **seed**) run.
- `sweep_summary_<tag>.csv` — seed-aggregated: `f1_mean`, `f1_std` (sample std), mean
  precision/recall/accuracy, modal scorer/decisioner.
- `<exp>_metrics.json` (exp name ends in `-<tag>`) — full per-run report (fed split sizes, target metrics,
  eval metrics, selected modules). Enables resume.

Runs are **resumable** (existing `<exp>_metrics.json` is skipped; `--overwrite` forces) and **crash-safe**
(each fit is isolated in try/except; a failed cell is recorded as `status=failed` and the sweep continues).

## 10. Tooling reference

| File | Purpose |
|---|---|
| `prepare_data.py` (`PrepareConfig`) | build one dataset JSON (subsample + `--eval-split`) |
| `run.py` (`RunConfig`) | optimize a single pipeline, dump metrics |
| `sweep.py` (`src/sweep.py` `SweepConfig` / `run_sweep`) | the grid sweep (sizes × balances × metrics × seeds × `--eval-split`) |
| `src/data.py` | GoEmotions load, one-hot conversion, subsamplers, feeding scheme |
| `src/pipeline.py` | build pipeline (preset + metric overrides), fit, report |

All three CLIs use cyclopts + frozen-dataclass configs, so the same runs are reproducible from Python, e.g.
`run_sweep(SweepConfig(sizes=[10], balances=["classwise"], seeds=[1, 2, 3], device="mps"))`.
