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
  prerequisite knob, settled once in Phase 1, then frozen)

## 2. Dataset

`google-research-datasets/go_emotions`, `simplified` config.

- **28 classes** (27 emotions + `neutral`), genuinely multilabel (an utterance can carry several labels).
- Heavily **imbalanced**: `neutral` dominates; the rarest class `grief` has only ~77 train examples.
- Splits used: GoEmotions `train` (43,410) and `validation` (5,426). GoEmotions `test` is **unused** (see
  §3). All 28 classes appear in `validation`.

## 3. How data is fed to AutoIntent

Implemented in `src/data.py` (`SPLIT_MAP`, `assemble_mapping`).

| GoEmotions split | Fed to AutoIntent as | Role |
|---|---|---|
| `train` (subsampled) | `train` | model fitting **and** the source AutoIntent carves HPO-validation from |
| `validation` (full, 5,426) | `test` | held-out evaluation set (fixed across every run) |
| `test` | — | intentionally unused |

We give AutoIntent **only `train` + `test`** (no explicit validation split). AutoIntent then carves its
own HPO-validation out of the (subsampled) `train` via `DataConfig.validation_size` (default 0.2). This is
deliberate: the balance treatment shapes the **HPO-validation** too, so an imbalanced train yields an
imbalanced model-selection signal — model selection is affected by balance, not just model fitting.

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

- **Target metrics**: one `(scoring, decision)` pair, chosen in Phase 1 then frozen.
- **Preset**: `classic-light` (KNN / linear / mlknn scorers; threshold / argmax / tunable / … deciders).
- **Embedder**: the preset default `intfloat/multilingual-e5-large-instruct`.
- **Device**: `mps`.
- **Eval set**: GoEmotions `validation` (fixed, §3).

## 6. The two phases

### Phase 1 — pick the target metrics (run once)

Sweep the scoring × decision `target_metric` grid on a representative mid cell and pick the pair with the
best mean macro-F1.

```bash
uv run sweep.py --device mps \
  --sizes 50 --balances classwise \
  --seeds 1 2 3          # default metric grid = 9 scoring x 4 decision = 36 combos x 3 seeds
```

Inspect `logs/sweep_summary.csv` / the "best per cell" log line. Freeze the winning `(scoring, decision)`.
A sensible default if you skip Phase 1: `scoring_f1` + `decision_f1` (decision target aligned with the
headline eval metric).

### Phase 2 — capability & scaling study

Fix the target metrics from Phase 1; vary size × balance × seed.

```bash
uv run sweep.py --device mps \
  --sizes 5 10 25 50 100 --balances classwise stratified \
  --scoring-metrics scoring_f1 --decision-metrics decision_f1 \
  --seeds 1 2 3
```

= 5 sizes × 2 balances × 1 × 1 × 3 seeds = **30 runs**. Read `f1_mean ± f1_std` vs size, per balance.

## 7. Reading the results

- **Capability curve**: `classwise` `f1_mean` vs shots → how AutoIntent's per-class multilabel ability
  grows with a controlled budget (expect a plateau once rare classes saturate).
- **Realism gap**: `stratified` vs `classwise` at comparable points → how much the natural imbalance costs.
  Note these differ in total size, so this is a directional comparison, not a clean balance-only ablation
  (see §8).
- **Stability**: `f1_std` quantifies subsample + HPO noise; large std at small N is expected.

## 8. Threats to validity / confounds

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

Per sweep (written to `logs/`, git-ignored):

- `sweep_runs.csv` — one row per (size, balance, scoring, decision, **seed**) run.
- `sweep_summary.csv` — seed-aggregated: `f1_mean`, `f1_std`, mean precision/recall/accuracy, modal
  scorer/decisioner.
- `<exp>_metrics.json` — full per-run report (fed split sizes, target metrics, test metrics, selected
  modules). Enables resume.

Runs are **resumable** (existing `<exp>_metrics.json` is skipped; `--overwrite` forces) and **crash-safe**
(each fit is isolated in try/except; a failed cell is recorded as `status=failed` and the sweep continues).

## 10. Tooling reference

| File | Purpose |
|---|---|
| `prepare_data.py` (`PrepareConfig`) | build one dataset JSON (classwise/stratified subsample) |
| `run.py` (`RunConfig`) | optimize a single pipeline, dump metrics |
| `sweep.py` (`src/sweep.py` `SweepConfig` / `run_sweep`) | the grid sweep (sizes × balances × metrics × seeds) |
| `src/data.py` | GoEmotions load, one-hot conversion, subsamplers, feeding scheme |
| `src/pipeline.py` | build pipeline (preset + metric overrides), fit, report |

All three CLIs use cyclopts + frozen-dataclass configs, so the same runs are reproducible from Python, e.g.
`run_sweep(SweepConfig(sizes=[10], balances=["classwise"], seeds=[1, 2, 3], device="mps"))`.
