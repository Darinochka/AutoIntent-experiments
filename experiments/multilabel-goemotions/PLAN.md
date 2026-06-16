# Overnight Run Plan — AutoIntent Capability Evaluation (GoEmotions multilabel)

**Author:** Claude (autonomous overnight run, started 2026-06-16 ~00:45).
**Goal:** Execute the two-phase study in `DESIGN.md` and deliver a report on AutoIntent's evaluated
multilabel capabilities. This file records *intentions + expected results per phase* so they survive
across the night and any context compaction. Update the "STATUS" checkboxes as phases complete.

---

## Hard constraints (machine safety)

- **17.2 GB RAM, 8 cores, MPS.** Embedder `intfloat/multilingual-e5-large-instruct` shares system RAM
  with MPS. **Run exactly ONE fit-process at a time.** Never launch parallel sweeps.
- Other Claude worktrees on this machine intermittently run `mypy` (~2 GB). Stay conservative; keep a
  memory margin. If `memory_pressure` free% drops below ~12%, pause new launches.
- All sweeps are **resumable** (existing `<exp>_metrics.json` skipped) and CSVs **accumulate** per
  eval tag, so an interrupted night loses only the un-run tail, never finished cells.

## Measured cost model (classic-light, mps, eval ≈ 5.4k)

| train rows | fit wall time |
|---|---|
| 256  (classwise-10) | 20 s |
| 2,460 (classwise-100) | 83 s |
| 28,233 (stratified-50) | **>17 min (aborted)** |

**Consequence:** classwise train caps at ~2,460 rows → the entire classwise study is cheap and is the
design's *primary* probe. Large stratified floors (≥14k rows) are prohibitively expensive **and** a
memory risk, so they are **excluded tonight** (documented deviation below).

## Documented deviations from DESIGN.md (compute-driven, to disclose in the report)

1. **Drop `decision_accuracy` from the Phase-1 decision grid.** DESIGN §4 already calls it degenerate
   (~0.92 all-negative baseline). Decision grid = {`decision_f1`,`decision_precision`,`decision_recall`}.
   Scoring grid = all 9. → **27 pairs** (was 36).
2. **Phase-1 uses 2 seeds** (design suggested 3). Selection is a *ranking*; 2 seeds is enough to rank and
   banks the primary classwise curve sooner. Phase-2 keeps the design's 5/3 seed schedule for headlines.
3. **Stratified Phase-1 grid run at floor 5 only** (train ~2.8k) instead of {25,50} (14k/28k). Winner
   stability optionally checked at floor 10 if time permits. Cross-balance independence is preserved
   (full 27-grid per balance); only the *size* is made cheap.
4. **Stratified Phase-2 capped at floor 25** (~14k train). Floors 50 (28k) and 100 (full 43k) are
   skipped to respect the machine. Stratified remains "directional realism anchor" per DESIGN §8 — this
   does not affect the primary classwise capability claim.

---

## Execution order (priority: secure the primary deliverable first)

### STEP A — Phase 1 classwise (eval=validation)  ⏳
```
uv run sweep.py --device mps --eval-split validation --balances classwise \
  --sizes 25 50 --seeds 1 2 \
  --scoring-metrics scoring_accuracy scoring_f1 scoring_hit_rate scoring_log_likelihood \
    scoring_map scoring_neg_coverage scoring_neg_ranking_loss scoring_precision scoring_recall \
  --decision-metrics decision_f1 decision_precision decision_recall
```
- ~108 fits, train ≤1280, ~40 s avg → **~70 min**.
- **Output:** `logs/sweep_summary_val.csv`, `logs/sweep_runs_val.csv`.
- **Expected report:** ranked (scoring,decision) by `f1_mean` at sizes 25 & 50 for classwise. Pick the
  pair with best mean macro-F1; confirm it agrees across the two sizes.
- **Prior expectation:** a decision target aligned with the headline (`decision_f1`) paired with
  `scoring_f1` or `scoring_map` is the likely winner. If `decision_recall` wins it means the optimizer
  is trading precision for rare-class recall — note it.
- **FROZEN CLASSWISE PAIR:** `scoring=scoring_neg_coverage  decision=decision_f1` (done 01:47, 42min, 0 fails).
  Notes: neg_coverage ties neg_ranking_loss exactly; at size 50 decision target is irrelevant (f1/prec/rec
  all tie 0.2728); winner uses `linear` scorer + `adaptive` decider. f1@25=0.2168, f1@50=0.2728.

### STEP B — Phase 2 classwise (eval=test)  ⏳  ← PRIMARY DELIVERABLE
```
uv run sweep.py --device mps --eval-split test --balances classwise \
  --sizes 5 10 --seeds 1 2 3 4 5 \
  --scoring-metrics <frozen> --decision-metrics <frozen>
uv run sweep.py --device mps --eval-split test --balances classwise \
  --sizes 25 50 100 --seeds 1 2 3 \
  --scoring-metrics <frozen> --decision-metrics <frozen>
```
- 19 fits → **~25 min**. Output accumulates into `logs/sweep_summary_test.csv`.
- **Expected report:** classwise `f1_mean ± f1_std` vs shots {5,10,25,50,100}. Capability curve.
- **Prior expectation:** monotonic rise then plateau as rare classes saturate (~77 cap). Large `f1_std`
  at 5/10. classwise-5 may record `failed` seeds (rare class absent from carved HPO-val, DESIGN §5.1) —
  that is expected, not a crash.

### STEP C — Phase 1 stratified (eval=validation)  ⏳
```
uv run sweep.py --device mps --eval-split validation --balances stratified \
  --sizes 5 --seeds 1 2 \
  --scoring-metrics scoring_accuracy scoring_f1 scoring_hit_rate scoring_log_likelihood \
    scoring_map scoring_neg_coverage scoring_neg_ranking_loss scoring_precision scoring_recall \
  --decision-metrics decision_f1 decision_precision decision_recall
```
- ~54 fits, train ~2.8k, ~95 s → **~85 min**.
- **FROZEN STRATIFIED PAIR → record here:** `scoring=____  decision=____`
- Optional stability: rerun winner pair at `--sizes 10 --seeds 1 2` (~4 fits, ~12 min) if time allows.

### STEP D — Phase 2 stratified (eval=test)  ⏳
```
uv run sweep.py --device mps --eval-split test --balances stratified \
  --sizes 5 10 --seeds 1 2 3 4 5 --scoring-metrics <frozen> --decision-metrics <frozen>
uv run sweep.py --device mps --eval-split test --balances stratified \
  --sizes 25 --seeds 1 2 3 --scoring-metrics <frozen> --decision-metrics <frozen>
```
- 13 fits; heavy tail (floor-25 ~8-12 min/fit) → **~45-60 min**. Cap at floor 25.
- **Expected report:** stratified `f1_mean` vs floor {5,10,25}; realism gap vs classwise (DIRECTIONAL
  only — sizes differ, DESIGN §8 "Size ≠ balance").

### STEP E — Capability report  ⏳
Synthesize: (1) classwise capability curve (primary), (2) realism gap (directional), (3) stability,
(4) frozen metrics per balance, (5) threats-to-validity per DESIGN §8 + the deviations above.

---

## STATUS (update as we go)
- [x] A: Phase 1 classwise → frozen `scoring_neg_coverage + decision_f1`
- [x] B: Phase 2 classwise (PRIMARY curve) — DONE, monotonic, see below
- [x] C: Phase 1 stratified → frozen `scoring_precision + decision_f1` (floor-5, 35min, 3/54 optuna-NaN fails)
  Caveat: top-4 scoring within 1 std (0.2499 vs 0.2462) → within-noise selection; decision target irrelevant.
  Winner scorer = mlknn/knn (vs classwise's linear) — optimal config differs by balance regime.
- [x] D: Phase 2 stratified — floors 5/10/25 (0.248/0.263/0.281), capped at 25 (~14k); floor-25 fit ~5.5min
- [x] E: report delivered → REPORT.md

### RESULT — classwise capability curve (eval=test, scoring_neg_coverage + decision_f1)
| shots | train | seeds | f1_mean | f1_std | prec | recall | scorer/dec |
|------|------|------|--------|-------|------|-------|-----------|
| 5   | 127  | 5 | 0.1517 | 0.0092 | 0.121 | 0.326 | mlknn/adaptive |
| 10  | 255  | 5 | 0.1735 | 0.0079 | 0.163 | 0.275 | mlknn/adaptive |
| 25  | 633  | 3 | 0.2173 | 0.0202 | 0.205 | 0.389 | linear/adaptive |
| 50  | 1261 | 3 | 0.2690 | 0.0089 | 0.223 | 0.509 | linear/adaptive |
| 100 | 2475 | 3 | 0.3160 | 0.0075 | 0.263 | 0.477 | linear/adaptive |
Monotonic, still rising at 100 (no plateau — most classes have >100 avail; only ~rarest cap ~77).
Scorer switches mlknn→linear at N≥25. precision<recall throughout (over-predicts). cw5 did NOT fail.

## Live notes / results log
- 2026-06-16 00:39 calibration: cw10=20s, cw100=83s, strat50(28k)>17min aborted. Embedder cached.
- 2026-06-16 01:55 BLOCKER+FIX: AutoIntent 0.3.0 embedder calls huggingface_hub.model_info() per embed()
  to key its cache -> tripped HF 1000-req/5min quota (429), compounded by other worktree agents sharing
  the IP; fatal under HF_HUB_OFFLINE. Fixed in src/pipeline.py `_patch_embedder_offline_cache_key`:
  resolve commit hash from local refs/main (== remote sha, so cache key unchanged). All sweeps now run
  with HF_HUB_OFFLINE=1. Phase 1 classwise (slow fits) was unaffected; freeze stands.
