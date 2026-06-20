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

---

# Phase 3 — Extended classwise capability curve (data-vs-imbalance limit)  (added 2026-06-18)

**Question (from user):** how far can we raise the classwise cap before we (a) saturate the data or
(b) make it "sufficiently imbalanced", and where does AutoIntent's macro-F1 plateau?

**Core finding that frames the whole phase:** the classwise cap keeps a row while *any* of its labels
is still under target, so the *balanced* property is hard-bounded by the rarest class — `grief`, 77
train rows. Past cap=77 `grief` is pinned and every extra sample only buys imbalance. So "saturate"
and "imbalanced" are not two endpoints: balance saturates at 77, *data* saturates only at the full
43,410 rows (no cap), and imbalance grows continuously in between. There is also an irreducible ~2×
imbalance even at cap=77 from multilabel co-occurrence (majority classes overshoot the cap).

GoEmotions train tail (per-class counts): grief 77 · pride 111 · relief 153 · nervousness 164 ·
embarrassment 303 · ... · median ~1,297 · admiration 4,130 · neutral 14,219. (28 classes, 16.4%
multilabel, mean 1.18 labels/row.)

## Realized size / imbalance / cost per cap (simulated classwise_subsample, seed 1)

| cap | train rows | classes pinned (<cap) | max/min | CV | regime | est. 1-fit time |
|----:|----:|:--:|--:|--:|---|---|
| 77   | 1,918  | 0  | 2.0 | 0.19 | balance saturates (anchor)     | ~70 s |
| 100  | 2,475  | 1  | 2.5 | 0.19 | controlled — ALREADY HAVE 0.316 | ~83 s |
| 200  | 4,708  | 4  | 4.7 | 0.24 | controlled→tipping             | ~2.4 min |
| 500  | 10,820 | 5  | 11.8| 0.37 | imbalance-dominant             | ~6 min |
| 1000 | 19,286 | 10 | 22.5| 0.46 | heavy (>14k safety line)       | ~12 min |
| full (cap 50000) | 43,410 | 27 | 185 | — | natural / data saturates | ~25–40 min ⚠ |

Note: cap 50000 ≡ no cap — all class totals < 50000 so every labelled train row is kept (full natural
distribution). Cost = linear fit to measured (2,475 rows→83 s; 28k→>17 min) ×1.3 above 10k rows.

## Machine-safety note (binding constraint)
Caps 1000 (19k) and full (43k) EXCEED the 14k-row line Phase-1/2 flagged as a memory/time risk (a 28k
fit was aborted at >17 min). Mitigation: run ONE fit at a time, sequential tiers, resumable (finished
cells skipped), monitor `memory_pressure` — pause new launches if free% < ~12%. At launch free was 38%.

## Execution (frozen pair scoring_neg_coverage + decision_f1, eval=test, accumulates into sweep_*_test.csv)
- TIER 1 (safe, <5k rows): `--sizes 77 200 --seeds 1 2 3`            (6 fits, ~11 min)
- TIER 2 (mid, 10.8k/19k):  `--sizes 500 1000 --seeds 1 2`           (4 fits, ~36 min)
- TIER 3 (full, 43k):       `--sizes 50000 --seeds 1`                (1 fit, ~25–40 min, run last)

All prefixed `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ... --device mps --balances classwise`.

**Expected:** macro-F1 keeps rising from 0.316 (cap100) through the controlled zone, then the curve
should bend as imbalance overtakes added data — the rare tail (pinned at 77) gets relatively starved,
so macro-F1 (which weights rare classes equally) plateaus or dips while micro/precision keep climbing.
Full-data number is the apples-to-apples vs published BERT 0.46 and the fine-tuning verdict: ~0.40+ ⇒
frozen embedder is not the bottleneck; stall ~0.33 ⇒ frozen embedding is the ceiling, FT needed.
Imbalance (max/min, #pinned) computed offline from the saved `ge_classwise_*_test.json` per point.

### STATUS (Phase 3) — DONE 2026-06-19 (AutoIntent 0.3.1; version-controlled vs 0.3.0)
- [x] T1 caps 77, 200 → 0.296±0.006 / 0.341±0.004 (3 seeds, all ok)
- [x] T2 caps 500, 1000 → 0.347±0.002 / 0.350±0.005 (2 seeds, all ok)
- [x] T3 full (cap 50000 = 43,410 rows) → **0.353 ± 0.002** (3 seeds; ~16 min/seed, mem >70% free)
- [x] REPORT.md extended-curve section → §9 added; §1/§5/§7/§8 updated
- KEY FINDING: macro-F1 plateaus ~0.35; 18× more data past cap-100 buys +0.04 → gap to BERT 0.46 is the
  FROZEN EMBEDDING, not data → fine-tuning genuinely required. §5's 0.38–0.43 prediction falsified.
- Version control: 0.3.1 reproduced 0.3.0 cap-100/seed-1 bit-for-bit (0.32409774762737037). Offline patch
  needed a one-line fix (revision arg) for 0.3.1's changed signature.

---

# Phase 4 — Fine-tuning vs the frozen-embedding ceiling  (added 2026-06-19)

**Question (from user):** Phase 3 showed `classic-light` (frozen e5) plateaus at macro-F1 ≈ 0.35 and that
the gap to fine-tuned BERT (0.46) is the *representation*, not data. Phase 4 tests that head-on: does
end-to-end **fine-tuning** a transformer at the same balanced caps (100, 300) close the gap?

**Method.** AutoIntent `transformers-no-hpo` preset → `bert` scoring node (HF Trainer, problem_type=
`multi_label_classification`, BCE loss, sigmoid outputs — verified proper multilabel FT). Overrides:
- model → **bert-base-uncased** — cached locally (runs OFFLINE) AND the exact model of the published
  GoEmotions baseline (0.46 macro-F1). So the comparison is apples-to-apples: same model + 28-class
  taxonomy + test split; the only difference vs the literature is *our balanced subsample* vs their full
  data. (deberta-v3-small, the preset default, is NOT cached → would need network; bert-base is better here.)
- device → mps; epochs ≤ 15 with early stopping (preset default: patience 3 on scoring_f1, val_fraction 0.2);
  batch_size 32 (MPS memory); lr 3e-5 (standard small-data BERT-FT recipe).
- scoring target → scoring_f1; decision target → **decision_f1** (matches Phase-3 headline ⇒ directly comparable).
- hpo n_trials small (5): the single scoring config trains the transformer ONCE; trials only tune the decision
  threshold. (`transformers-light` HPO-searches bs×lr over 40 trials → up to 40 trainings → hours on MPS; the
  no-hpo preset avoids that.)

**Runner:** `run_finetune.py` (injects the overrides into the preset dict; writes a sweep-compatible metrics
JSON with a `finetune` block). Caps: classwise 100 (have dataset, ~2,475 rows) and 300 (build via
`sweep.py --sizes 300`, ~6.5k rows). Seed 1 (+ seed 2 at cap 100 if time permits, for variance).

**Feasibility (smoke tests cap-10, mps):** bert-base FT runs offline on MPS (after `uv add accelerate` —
HF Trainer needs accelerate≥0.26; the sentence-transformers extra omits it). **Cost ≈ ONE training per cap,
independent of n_trials:** cap-10 wall time was 44s at n_trials=1 and 47s at n_trials=8 (+3s for 8× trials)
→ the single fixed scoring config trains the transformer once; extra trials only re-tune the decision
threshold on cached scores. Projected from the per-step rate: cap-100 ≈ 8–18 min, cap-300 ≈ 20–45 min
(early stopping, patience 3, likely shortens both). Confirms the no-hpo preset is the right choice.

**Machine safety:** bert-base (110M params) FT on MPS shares the 17 GB unified memory. ONE fit at a time,
batch 32, watchdog on memory (<13% free → pause). Run offline: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTORCH_ENABLE_MPS_FALLBACK=1`.

**Expected / decision rule:** FT should beat frozen-e5 at matched data; the question is by how much.
- cap-100 FT ≫ 0.316 (say ≥ 0.40) ⇒ fine-tuning IS the lever Phase 3 predicted; representation was the wall.
- cap-100 FT ≈ frozen-e5 (~0.32–0.35) ⇒ at a few-k rows bert-base can't yet exploit FT; the 0.46 baseline's
  edge is mostly its full-data training, so "FT needs *data*" — nuances the Phase-3 verdict.
Either way Phase 4 quantifies the FT lift at fixed small data and sharpens the §9 conclusion.

### STATUS (Phase 4)
- [x] Smoke test (cap 10) — offline/MPS OK; needed `uv add accelerate`; bert trains once (~11min full-15ep @cap100)
- [x] `run_finetune.py` runner + PLAN section + recipe debugging (diag_ft.py / diag_ft_warmup.py)
- [x] cap 100 FT → **0.170 ± 0.003** (2 seeds) ≪ frozen-e5 0.316 (FT data-starved, loses)
- [x] cap 200 FT → **0.399** (seed 1) > frozen-e5 0.341 (crossover already passed)
- [x] cap 300 FT → **0.407 ± 0.001** (2 seeds, ~31min/seed) ≫ frozen-e5's 0.353 full-data ceiling; ~88% of BERT 0.46
- [x] exact frozen-e5 cap-300 (matched 6816-row dataset) = 0.340 → FT beats frozen by +0.067 at cap-300
- [x] REPORT.md §10 + §1/§5/§7 updates + figure (plot_ft.py → figures/phase4_ft_vs_frozen.png)
- VERDICT: **FT scales steeply with data and breaks the frozen ceiling.** cap-100: FT 0.173 < frozen 0.316
  (FT data-starved, loses). cap-300: FT **0.408** > frozen's 0.353 asymptote, nearing BERT-full 0.46. The
  crossover is between 100 and 300. CONFIRMS Phase-3 "FT needed to beat ~0.35" AND nuances it: FT needs a
  data threshold (~few-k balanced rows); below it, frozen e5 + linear is the better choice.

---

## Live notes / results log
- 2026-06-16 00:39 calibration: cw10=20s, cw100=83s, strat50(28k)>17min aborted. Embedder cached.
- 2026-06-16 01:55 BLOCKER+FIX: AutoIntent 0.3.0 embedder calls huggingface_hub.model_info() per embed()
  to key its cache -> tripped HF 1000-req/5min quota (429), compounded by other worktree agents sharing
  the IP; fatal under HF_HUB_OFFLINE. Fixed in src/pipeline.py `_patch_embedder_offline_cache_key`:
  resolve commit hash from local refs/main (== remote sha, so cache key unchanged). All sweeps now run
  with HF_HUB_OFFLINE=1. Phase 1 classwise (slow fits) was unaffected; freeze stands.
- 2026-06-19 00:10 Phase 3 start. autointent upgraded 0.3.0->0.3.1 (pinned in pyproject). Pre-flight:
  (a) FIXED offline patch — 0.3.1 calls `_get_latest_commit_hash(model_name, revision)` (2 args); old
  1-arg patch would TypeError every fit. Added optional `revision` param. (b) Added `autointent_version`
  to report JSON for per-point provenance. (c) Version control fit: cap-100/seed-1 on 0.3.1 == 0.3.0
  bit-for-bit (0.32409774762737037, same modules) -> banked 0.3.0 caps 5-100 comparable to fresh 0.3.1.
  (d) Backed up logs/ to logs_backup_v030_*. (e) Wrote analyze_imbalance.py (max/min, CV, #pinned per cap).
- 2026-06-19 00:13-00:59 Tiers 1-3 ran sequentially (one fit-process at a time, watchdog on mem<13%/errors,
  free% stayed 72-78%). All cells ok, 0 failures, 0 HF-429. Full-data (43k) fit = ~16 min.
- 2026-06-19 01:00 RESULT: curve plateaus ~0.35 (full=0.353±0.002, 3 seeds). Imbalance climbs 2.0->185
  (CV 0.19->1.41, pinned 0->28). prec rises 0.22->0.344, recall falls 0.51->0.394 as imbalance grows.
  Verdict: gap to BERT 0.46 is the frozen representation, not data -> fine-tuning genuinely required.
- 2026-06-19 02:30 Phase 4 (FT) setup: transformers-no-hpo preset -> bert scorer. Needed `uv add accelerate`
  (HF Trainer dep, omitted by sentence-transformers extra). Use bert-base-uncased (cached/offline + ==
  literature baseline). deberta-v3-small NOT cached.
- 2026-06-19 02:50 BLOCKER: stock recipe (lr 3e-5) COLLAPSES at cap-100 -> degenerate all-positive after
  thresholding, decision_f1=0.107 (< frozen-e5 0.316). Diagnosis (diag_ft.py, standalone bert + threshold
  sweep): BCE plateaus at the base-rate floor ~0.17, sigmoid outputs near-constant across classes.
  Recipe sweep (full epochs, no early stop): lr 1e-5 collapses (macro-F1 0.03), lr 3e-5 collapses,
  **lr 2e-5 learns (macro-F1 0.224 @ thr 0.1, micro 0.31)** — 2e-5 is the sweet spot. Warmup 0.1 does NOT
  help (0.178). Also: the preset early-stops on scoring_f1@0.5, which is ~0 for sparse multilabel (probs<0.5)
  -> stops at step 1 and restores a near-random model; MUST disable (run_finetune.py disable_early_stopping).
- 2026-06-19 03:15 FINAL recipe: bert-base, lr 2e-5, 15 epochs, no early stop, n_trials 5 (bert trains once;
  trials tune decision only). KEY (pre-result): even tuned, FT @cap-100 ~0.22 << frozen-e5 0.316 -> at small
  balanced data, fine-tuning bert-base from scratch LOSES to a linear head on frozen e5. FT needs more data.
