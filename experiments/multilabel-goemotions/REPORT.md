# AutoIntent Multilabel Capability — Results Report (GoEmotions)

**Run:** autonomous overnight, 2026-06-16, 00:45–03:15.
**Setup:** AutoIntent 0.3.0, preset `classic-light`, embedder `intfloat/multilingual-e5-large-instruct`,
device `mps`. GoEmotions `simplified`, 28 classes, genuinely-but-sparsely multilabel (mean 1.17
labels/example). Two-phase design per `DESIGN.md`: Phase 1 picks `(scoring, decision)` target metrics on
the `validation` split; Phase 2 reports the capability curve on the disjoint `test` split. Headline metric:
**macro `decision_f1`** (every class weighted equally, so the rare tail dominates — the multilabel ability
we want to probe). `decision_accuracy` ignored (≈0.92 all-negative baseline).

---

## 1. Capability verdict (TL;DR)

- **AutoIntent's `classic-light` AutoML does learn this 28-class sparse-multilabel task.** Macro-F1 rises
  monotonically with per-class budget, from **0.152 (5-shot) to 0.316 (100-shot)** on the held-out test
  split, and is **still climbing at 100-shot** (no plateau) — per-class data is the binding constraint.
- **Balanced per-class data is far more sample-efficient than raw volume.** At a matched ~2.7k training
  rows, balanced data scores **0.316** vs imbalanced **0.248** (+0.068 macro-F1). Imbalanced data needs
  **14.1k** rows to reach just 0.281 — still below what balanced data achieves with 2.5k.
- **AutoIntent adapts its pipeline to the regime.** It selects `mlknn` at tiny N, switches to `linear`
  (logreg on e5 embeddings) for balanced moderate N, and `knn` for imbalanced data — and the *optimal
  target metric differs by balance* (see §4), vindicating the design's per-balance selection.
- **The optimizer over-predicts** (recall > precision throughout) — it favors coverage of the rare tail
  at the cost of precision, which is the right bias for macro-F1 on a long tail.

---

## 2. Primary result — classwise capability curve (balanced, eval=test)

Frozen target metrics: **`scoring_neg_coverage` + `decision_f1`**. `classwise-N` = ≤N samples/class.

| shots | train rows | seeds | **macro-F1** | ±std | precision | recall | selected scorer/decider |
|------:|-----------:|:-----:|:------------:|:----:|:---------:|:------:|:------------------------|
| 5   | 127  | 5 | **0.152** | 0.009 | 0.121 | 0.326 | mlknn / adaptive |
| 10  | 255  | 5 | **0.174** | 0.008 | 0.163 | 0.275 | mlknn / adaptive |
| 25  | 633  | 3 | **0.217** | 0.020 | 0.205 | 0.389 | linear / adaptive |
| 50  | 1261 | 3 | **0.269** | 0.009 | 0.223 | 0.509 | linear / adaptive |
| 100 | 2475 | 3 | **0.316** | 0.008 | 0.263 | 0.477 | linear / adaptive |

- **Monotonic, low variance.** Tight std (≤0.02) even at 5/10-shot. `classwise-5` did **not** fail (the
  design's feared "rare class absent from carved HPO-val" did not materialize for these seeds).
- **Module crossover at N≥25:** `mlknn` (a multilabel-native KNN) wins when data is scarce; `linear`
  (one-vs-rest logreg on e5 embeddings) takes over once there are ~25+/class. A real capability signal:
  with enough balanced data, a simple linear head on strong embeddings is AutoIntent's best choice here.
- Absolute numbers look low because the metric is **macro** on a 28-class long tail (test's rarest class
  has only 6 examples); this is intentional, not underperformance.

## 3. Realism anchor — stratified curve (imbalanced, eval=test)

Frozen target metrics: **`scoring_precision` + `decision_f1`**. `stratified-N` = natural proportions with a
floor of N/class (so total size grows with N). Capped at floor-25 (~14k) to protect the machine.

| floor | train rows | seeds | **macro-F1** | ±std | precision | recall | selected scorer/decider |
|------:|-----------:|:-----:|:------------:|:----:|:---------:|:------:|:------------------------|
| 5  | 2,819  | 5 | **0.248** | 0.013 | 0.228 | 0.307 | mlknn / adaptive |
| 10 | 5,658  | 5 | **0.263** | 0.015 | 0.252 | 0.314 | mlknn / threshold |
| 25 | 14,121 | 3 | **0.281** | 0.008 | knn / threshold | | |

(precision≈recall here, unlike classwise's recall-heavy predictions.)

### The balance comparison — done honestly (size-controlled)

**Naïve, shots-matched comparison is misleading** and must not be used: `stratified-5` (0.248) appears to
beat `classwise-5` (0.152), but only because `stratified-5` carries **2,819** training rows vs
`classwise-5`'s **127** (a 22× data advantage). This is the exact size≠balance confound `DESIGN §8` warns
about.

**Size-matched comparison (the valid one):** at ~2.7k training rows,
- balanced `classwise-100` (2,475 rows) = **0.316**
- imbalanced `stratified-5` (2,819 rows) = **0.248**

→ **balancing the per-class budget is worth ~+0.07 macro-F1 at fixed data volume**, and imbalanced data is
dramatically less efficient (14.1k imbalanced rows reach only 0.281). Because macro-F1 weights the rarest
classes equally, guaranteeing each class a budget is what drives the gap. This is a *directional* result
(the two regimes still differ in composition, not just balance), but the direction is unambiguous.

## 4. Target-metric selection (Phase 1, eval=validation)

Per-balance selection over the 9 scoring × 3 decision grid (`decision_accuracy` dropped as degenerate),
2 seeds. **The winners differ by balance — the design's per-balance selection was the right call:**

| balance | frozen scoring | frozen decision | winning scorer | notes |
|---|---|---|---|---|
| classwise  | `scoring_neg_coverage` | `decision_f1` | linear | ties `neg_ranking_loss` exactly; clear winner |
| stratified | `scoring_precision`    | `decision_f1` | mlknn/knn | top-4 within 1 std → **within-noise** |

- **The decision target barely matters.** In both balances, `decision_f1/precision/recall` produced near-
  identical results for a given scorer (the deciders self-tune thresholds). The **scoring** metric and the
  **scorer module** are what move the needle.
- `decision_f1` was chosen as the freeze in both (aligned with the headline metric; it tied or led).

---

## 5. How these results compare to the GoEmotions literature

Our headline (macro `decision_f1`) is directly comparable to the published GoEmotions macro-F1: **same
28-class taxonomy, same held-out `test` split**. The difference is purely on the *training* side — published
baselines **fully fine-tune a transformer on all ~43,410 training rows**, whereas AutoIntent here uses a
**frozen e5 embedder + a shallow linear/KNN head on ≤2,475 balanced rows (~6% of the data)**.

| Approach | Train | Fine-tuned | macro-F1 | micro-F1 |
|---|---|:--:|:--:|:--:|
| BiLSTM — original baseline [1] | full ~43k | — | 0.41 | — |
| **BERT-base — original baseline [1]** | full ~43k | full | **0.46** | 0.51 |
| RoBERTa + psycholinguistic features (as reported) [3,4] | full | full | ~0.59 | — |
| PK-GAT (recent, as reported) [4] | full | full | 0.587 | 0.699 |
| **AutoIntent `classic-light` (this run)** | **~2.5k balanced** | none (frozen e5) | **0.316** | n/a |

(A "RoBERTa 0.85 macro-F1" figure circulates online but is **not** a credible 28-class macro number — likely
accuracy or a coarse sentiment grouping. The real 28-class macro range is ~0.41–0.59.)

**Verdict — good for what it is, not vs the leaderboard.** As an absolute number, 0.316 is ~0.15 below
fine-tuned BERT and ~0.27 below SOTA. But it reaches **~69% of fine-tuned BERT's macro-F1 with ~6% of the
data and zero fine-tuning**, and the curve had not plateaued at 100-shot — good sample efficiency from a
no-GPU AutoML pipeline. AutoIntent is not built to win this leaderboard; it is built to produce a sensible
classifier cheaply, which it does.

**Do you need transformer fine-tuning?** To *maximize* macro-F1, yes — every number above 0.46 comes from a
fine-tuned transformer, because fine-grained emotion is subtle and a frozen *retrieval* embedder (e5, tuned
for semantic similarity) does not emphasize the emotion-discriminative features a shallow head could exploit.
**However, this run cannot yet prove FT is *necessary*:** it conflates two effects (below) because we capped
training at ~2.5k and never measured AutoIntent's frozen-embedding ceiling on the full data.

**Where the 0.316 → 0.46 gap comes from (decomposed):**
1. **Data starvation (largest, most certain).** ~6% of the training data; the curve is still rising.
2. **Frozen, task-agnostic embeddings.** No task adaptation of the representation — the structural ceiling.
3. **Macro-F1 on a brutal tail.** Rare classes (`grief` ~77 even in full data) score ~0 even for SOTA; our
   test tail's rarest class has 6 examples → high-variance, drags the macro average down.
4. **Multilabel thresholding.** Low precision (0.12–0.26) with higher recall → over-prediction across 28
   classes; fine-tuned models with learned per-class thresholds handle this better.
5. **Preset capacity.** `classic-light` is KNN/linear/mlknn only; `classic-medium/heavy` or an FT preset
   has more headroom.

Causes (1) and (2) dominate and are entangled; the runs in §8 are designed to separate them. Rough
prediction: full-data frozen e5 + linear likely reaches ~**0.38–0.43** (closing most of the gap to BERT but
stalling short), leaving the final ~0.05–0.15 up to SOTA as the part that genuinely needs fine-tuning.

## 6. Threats to validity & honest caveats

1. **Stratified is truncated at floor-25 (14k rows).** Floors 50 (28k) and 100 (full 43k) were excluded:
   a single 28k-row fit ran >17 min and risked the 17 GB machine. So there is **no "stratified full-data"
   point**; the stratified curve stops at 14k. The classwise (primary) claim is unaffected.
2. **Cross-balance is directional only.** The size-matched comparison in §3 is the valid one; the
   shots-matched view is a data-volume artifact and is explicitly *not* used for conclusions.
3. **Stratified metric selection is within-noise** (top-4 scoring pairs within one std at 2 seeds). The
   frozen stratified pair should be read as "a reasonable choice," not "the proven best."
4. **Phase-1 economy:** 2 seeds; classwise metric frozen from sizes {25,50}, stratified from floor-5 only
   (no cross-size stability sweep for stratified). Applied across the whole ladder.
5. **Thin test tail:** test's rarest class has 6 examples, so the macro-F1 tail contribution is
   high-variance; seeds average *model* noise but cannot add eval examples.
6. **Sparse multilabel:** mean 1.17 labels/example — the genuinely-multilabel signal is a minority slice.
7. **Single embedder/preset (`classic-light` + e5-large, mps).** Not a statement about transformer-FT
   presets or other embedders.
8. **Selection vs reporting are disjoint splits** (validation → test), so headline numbers are not
   selected-on — the one good-news caveat. Per-run HPO (module + thresholds) is tuned only on the
   HPO-validation AutoIntent carves from train.

## 7. Operational finding (worth fixing upstream)

**AutoIntent 0.3.0 calls `huggingface_hub.model_info()` on every `embed()`** (to key its embeddings cache
by the model's remote commit hash, `_wrappers/embedder/sentence_transformers.py:61`). Under a fast sweep
this floods HF's **1000-requests/5-min** quota → HTTP 429 (compounded here by other worktree agents sharing
the IP), and it is **fatal under `HF_HUB_OFFLINE`**. It silently failed 14 Phase-2 cells before diagnosis.
**Fix applied locally** in `src/pipeline.py::_patch_embedder_offline_cache_key`: resolve the commit hash
from the cached `refs/main` (identical to the remote sha, so the cache key is unchanged) and fall back to
the model name — no network. All sweeps then ran with `HF_HUB_OFFLINE=1`. Recommend upstreaming a local-
first hash resolver to AutoIntent.

## 8. Suggested next runs (not done tonight)

To close the leaderboard gap and answer "is fine-tuning needed" (§5), in priority order:

- **`classic-light` on the FULL 43k train (frozen, no subsample)** — measures AutoIntent's frozen-embedding
  *ceiling* against the 0.46 fine-tuned BERT baseline. If it lands ~0.40–0.45, the gap was mostly data and
  FT buys little; if it plateaus ~0.33–0.37, that is the frozen-embedding wall and FT is genuinely required.
  Needs a GPU / high-RAM box — the 43k-row fit ran >17 min/cell on this 17 GB MPS machine.
- **An AutoIntent fine-tuning preset** (cross-encoder / fine-tuned scoring node, if available) on full data —
  directly quantifies the FT lift over frozen embeddings.
- Complete the stratified curve at floors 50/100 (full data) on the same bigger box, for a true
  balance-vs-volume crossover.
- Re-confirm the stratified frozen pair with more seeds / a cross-size stability check (it's within-noise).
- A size-matched `natural` ablation (same total size as classwise-N, natural proportions) for a clean
  balance-only contrast at several points, not just the single ~2.7k overlap.

## References

1. Demszky, Movshovitz-Attias, Ko, Cowen, Nemade, Ravi. *GoEmotions: A Dataset of Fine-Grained Emotions.*
   ACL 2020 — original 28-class taxonomy and the BERT / BiLSTM baselines (macro-F1 0.46 / 0.41).
   https://aclanthology.org/2020.acl-main.372/
2. *Uncovering the Limits of Text-based Emotion Detection.* arXiv 2109.01900 — analysis of the GoEmotions
   ceiling and per-class limits. https://arxiv.org/pdf/2109.01900
3. *Large Language Models on Fine-grained Emotion Detection Dataset with Data Augmentation and Transfer
   Learning.* arXiv 2403.06108 — recent transfer-learning / augmentation results. https://arxiv.org/html/2403.06108v1
4. *Improving Fine-Grained Emotion Detection in Text with BERT and GoEmotions: An Experimental Study.*
   Premier Science — recent fine-tuned baselines and the ~0.59 / PK-GAT figures cited as reported.
   https://premierscience.com/pjs-25-1204/
5. GoEmotions dataset overview (secondary summary). https://www.emergentmind.com/topics/goemotions-dataset

*Numbers above 0.46 are quoted as reported in the cited secondary/primary sources and were not independently
re-run; treat them as the published landscape, not verified-by-us figures.*

---

*Artifacts: `logs/sweep_summary_{val,test}.csv` (aggregated), `logs/sweep_runs_{val,test}.csv` (per-seed),
`logs/gm-*_metrics.json` (per-run full reports). Plan & live log: `PLAN.md`.*
