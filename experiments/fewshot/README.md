# Few-shot multiclass experiment

Compares **AutoIntent** (`classic-light`) against the **AutoGluon** and **H2O**
AutoML baselines on three intent-classification datasets (`hwu64`, `minds14`,
`snips`) as the training set is subsampled to `n` examples per class
(`n ∈ {4, 8, 16, 32, 64, 128}`). It reproduces the few-shot figure used in the
thesis: AutoIntent stays high from very few shots, while the baselines only
catch up at 32–128 shots.

The raw metrics were logged to public Weights & Biases projects by
`samoed-roman`. These scripts pull them back down, merge them, and plot them.

## Where the experiments were actually run

This folder only **downloads and visualises** results. The code that *ran* the
experiments lives in [`../sweep-whole-search-space/`](../sweep-whole-search-space/):

- `automl_frameworks.py` — runs the baselines and logs them to the `AutoML-Eval`
  W&B project (`evaluate_gluon` = AutoGluon, `evaluate_h2o` = H2O AutoML + Word2Vec,
  `evaluate_lama` = LightAutoML, `evalute_fedot`, `evaluate_gama`).
- `tune_bert.py`, `quality_*.py`, `scoring_*.py`, the seed-aggregation scripts — the
  AutoIntent sweeps and analysis.
- `plots/few_shot.ipynb` and `results_download/` — the original notebooks this
  folder is a cleaned-up, scripted version of.

## Scripts

| Script | Reads | Writes | Purpose |
| --- | --- | --- | --- |
| `download_results.py` | W&B | `data/few_shot_results.csv`, `data/automl_eval_results.csv` | Download AutoIntent runs (`new_autointent_few_shot2`) and AutoML baselines (`AutoML-Eval`). |
| `build_csv.py` | the two raw CSVs | `data/comparison_few_shot.csv` | Merge into one tidy table: `dataset, framework, few_shot, f1, precision, recall, accuracy`. Keeps AutoIntent / AutoGluon / H2O; drops the `few_shot=2` probe runs. |
| `plot.py` | `data/comparison_few_shot.csv` | `figures/few_shot_accuracy.{png,svg}` | One subplot per dataset, all frameworks overlaid, **accuracy only**. The `full` (full-training-set) point is excluded — this is a few-shot study. |

`data/` and `figures/` are checked in, so you can run `build_csv.py` / `plot.py`
without W&B access. Re-run `download_results.py` only to refresh from W&B.

## Running

Dependencies are pulled on the fly with [`uv`](https://docs.astral.sh/uv/) — no
project env needed. Run from this folder, in order:

```bash
# 1. download (needs a W&B API key, see below)
uv run --no-project --with wandb --with pandas python download_results.py

# 2. merge
uv run --no-project --with pandas python build_csv.py

# 3. plot
uv run --no-project --with pandas --with matplotlib --with seaborn python plot.py
```

### W&B authentication

`download_results.py` reads `WANDB_API_KEY` (or `WB_API_KEY`) from the
environment, or from a `.env` file in this folder or any parent directory:

```
WB_API_KEY=<your key>
```

The source projects are public, so any valid key works. Do **not** commit the
`.env` file. `build_csv.py` and `plot.py` need no key.
