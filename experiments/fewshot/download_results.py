"""Download few-shot results from Weights & Biases (public projects of samoed-roman).

Writes two raw CSVs into ./data:
  - few_shot_results.csv      (AutoIntent runs, project new_autointent_few_shot2)
  - automl_eval_results.csv   (AutoML baselines, project AutoML-Eval)

Auth: set WANDB_API_KEY (or WB_API_KEY) in the environment, or place it in a
.env file in this folder or any parent directory.

Run:  uv run --no-project --with wandb --with pandas python download_results.py
"""

import os
from collections import defaultdict
from pathlib import Path

import pandas as pd
import wandb

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

AUTOINTENT_PROJECT = "samoed-roman/new_autointent_few_shot2"
AUTOML_PROJECT = "samoed-roman/AutoML-Eval"


def resolve_api_key() -> str:
    for var in ("WANDB_API_KEY", "WB_API_KEY"):
        if os.environ.get(var):
            return os.environ[var]
    for folder in (HERE, *HERE.parents):
        env = folder / ".env"
        if env.is_file():
            for line in env.read_text().splitlines():
                line = line.strip()
                for var in ("WANDB_API_KEY=", "WB_API_KEY="):
                    if line.startswith(var):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("Set WANDB_API_KEY / WB_API_KEY (env or .env file)")


def dictify(value):
    if hasattr(value, "items"):
        return {k: dictify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [dictify(v) for v in value]
    return value


def download_autointent(api: "wandb.Api") -> pd.DataFrame:
    runs = api.runs(AUTOINTENT_PROJECT, filters={"displayName": "final_metrics"})
    results: dict = defaultdict(dict)
    for run in runs:
        dataset, few_shot = run.group.rsplit("_", 1)
        few_shot = int(few_shot) if few_shot != "None" else "full"
        results[dataset][few_shot] = {
            "config": dictify(run.config),
            **{
                k: dictify(v)
                for k, v in run.summary.items()
                if k not in ("_wandb", "_timestamp", "_step")
            },
        }
    df = pd.concat({ds: pd.DataFrame(rows).T for ds, rows in results.items()})
    return df.rename_axis(["dataset", "few_shot"])


def download_automl(api: "wandb.Api") -> pd.DataFrame:
    runs = api.runs(AUTOML_PROJECT)
    results: dict = defaultdict(lambda: defaultdict(dict))
    for run in runs:
        metrics = {}
        for k, v in run.summary.items():
            if k in ("accuracy", "_runtime"):
                metrics[k] = v
            if k in ("macro avg", "weighted avg"):
                for sub_k, sub_v in v.items():
                    metrics[f"{k}_{sub_k}"] = sub_v
        results[run.config["dataset"]][run.config["framework"]][
            run.config.get("few_shot", "full")
        ] = {"config": dictify(run.config), **metrics}
    df = pd.concat(
        {
            ds: pd.concat({fw: pd.DataFrame(rows).T for fw, rows in fws.items()})
            for ds, fws in results.items()
        }
    )
    df.index.names = ["dataset", "framework", "few_shot"]
    return df


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_API_KEY"] = resolve_api_key()
    os.environ.setdefault("WANDB_SILENT", "true")
    api = wandb.Api()

    ai = download_autointent(api)
    ai.to_csv(DATA / "few_shot_results.csv")
    print(f"[autointent] {len(ai)} rows -> {DATA / 'few_shot_results.csv'}")

    ml = download_automl(api)
    ml.to_csv(DATA / "automl_eval_results.csv")
    print(f"[automl] {len(ml)} rows -> {DATA / 'automl_eval_results.csv'}")


if __name__ == "__main__":
    main()
