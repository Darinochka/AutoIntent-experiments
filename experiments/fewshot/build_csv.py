"""Merge raw W&B exports into a single comparison table.

Reads  ./data/few_shot_results.csv  and  ./data/automl_eval_results.csv
Writes ./data/comparison_few_shot.csv with columns:
  dataset, framework, few_shot, f1, precision, recall, accuracy

Frameworks kept: AutoIntent, AutoGluon (gluon), H2O (h2o). The few_shot=2
probe runs are dropped. The "full" point is kept here (it is filtered out at
plot time, see plot.py).

Run:  uv run --no-project --with pandas python build_csv.py
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

COLS = ["dataset", "framework", "few_shot", "f1", "precision", "recall", "accuracy"]


def main() -> None:
    ai = pd.read_csv(DATA / "few_shot_results.csv")
    ai = ai.rename(
        columns={
            "decision_accuracy": "accuracy",
            "decision_f1": "f1",
            "decision_precision": "precision",
            "decision_recall": "recall",
        }
    )
    ai["framework"] = "AutoIntent"
    ai = ai[COLS]

    ml = pd.read_csv(DATA / "automl_eval_results.csv")
    ml = ml.rename(
        columns={
            "macro avg_f1-score": "f1",
            "macro avg_precision": "precision",
            "macro avg_recall": "recall",
        }
    )
    ml = ml[ml["framework"].isin(["gluon", "h2o"]) & (ml["few_shot"].astype(str) != "2")]
    ml["framework"] = ml["framework"].replace({"gluon": "AutoGluon", "h2o": "H2O"})
    ml = ml[ml["dataset"].isin(ai["dataset"].unique())]
    ml = ml[COLS]

    comp = pd.concat([ai, ml]).dropna(subset=["few_shot"]).reset_index(drop=True)
    comp = comp.sort_values(by=["dataset", "framework", "few_shot"])
    out = DATA / "comparison_few_shot.csv"
    comp.to_csv(out, index=False)
    print(f"{len(comp)} rows -> {out}")


if __name__ == "__main__":
    main()
