"""Проверка 6.4: детекция внедоменных запросов (OOS).

Обучение пресетом `classic-light` с настройкой порога в модуле принятия решения,
измерение внутридоменной accuracy и бинарной F1 по классу «внедоменный против
внутридоменного» на наборе clinc150 из офлайн-зеркала.
"""

import argparse
import json
from pathlib import Path

from sklearn.metrics import f1_score

from autointent import Dataset, Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    dataset = Dataset.from_json(args.datasets_dir / "clinc150.json")
    pipeline = Pipeline.from_preset("classic-light", seed=42)
    pipeline.fit(dataset)

    utterances = dataset["test"]["utterance"]
    y_true = dataset["test"]["label"]
    y_pred = pipeline.predict(utterances)

    # Внедоменный запрос кодируется значением None в поле метки.
    oos_true = [1 if y is None else 0 for y in y_true]
    oos_pred = [1 if y is None else 0 for y in y_pred]
    f1_oos = f1_score(oos_true, oos_pred)

    in_domain = [(t, p) for t, p in zip(y_true, y_pred) if t is not None]
    acc_in_domain = sum(t == p for t, p in in_domain) / len(in_domain)

    summary = {
        "in_domain_accuracy": round(100 * acc_in_domain, 2),
        "oos_binary_f1": round(100 * f1_oos, 2),
        "n_test": len(y_true),
        "n_oos_true": sum(oos_true),
    }
    print("SUMMARY:", json.dumps(summary, indent=2))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
