"""Проверка 6.3: качество многоклассовой классификации намерений.

Два пресета × пять наборов данных × три начальных значения генератора,
результаты усредняются по запускам.
"""

import argparse
import json
from pathlib import Path

from autointent import Dataset, Pipeline
from autointent.metrics import decision_accuracy

DATASETS = ["banking77", "hwu64", "massive", "minds14", "snips"]
PRESETS = ["classic-light", "classic-medium"]
SEEDS = [0, 1, 2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    results: dict[str, dict[str, list[float]]] = {}
    for preset in PRESETS:
        for name in DATASETS:
            for seed in SEEDS:
                # Набор перечитывается на каждом запуске намеренно, а не один раз на
                # (пресет, набор). Pipeline.fit передаёт объект в DataHandler, который
                # присваивает его без копирования (context/data_handler/_data_handler.py:51)
                # и делит на месте: при scheme="ho" (умолчание, в пресетах нет data_config)
                # строка 298 выполняет
                # `self.dataset[TRAIN], self.dataset[VALIDATION] = split_dataset(...)`.
                # Перед этим строка 216 вычисляет has_validation_split, и на строках 229—230
                # деление пропускается, если валидационный сплит уже есть. Общий объект дал бы
                # деление только на seed=0, а seed=1 и seed=2 обучались бы на его сплите —
                # три запуска перестали бы быть независимыми, а усреднение по ним потеряло бы
                # смысл. Перечитывание даёт каждому сиду нетронутый набор и верно независимо
                # от того, есть ли в зеркале готовый валидационный сплит.
                dataset = Dataset.from_json(args.datasets_dir / f"{name}.json")
                pipeline = Pipeline.from_preset(preset, seed=seed)
                pipeline.fit(dataset)
                preds = pipeline.predict(dataset["test"]["utterance"])
                acc = decision_accuracy(dataset["test"]["label"], preds)
                results.setdefault(preset, {}).setdefault(name, []).append(acc)
                print(f"{preset} {name} seed={seed} accuracy={acc * 100:.2f}", flush=True)

    summary = {
        preset: {
            **{name: round(100 * sum(v) / len(v), 2) for name, v in per_dataset.items()},
            "avg": round(100 * sum(sum(v) / len(v) for v in per_dataset.values()) / len(per_dataset), 2),
        }
        for preset, per_dataset in results.items()
    }
    print("SUMMARY:", json.dumps(summary, indent=2))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
