"""Проверка 6.5: устойчивость качества при малом объеме обучающих данных.

Состав повторяет опубликованный эксперимент: три набора данных,
n на класс из {4, 8, 16, 32, 64, 128} плюс полный обучающий набор,
один прогон на точку с фиксированным начальным значением генератора.
Итого 21 строка журнала (3 набора x 7 точек).

Своей реализации подвыборки здесь нет: используется штатный режим образца
DataConfig(is_few_shot_train=True, examples_per_intent=n) — тот же, которым
пользовался опубликованный эксперимент (sweep-whole-search-space/quality_scorers.py,
строка 126).
"""

import argparse
import json
from pathlib import Path

from autointent import Dataset, Pipeline
from autointent.configs import DataConfig
from autointent.metrics import decision_accuracy

DATASETS = ["hwu64", "minds14", "snips"]
SHOTS: list[int | None] = [4, 8, 16, 32, 64, 128, None]  # None — полный обучающий набор
# Начальное значение генератора — значение по умолчанию Pipeline.from_preset
# (_pipeline/_pipeline.py:124, seed=42). Тот же сид Pipeline.fit передает в Context и
# далее в DataHandler (_pipeline.py:275 `context.set_dataset(dataset, self.data_config)`),
# поэтому им же отбираются n примеров на класс. То же значение задано в подвыборке
# базовых фреймворков (create_few_shot_split(..., random_seed=42)). Механизмы отбора у
# образца и у базовых фреймворков разные, тождественность подвыборок не гарантируется;
# фиксируются объем n, сид каждой стороны и неизменность тестовой части.
SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--preset", default="classic-light")
    args = parser.parse_args()

    results: dict[str, dict[str, float]] = {}
    raw: dict[str, dict[str, float]] = {}
    for name in DATASETS:
        for n in SHOTS:
            # Набор перечитывается на каждой точке намеренно, а не один раз на набор.
            # Pipeline.fit передает объект в DataHandler, который присваивает его без
            # копирования (context/data_handler/_data_handler.py:50) и делит на месте:
            # строка 230 при отсутствии готового валидационного сплита выполняет
            # _split_validation_from_train(...), строка 232 — _split_few_shot(...).
            # Общий объект означал бы, что вторая и последующие точки обучаются на
            # подвыборке первой (n=4), то есть все семь точек по n перестали бы
            # различаться. Перечитывание дает каждой точке нетронутый набор и верно
            # независимо от того, есть ли в зеркале готовый валидационный сплит.
            dataset = Dataset.from_json(args.datasets_dir / f"{name}.json")
            pipeline = Pipeline.from_preset(args.preset, seed=SEED)
            if n is not None:
                # Подвыборка выполняется штатным механизмом образца, а не собственной
                # реализацией: DataHandler отбирает n примеров на класс с сидом пайплайна.
                pipeline.set_config(DataConfig(is_few_shot_train=True, examples_per_intent=n))
            pipeline.fit(dataset)
            preds = pipeline.predict(dataset["test"]["utterance"])
            acc = 100 * decision_accuracy(dataset["test"]["label"], preds)
            key = "full" if n is None else str(n)
            raw.setdefault(name, {})[key] = acc
            results.setdefault(name, {})[key] = round(acc, 2)
            print(f"{name} n={key} accuracy={acc:.2f}", flush=True)

    # Деградация считается по неокругленным значениям: округление разности округленных
    # величин расходится со справочными числами ПМИ на 0,01 п.п. (hwu64 дал бы 3,99 вместо 4,00).
    for name, per_n in results.items():
        per_n["degradation_at_16"] = round(raw[name]["full"] - raw[name]["16"], 2)

    print("SUMMARY:", json.dumps(results, indent=2))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
