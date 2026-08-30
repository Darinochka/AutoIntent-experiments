"""Базовые AutoML-фреймворки в проверке 6.4: детекция внедоменных запросов.

Своей реализации моделей в этом файле нет. Используются функции опубликованного
модуля ../sweep-whole-search-space/automl_frameworks.py:

  load_data              загрузка набора с хаба; примерам без метки (внедоменным)
                         присваивается дополнительный класс max_label + 1
  evaluate_gluon         AutoGluon high_quality_hpo на deberta-v3-small
  evaluate_h2o           H2O AutoML поверх word2vec, seed=42
  evaluate_oos_accuracy  внутридоменная accuracy, recall и precision по классу
                         внедоменных запросов (порядок возврата именно такой,
                         подтвержден чтением automl_frameworks.py:44-55)

Бинарная F1 по классу «внедоменный» в опубликованном коде не считается: она
пересчитывается здесь из precision и recall — так же, как получены справочные
значения ПМИ из таблицы 5 опубликованного отчета (AutoGluon 98,47 и 32,20 -> 48,53;
H2O 82,57 и 27,00 -> 40,69).

Запуск — по одному фреймворку на вызов, окружения фреймворков несовместимы:
    WANDB_MODE=offline uv run --no-project --python 3.12 --with pandas --with datasets \
        --with scikit-learn --with wandb --with 'autogluon>=1.3.0' \
        python baselines_oos.py --framework gluon-high \
            --dataset DeepPavlov/clinc150 --out runs/6.4/baselines.json
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

SWEEP_DIR = Path(__file__).resolve().parents[2] / "sweep-whole-search-space"


def load_published(module_name: str):
    """Импортировать модуль опубликованного эксперимента по пути к файлу."""
    path = SWEEP_DIR / f"{module_name}.py"
    if not path.is_file():
        raise SystemExit(f"не найден опубликованный модуль {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def as_percent(value: float) -> float:
    """Доля -> проценты; отсутствие предсказаний класса дает 0, а не NaN."""
    return 0.0 if pd.isna(value) else round(100 * float(value), 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", required=True, choices=["gluon-high", "h2o"])
    parser.add_argument("--dataset", default="DeepPavlov/clinc150")
    parser.add_argument("--out", type=Path, required=True, help="файл журнала метрик")
    args = parser.parse_args()

    frameworks = load_published("automl_frameworks")
    evaluate = {
        "gluon-high": frameworks.evaluate_gluon,
        "h2o": frameworks.evaluate_h2o,
    }[args.framework]

    train_df, test_df = frameworks.load_data(args.dataset)
    test_df = test_df.reset_index(drop=True)
    # Приведение как в main() опубликованного модуля: предсказание сводится к
    # целочисленному идентификатору класса, внедоменный класс — max_label + 1.
    predictions = pd.Series(evaluate(train_df, test_df)).round(0).astype(int).reset_index(drop=True)

    # Порядок возврата evaluate_oos_accuracy подтвержден чтением
    # automl_frameworks.py:44-55: accuracy, recall, precision.
    in_domain_acc, oos_recall, oos_precision = frameworks.evaluate_oos_accuracy(predictions, test_df)
    in_domain_acc = as_percent(in_domain_acc)
    oos_recall = as_percent(oos_recall)
    oos_precision = as_percent(oos_precision)
    denominator = oos_precision + oos_recall
    oos_f1 = 0.0 if denominator == 0 else round(2 * oos_precision * oos_recall / denominator, 2)

    print(
        f"{args.dataset} {args.framework} in_domain_accuracy={in_domain_acc:.2f} "
        f"oos_binary_f1={oos_f1:.2f}",
        flush=True,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    records = json.loads(args.out.read_text(encoding="utf-8")) if args.out.is_file() else []
    records.append(
        {
            "dataset": args.dataset,
            "framework": args.framework,
            "in_domain_accuracy": in_domain_acc,
            "oos_precision": oos_precision,
            "oos_recall": oos_recall,
            "oos_binary_f1": oos_f1,
        }
    )
    args.out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
