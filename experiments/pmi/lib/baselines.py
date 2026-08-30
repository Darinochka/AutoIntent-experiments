"""Прогон базовых AutoML-фреймворков поверх опубликованного кода эксперимента.

Своей реализации моделей в этом файле нет. Используются функции из
../sweep-whole-search-space:

  frameworks_duration.load_data             загрузка набора с хаба, метка для OOS
  frameworks_duration.create_few_shot_split подвыборка n примеров на класс,
                                            random_seed=42
  frameworks_duration.evaluate_glueon       AutoGluon, пресет не задается
                                            (умолчание), time_limit=600
  frameworks_duration.evaluate_h2o          H2O AutoML поверх word2vec, seed=42
  automl_frameworks.evaluate_gluon          AutoGluon high_quality_hpo
  automl_frameworks.evaluate_lama           LightAutoML (TabularNLPAutoML)

Известный дефект опубликованной реализации LightAutoML: evaluate_lama возвращает
np.argmax по столбцам предсказания, то есть номер столбца во внутреннем порядке
классов LightAutoML, а не метку набора данных. Обертка сопоставление не
исправляет (опубликованный код не переписывается), поэтому accuracy строки lama
мерой качества классификации не является: именно этот дефект дал 6,59 в таблице 4
опубликованного отчета. Значение вносится в протокол справочно с этой оговоркой,
в критерий приемки 6.3 LightAutoML не входит.

Функции main() опубликованных модулей не вызываются: они пишут результат в
Weights & Biases. Здесь accuracy считается локально и дописывается в JSON.

Запуск — по одному фреймворку на вызов, окружения фреймворков несовместимы:
    WANDB_MODE=offline uv run --no-project --with pandas --with datasets \\
        --with scikit-learn --with wandb --with 'autogluon>=1.3.0' \\
        python baselines.py --framework gluon-default --dataset DeepPavlov/snips \\
            --out runs/6.3/baselines.json
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score

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


def get_evaluator(framework: str):
    """Вернуть модуль с загрузкой данных и функцию обучения нужного фреймворка."""
    duration = load_published("frameworks_duration")
    if framework == "gluon-default":
        return duration, duration.evaluate_glueon
    if framework == "h2o":
        return duration, duration.evaluate_h2o
    frameworks = load_published("automl_frameworks")
    if framework == "gluon-high":
        return duration, frameworks.evaluate_gluon
    if framework == "lama":
        return duration, frameworks.evaluate_lama
    raise SystemExit(f"неизвестный фреймворк {framework}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--framework", required=True, choices=["gluon-default", "gluon-high", "h2o", "lama"]
    )
    parser.add_argument("--dataset", required=True, help="репозиторий набора данных на хабе")
    parser.add_argument(
        "--few-shot", type=int, default=None,
        help="число примеров на класс; без этого параметра берется полный обучающий набор",
    )
    parser.add_argument("--out", type=Path, required=True, help="файл журнала метрик")
    args = parser.parse_args()

    duration, evaluate = get_evaluator(args.framework)
    train_df, test_df = duration.load_data(args.dataset)
    if args.few_shot is not None:
        train_df = duration.create_few_shot_split(
            train_df, examples_per_label=args.few_shot, random_seed=42
        )

    predictions = pd.Series(evaluate(train_df, test_df)).astype(int).reset_index(drop=True)
    gold = test_df["label"].astype(int).reset_index(drop=True)
    accuracy = 100 * accuracy_score(gold, predictions)

    shots = "full" if args.few_shot is None else str(args.few_shot)
    print(f"{args.dataset} {args.framework} n={shots} accuracy={accuracy:.2f}", flush=True)
    if args.framework == "lama":
        print(
            "ВНИМАНИЕ: опубликованная реализация LightAutoML возвращает номер столбца во "
            "внутреннем порядке классов, а не метку набора данных; приведенная accuracy мерой "
            "качества классификации не является и в критерий приемки 6.3 не входит",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    records = json.loads(args.out.read_text(encoding="utf-8")) if args.out.is_file() else []
    records.append(
        {
            "dataset": args.dataset,
            "framework": args.framework,
            "few_shot": shots,
            "accuracy": round(accuracy, 2),
            "seed": 42,
            "n_train": int(len(train_df)),
        }
    )
    args.out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
