"""Проверка 6.2: интерфейс fit/predict и сериализация обученного алгоритма."""

import argparse
from pathlib import Path

from autointent import Dataset, Pipeline

UTTERANCES = ["show me my latest transactions", "what is the weather tomorrow"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--preset", default="classic-light")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = Dataset.from_json(args.dataset)
    print("splits:", sorted(dataset.keys()))
    print("n_classes:", dataset.n_classes, "multilabel:", dataset.multilabel)

    pipeline = Pipeline.from_preset(args.preset, seed=args.seed)
    pipeline.fit(dataset)

    before = pipeline.predict(UTTERANCES)
    print("predictions before dump:", before)

    pipeline.dump(str(args.dump_dir))
    loaded = Pipeline.load(str(args.dump_dir))
    after = loaded.predict(UTTERANCES)
    print("predictions after load:", after)

    if before != after:
        msg = "предсказания загруженного алгоритма не совпадают с исходными"
        raise AssertionError(msg)
    print("SERIALIZATION OK")


if __name__ == "__main__":
    main()
