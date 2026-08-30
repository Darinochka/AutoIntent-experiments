"""Выгрузка наборов данных испытаний в офлайн-зеркало с фиксацией ревизий.

Канонический источник — репозитории организации DeepPavlov в Hugging Face Hub.
Сценарий сохраняет каждый набор в формате AutoIntent и записывает манифест
с идентификатором ревизии каждого репозитория, чтобы прогоны читали зеркало,
а не сеть, и чтобы состояние данных было воспроизводимо.

Закрепление ревизии. В версии 0.4.0 Dataset.from_hub ревизию не принимает:
сигнатура from_hub(repo_name, data_split="default", intent_subset_name="intents")
(src/autointent/_dataset/_dataset.py), внутри — datasets.load_dataset без revision.
Поэтому вызов dataset_info(repo).sha и загрузка были бы независимы, и в манифест
мог бы попасть sha, не совпадающий со скачанным. Здесь загрузка повторяет
from_hub напрямую через datasets.load_dataset(..., revision=sha) и публичный
Dataset.from_dict, а sha перечитывается после выгрузки: расхождение — ошибка.

Запуск:
    uv run python experiments/pmi/prepare_datasets.py --out datasets \
        --manifest runs/datasets_manifest.json
"""

import argparse
import json
from pathlib import Path

from autointent import Dataset
from datasets import get_dataset_config_names, load_dataset
from huggingface_hub import HfApi

DATA_CONFIG = "default"
INTENTS_CONFIG = "intents"

REPOS = [
    "DeepPavlov/banking77",
    "DeepPavlov/hwu64",
    "DeepPavlov/massive",
    "DeepPavlov/minds14",
    "DeepPavlov/snips",
    "DeepPavlov/clinc150",
    "DeepPavlov/clinc150_subset",
]


def load_pinned(repo: str, revision: str) -> Dataset:
    """Повторяет Dataset.from_hub, но с закреплением ревизии репозитория.

    Соответствует реализации v0.4.0: конфигурация `default` — это сплиты,
    отдельная конфигурация `intents` — метаданные намерений.
    """
    splits = load_dataset(repo, DATA_CONFIG, revision=revision)
    mapping = {name: split.to_list() for name, split in splits.items()}
    if INTENTS_CONFIG in get_dataset_config_names(repo, revision=revision):
        mapping[INTENTS_CONFIG] = load_dataset(
            repo, name=INTENTS_CONFIG, split=INTENTS_CONFIG, revision=revision
        ).to_list()
    return Dataset.from_dict(mapping)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("datasets"))
    parser.add_argument("--manifest", type=Path, default=Path("runs/datasets_manifest.json"))
    parser.add_argument("--check", action="store_true", help="только проверить доступность репозиториев")
    args = parser.parse_args()

    api = HfApi()
    entries = []
    for repo in REPOS:
        revision = api.dataset_info(repo).sha
        name = repo.split("/", 1)[1]
        print(f"{repo} revision={revision}", flush=True)
        if args.check:
            continue
        dataset = load_pinned(repo, revision)
        revision_after = api.dataset_info(repo).sha
        if revision_after != revision:
            msg = (
                f"{repo}: ревизия изменилась во время выгрузки "
                f"({revision} -> {revision_after}); выгрузка не воспроизводима, повторите"
            )
            raise SystemExit(msg)
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / f"{name}.json"
        dataset.to_json(path)
        entries.append(
            {
                "repo": repo,
                "revision": revision,
                "path": str(path),
                "splits": {split: len(rows) for split, rows in dataset.items()},
            }
        )

    if args.check:
        print("CHECK OK: все репозитории доступны")
        return

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({"datasets": entries}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"манифест: {args.manifest}")
    print("PREPARE OK")


if __name__ == "__main__":
    main()
