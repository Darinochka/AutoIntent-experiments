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

Атомарность записи. Манифест — доказательство приёмки прогона, поэтому "семь
валидных файлов зеркала рядом с обрезанным манифестом" должно быть невозможно,
а не просто маловероятно. Каждый файл (и файл зеркала, и манифест) сначала
пишется во временный файл в том же каталоге, что и итоговый путь (иначе
os.replace не атомарен — атомарность гарантирована только в пределах одной
файловой системы), а затем атомарно переименовывается на место через
os.replace. При сбое между записью и переименованием временный файл удаляется,
а итоговый путь остаётся нетронутым (прежним файлом или отсутствующим).
"""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Callable

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


def atomic_write(path: Path, writer: Callable[[Path], None]) -> None:
    """Записывает файл атомарно: writer пишет во временный путь, затем — os.replace на место.

    Временный файл создаётся в том же каталоге, что и path, чтобы переименование было
    атомарным (это гарантировано только в пределах одной файловой системы). При сбое
    во время writer временный файл удаляется, а path остаётся нетронутым.
    """
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        writer(tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


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
        atomic_write(path, dataset.to_json)
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
    manifest_text = json.dumps({"datasets": entries}, ensure_ascii=False, indent=2)
    atomic_write(args.manifest, lambda tmp_path: tmp_path.write_text(manifest_text, encoding="utf-8"))
    print(f"манифест: {args.manifest}")
    print("PREPARE OK")


if __name__ == "__main__":
    main()
