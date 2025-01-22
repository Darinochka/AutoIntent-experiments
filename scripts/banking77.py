# noqa: INP001
"""Convert events dataset to autointent internal format and scheme."""

import json

import requests
from datasets import Dataset as HFDataset
from datasets import load_dataset

from autointent import Dataset
from autointent.schemas import Intent, Sample


def get_intents_data(github_file: str | None = None) -> list[Intent]:
    """Load specific json from HF repo."""
    github_file = github_file or "https://huggingface.co/datasets/PolyAI/banking77/resolve/main/dataset_infos.json"
    raw_text = requests.get(github_file, timeout=5).text
    dataset_description = json.loads(raw_text)
    intent_names = dataset_description["default"]["features"]["label"]["names"]
    return [Intent(id=i, name=name) for i, name in enumerate(intent_names)]


def convert_banking77(
    banking77_split: HFDataset, intents_data: list[Intent], shots_per_intent: int | None = None
) -> list[Sample]:
    """Convert one split into desired format."""
    all_labels = sorted(banking77_split.unique("label"))

    n_classes = len(intents_data)
    if all_labels != list(range(n_classes)):
        msg = "Something's wrong"
        raise ValueError(msg)

    classwise_samples = [[] for _ in range(n_classes)]

    for sample in banking77_split:
        target_list = classwise_samples[sample["label"]]
        if shots_per_intent is not None and len(target_list) >= shots_per_intent:
            continue
        target_list.append(Sample(utterance=sample["text"], label=sample["label"]))

    samples = [sample for samples_from_one_class in classwise_samples for sample in samples_from_one_class]
    print(f"{len(samples)=}")
    return samples


if __name__ == "__main__":
    intents_data = get_intents_data()
    banking77 = load_dataset("PolyAI/banking77", trust_remote_code=True)

    train_samples = convert_banking77(banking77["train"], intents_data=intents_data)
    test_samples = convert_banking77(banking77["test"], intents_data=intents_data)

    banking77_converted = Dataset.from_dict(
        {"train": train_samples, "test": test_samples, "intents": intents_data}
    )
    banking77_converted.to_json("datasets_converted/data/banking77.json")
