from datasets import Dataset as HFDataset
from datasets import load_dataset

from autointent import Dataset
from autointent.schemas import Intent, Sample


def extract_intents_info(split: HFDataset) -> tuple[list[Intent], dict[str, int]]:
    """Extract metadata."""
    intent_names = sorted(split.unique("label"))
    intent_names.remove("cooking_query")
    intent_names.remove("audio_volume_other")
    n_classes = len(intent_names)
    name_to_id = dict(zip(intent_names, range(n_classes), strict=False))
    intents_data = [Intent(id=i, name=intent_names[i]) for i in range(n_classes)]
    return intents_data, name_to_id


def convert_massive(split: HFDataset, name_to_id: dict[str, int]) -> list[Sample]:
    """Extract utterances and labels."""
    return [Sample(utterance=s["text"], label=name_to_id[s["label"]]) for s in split if s["label"] in name_to_id]


if __name__ == "__main__":
    massive = load_dataset("mteb/amazon_massive_intent", "en")
    intents, name_to_id = extract_intents_info(massive["train"])
    train_samples = convert_massive(massive["train"], name_to_id)
    test_samples = convert_massive(massive["test"], name_to_id)
    validation_samples = convert_massive(massive["validation"], name_to_id)
    dataset = Dataset.from_dict(
        {"intents": intents, "train": train_samples, "test": test_samples, "validation": validation_samples}
    )
    dataset.to_json("data/massive.json")
