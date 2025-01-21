# noqa: INP001
"""Convert clincq50 dataset to autointent internal format and scheme."""

from datasets import Dataset as HFDataset
from datasets import load_dataset

from autointent import Dataset
from autointent.schemas import Intent, Sample

# these classes contain too few sampls
names_to_remove = [
    "partnerships & alliances",
    "patent publication",
    "subsidiary establishment",
    "department establishment",
]

def extract_intents_data(events_dataset: HFDataset) -> tuple[list[Intent], dict[str, int]]:
    """Extract intent names and assign ids to them."""
    intent_names = sorted({name for intents in events_dataset["train"]["all_labels"] for name in intents})
    for n in names_to_remove:
        intent_names.remove(n)
    name_to_id = {name: i for i, name in enumerate(intent_names)}
    intents_data = [Intent(id=i,name=name) for i, name in enumerate(intent_names)]
    return intents_data, name_to_id


def converting_mapping(example: dict, name_to_id: dict[str, int]) -> dict[str, str | list[int]]:
    """Extract utterance and label and drop the rest."""
    return {
        "utterance": example["content"],
        "label": [
            name_to_id[intent_name] for intent_name in example["all_labels"] if intent_name not in names_to_remove
        ],
    }


def convert_events(events_split: HFDataset, name_to_id: dict[str, int]) -> list[Sample]:
    """Convert one split into desired format."""
    events_split = events_split.map(
        converting_mapping, remove_columns=events_split.features.keys(),
        fn_kwargs={"name_to_id": name_to_id}
    )

    in_domain_samples = []
    oos_samples = []  # actually this dataset doesn't contain oos_samples so this will stay empty
    for sample in events_split.to_list():
        if sample["utterance"] is None:
            continue
        if len(sample["label"]) == 0:
            sample.pop("label")
            oos_samples.append(sample)
        else:
            in_domain_samples.append(sample)

    return [Sample(**sample) for sample in in_domain_samples + oos_samples]

if __name__ == "__main__":
    events_dataset = load_dataset("knowledgator/events_classification_biotech", trust_remote_code=True)

    intents_data, name_to_id = extract_intents_data(events_dataset)

    train_samples = convert_events(events_dataset["train"], name_to_id)
    test_samples = convert_events(events_dataset["test"], name_to_id)

    events_converted = Dataset.from_dict(
        {"train": train_samples, "test": test_samples, "intents": intents_data}
    )
    events_converted.to_json("datasets_converted/data/events.json")
