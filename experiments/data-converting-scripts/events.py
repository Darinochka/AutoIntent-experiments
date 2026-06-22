# noqa: INP001
"""Convert events dataset to autointent internal format and scheme."""

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

def extract_intents_data(events_dataset: HFDataset) -> list[Intent]:
    """Extract intent names and assign ids to them."""
    intent_names = sorted({name for intents in events_dataset["train"]["all_labels"] for name in intents})
    for n in names_to_remove:
        intent_names.remove(n)
    return [Intent(id=i,name=name) for i, name in enumerate(intent_names)]


def converting_mapping(example: dict, intents_data: list[Intent]) -> dict[str, str | list[int] | None]:
    """Extract utterance and OHE label and drop the rest."""
    res = {
        "utterance": example["content"],
        "label": [
            int(intent.name in example["all_labels"]) for intent in intents_data
        ]
    }
    if sum(res["label"]) == 0:
        res["label"] = None
    return res


def convert_events(events_split: HFDataset, intents_data: dict[str, int]) -> list[Sample]:
    """Convert one split into desired format."""
    events_split = events_split.map(
        converting_mapping, remove_columns=events_split.features.keys(),
        fn_kwargs={"intents_data": intents_data}
    )

    samples = []
    for sample in events_split.to_list():
        if sample["utterance"] is None:
            continue
        samples.append(sample)

    mask = [sample["label"] is None for sample in samples]
    n_oos_samples = sum(mask)
    n_in_domain_samples = len(samples) - n_oos_samples
    
    print(f"{n_oos_samples=}")
    print(f"{n_in_domain_samples=}\n")

    # actually there are too few oos samples to include them, so filter out
    samples = list(filter(lambda sample: sample["label"] is not None, samples))

    return [Sample(**sample) for sample in samples]

if __name__ == "__main__":
    events_dataset = load_dataset("knowledgator/events_classification_biotech", trust_remote_code=True)

    intents_data = extract_intents_data(events_dataset)

    train_samples = convert_events(events_dataset["train"], intents_data)
    test_samples = convert_events(events_dataset["test"], intents_data)

    events_converted = Dataset.from_dict(
        {"train": train_samples, "test": test_samples, "intents": intents_data}
    )
    events_converted.to_json("datasets_converted/data/events.json")
