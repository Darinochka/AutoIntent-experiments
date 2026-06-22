# define util  # noqa: INP001
"""Convert clincq50 dataset to autointent internal format and scheme."""

from datasets import Dataset as HFDataset
from datasets import load_dataset

from autointent import Dataset
from autointent.schemas import Intent, Sample


def extract_intents_data(
    clinc150_split: HFDataset, oos_intent_name: str = "ood"
) -> tuple[list[Intent], dict[str, int]]:
    """Extract intent names and assign ids to them."""
    intent_names = sorted(clinc150_split.unique("labels"))
    oos_intent_id = intent_names.index(oos_intent_name)
    intent_names.pop(oos_intent_id)

    n_classes = len(intent_names)
    assert n_classes == 150  # noqa: PLR2004, S101

    name_to_id = dict(zip(intent_names, range(n_classes), strict=False))
    intents_data = [Intent(id=i, name=name) for name, i in name_to_id.items()]
    return intents_data, name_to_id


def convert_clinc150(
    clinc150_split: HFDataset,
    name_to_id: dict[str, int],
    shots_per_intent: int | None = None,
    oos_intent_name: str = "ood",
) -> list[Sample]:
    """Convert one split into desired format."""
    oos_samples = []
    classwise_samples = [[] for _ in range(len(name_to_id))]
    n_unrecognized_labels = 0

    for batch in clinc150_split.iter(batch_size=16, drop_last_batch=False):
        for txt, name in zip(batch["data"], batch["labels"], strict=False):
            if name == oos_intent_name:
                oos_samples.append(Sample(utterance=txt))
                continue
            intent_id = name_to_id.get(name, None)
            if intent_id is None:
                n_unrecognized_labels += 1
                continue
            target_list = classwise_samples[intent_id]
            if shots_per_intent is not None and len(target_list) >= shots_per_intent:
                continue
            target_list.append(Sample(utterance=txt, label=intent_id))

    in_domain_samples = [sample for samples_from_single_class in classwise_samples for sample in samples_from_single_class]
    
    print(f"{len(in_domain_samples)=}")
    print(f"{len(oos_samples)=}")
    print(f"{n_unrecognized_labels=}\n")
    
    return in_domain_samples + oos_samples


if __name__ == "__main__":
    clinc150 = load_dataset("cmaldona/All-Generalization-OOD-CLINC150")

    intents_data, name_to_id = extract_intents_data(clinc150["train"])

    train_samples = convert_clinc150(clinc150["train"], name_to_id)
    validation_samples = convert_clinc150(clinc150["validation"], name_to_id)
    test_samples = convert_clinc150(clinc150["test"], name_to_id)

    clinc150_converted = Dataset.from_dict(
        {"train": train_samples, "validation": validation_samples, "test": test_samples, "intents": intents_data}
    )
    clinc150_converted.to_json("datasets_converted/data/clinc150.json")
