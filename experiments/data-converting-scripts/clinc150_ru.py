from autointent import Dataset
from autointent.schemas import Sample
from datasets import load_from_disk, Dataset as HFDataset


def convert_ruclinc150(clinc150_train: HFDataset, ood_index=42):
    all_labels = sorted(clinc150_train.unique("intent"))
    assert all_labels == list(range(151))

    in_domain_samples = clinc150_train.filter(lambda x: x["intent"] != ood_index)
    oos_samples = clinc150_train.filter(lambda x: x["intent"] == ood_index)

    classwise_samples = [[] for _ in range(150)]

    for batch in in_domain_samples.iter(batch_size=16, drop_last_batch=False):
        for txt, intent_id in zip(batch["text"], batch["intent"], strict=False):
            intent_id -= int(intent_id > ood_index)
            target_list = classwise_samples[intent_id]
            target_list.append({"utterance": txt, "label": intent_id})

    train_samples = [sample for samples_from_one_class in classwise_samples for sample in samples_from_one_class]
    oos_samples = [{"utterance": txt} for txt in oos_samples["text"]]

    return [Sample(**sample) for sample in train_samples + oos_samples]

if __name__ == "__main__":
    # git clone git@github.com:LadaNikitina/clinc150 data/RuClinc150
    # rm -rf data/RuClinc150/.git

    clinc150 = load_from_disk("data/RuClinc150")
    train_samples = convert_ruclinc150(clinc150["train"])
    val_samples = convert_ruclinc150(clinc150["validation"])
    test_samples = convert_ruclinc150(clinc150["test"])

    clinc150_converted = Dataset.from_dict(
        {"train": train_samples, "validation": val_samples, "test": test_samples}
    )
    clinc150_converted.to_json("datasets_converted/data/clinc150_ru.json")
