from autointent import Pipeline, Dataset
from autointent.configs import LoggingConfig, EmbedderConfig
import torch
import wandb
import json
import os

os.environ["WANDB_PROJECT"] = "UtteranceEvolver"

def run_experiment(dataset, model_name: str, n_aug=0, ds_type: str = "json", name: str = "dataset"):
    dataset_base = name.split("/")[-1]

    log_config = LoggingConfig(
        run_name=f"{model_name.split('/')[-1]}_{dataset_base}_naug_{n_aug}",
        report_to=["wandb"], clear_ram=True, dump_modules=True)

    emb_config = EmbedderConfig(use_cache=True, batch_size=16, device="cuda")

    if ds_type == "json":
        dataset = Dataset.from_json(dataset)
    else:
        dataset = Dataset.from_dict(dataset)

    pipeline_optimizer = Pipeline.from_preset(search_space)
    pipeline_optimizer.set_config(log_config)
    pipeline_optimizer.set_config(emb_config)
    pipeline_optimizer.fit(dataset)
    torch.cuda.empty_cache()
    wandb.finish()


if __name__ == "__main__":
    datasets = [
        "DeepPavlov/banking77_ru",
        "DeepPavlov/clinc150_ru",
        "DeepPavlov/banking77",
        "DeepPavlov/snips",
        "DeepPavlov/snips_ru"
    ]
    max_utterances_per_intent = 10
    model_name = "Qwen2.5-7B-Instruct-AWQ"
    search_space = "heavy_moderate"
    n_evolutions = 10

    datasets_subset = []
    datasets_subset_aug = []

    for dataset_name in datasets:
        base_name = dataset_name.split("/")[-1]
        path_to_datasets = "/home/darinka/AutoIntent-experiments/experiments/utterance_augmentation/datasets"
        dataset_base = f"{path_to_datasets}/{base_name}_subset_{max_utterances_per_intent}.json"
        dataset_subset = f"{path_to_datasets}/{model_name.split('/')[-1]}/{base_name}_subset_{max_utterances_per_intent}.json"
        dataset_subset_aug = dataset_subset.replace(".json", "_augmented.json")

        datasets_subset.append(dataset_base)
        datasets_subset_aug.append(dataset_subset_aug)

    for subset, aug in zip(datasets_subset[3:], datasets_subset_aug[3:]):
        print(f"Processing dataset: {subset}")
        print(f"Augmented dataset: {aug}")
        
        run_experiment(
            dataset=subset, 
            model_name=model_name,
            n_aug=0,
            ds_type="json",
            name=subset)
        
        with open(subset, "r") as f:
            dataset = json.load(f)
        
        n = len(dataset["train"])

        with open(aug, "r") as f:
            intents = {}
            dataset_aug = json.load(f)

            for utterance in dataset_aug["train"][n:]:
                label = utterance["label"]
                if label not in intents:
                    intents[label] = []
                intents[label].append(utterance)

            for i in range(1, n_evolutions + 1):
                train = dataset_aug["train"][:n]
               
                for label, utterances in intents.items():
                    j = 0
                    while j < len(utterances):
                        train.extend(utterances[j:j + i])
                        j += i 
                        j += (n_evolutions - i)

                dataset_aug["train"] = train
                run_experiment(
                    dataset=dataset_aug, 
                    model_name=model_name,
                    n_aug=i,
                    ds_type="dict",
                    name=aug
                )
