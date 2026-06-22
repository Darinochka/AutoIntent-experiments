import glob
import torch
import os
import re
import wandb
from autointent import Pipeline, Dataset
from autointent.configs import LoggingConfig

if "WANDB_API_KEY" in os.environ:
    wandb.login(key=os.environ["WANDB_API_KEY"])
else:
    raise ValueError("WANDB_API_KEY not found in environment variables!")

base_search_space = [
    {
        "node_type": "embedding",
        "target_metric": "retrieval_hit_rate",
        "search_space": [
            {
                "module_name": "retrieval",
                "k": [10],
                "embedder_config": [
                    {"model_name": "avsolatorio/GIST-small-Embedding-v0", "batch_size": 16, "device": "cuda"},
                    {"model_name": "infgrad/stella-base-en-v2", "batch_size": 16, "device": "cuda"},
                    {"model_name": "sentence-transformers/all-MiniLM-L6-v2", "batch_size": 16, "device": "cuda"},
                    {"model_name": "BAAI/bge-reranker-base", "batch_size": 16, "device": "cuda"}
                ]
            }
        ],
    },
    {
        "node_type": "scoring",
        "target_metric": "scoring_roc_auc",
        "search_space": [
            {
                "module_name": "knn",
                "k": [1, 3, 5, 10],
                "weights": ["uniform", "distance", "closest"],
            },
            {"module_name": "linear"},
        ],
    },
    {
        "node_type": "decision",
        "target_metric": "decision_accuracy",
        "search_space": [
            {"module_name": "argmax"}
        ],
    },
]

clinc150_search_space = [
    {
        "node_type": "decision",
        "target_metric": "decision_accuracy",
        "search_space": [
            {"module_name": "tunable"},
            {"module_name": "jinoos"}
        ],
    }
]

log_config = LoggingConfig(report_to=["wandb"], clear_ram=True, dump_modules=True)

def get_search_space(dataset_name: str):
    if "clinc150" in dataset_name.lower():
        return base_search_space[:-1] + clinc150_search_space
    return base_search_space

def run_experiment(dataset_path: str):
    base_name = dataset_path.split("/")[-1]
    n_examples = int(base_name.split("_")[1])
    dataset_name = "clinc150" if "clinc150" in dataset_path.lower() else "other"
    
    log_config.run_name = f"{dataset_name}_examples_{n_examples}"
    
    dataset = Dataset.from_json(dataset_path)
    
    current_search_space = get_search_space(dataset_path)
    
    pipeline = Pipeline.from_search_space(current_search_space)
    pipeline.set_config(log_config)
    
    pipeline.fit(dataset)
    
    torch.cuda.empty_cache()
    wandb.finish()


if __name__ == "__main__":
    clinc_datasets = glob.glob("augmented_datasets/*clinc150*_examples.json")
    other_datasets = [f for f in glob.glob("augmented_datasets/dataset_*_examples.json") 
                     if "clinc150" not in f.lower()]
    
    # Функция для извлечения числа из имени файла
    def extract_number(path: str) -> int:
        match = re.search(r'dataset_(\d+)_examples', path)
        if not match:
            raise ValueError(f"Invalid filename format: {path}")
        return int(match.group(1))
    
    clinc_datasets.sort(key=extract_number)
    other_datasets.sort(key=extract_number)
    
    print("Testing CLINC150 datasets:")
    for dataset in clinc_datasets:
        print(f"\n{'='*40}\nProcessing: {dataset}\n{'='*40}")
        run_experiment(dataset)
    
    print("\nTesting other datasets:")
    for dataset in other_datasets:
        print(f"\n{'='*40}\nProcessing: {dataset}\n{'='*40}")
        run_experiment(dataset)