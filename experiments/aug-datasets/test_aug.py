import json
import glob
import torch
import wandb
from autointent import Pipeline, Dataset
from autointent.configs import LoggingConfig, EmbedderConfig

search_space = [
    {
        "node_type": "embedding",
        "target_metric": "retrieval_hit_rate",
        "search_space": [
            {
                "module_name": "retrieval",
                "k": [10],
                "embedder_name": [
                    "avsolatorio/GIST-small-Embedding-v0",
                    "infgrad/stella-base-en-v2",
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "BAAI/bge-reranker-base"
                ],
            }
        ],
    },
    {
        "node_type": "scoring",
        "target_metric": "scoring_roc_auc",
        "search_space": [
            {
                "module_name": "knn",
                "k": [5, 10],
                "weights": ["uniform", "distance", "closest"],
            },
            {"module_name": "linear"},
        ],
    },
    {
        "node_type": "decision",
        "target_metric": "decision_accuracy",
        "search_space": [
            {"module_name": "threshold", "thresh": [0.5]},
        ],
    },
]

log_config = LoggingConfig(report_to=["wandb"], clear_ram=True, dump_modules=True)
emb_config = EmbedderConfig(batch_size=16, device="cuda")

def run_experiment(dataset_path: str):
    base_name = dataset_path.split("/")[-1]
    n_examples = int(base_name.split("_")[1])
    
    log_config.run_name = f"minds14_ru_examples_{n_examples}"
    
    dataset = Dataset.from_json(dataset_path)
    
    pipeline = Pipeline.from_search_space(search_space)
    pipeline.set_config(log_config)
    pipeline.set_config(emb_config)
    
    pipeline.fit(dataset)
    
    torch.cuda.empty_cache()
    wandb.finish()

if __name__ == "__main__":
    datasets = glob.glob("augmented_datasets/dataset_*_examples.json")
    
    datasets.sort(key=lambda x: int(x.split("_")[1]))
    
    for dataset_path in datasets:
        print(f"\n{'='*40}\nProcessing: {dataset_path}\n{'='*40}")
        run_experiment(dataset_path)