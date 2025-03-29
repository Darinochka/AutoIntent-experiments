from dotenv import load_dotenv
import os
import yaml
from autointent import setup_logging
from autointent import Dataset, Pipeline
from autointent.configs import LoggingConfig, EmbedderConfig


search_space_raw = """
- node_type: scoring
  target_metric: scoring_accuracy
  metrics: [scoring_roc_auc, scoring_precision, scoring_recall, scoring_f1]
  search_space:
    - module_name: knn
      k:
        low: 1
        high: 20
      weights: [uniform, distance, closest]
    - module_name: linear
    - module_name: dnnc
      k:
        low: 1
        high: 5
    - module_name: rerank
      k: [30]
      weights: [uniform]
      m: [ 1, 2, 3, 4, 5 ]
    - module_name: sklearn
      clf_name: [RandomForestClassifier]
      n_estimators: [50, 100, 150]
      max_depth: [10, 30, 60]
      max_features: [sqrt, log2, null]
- node_type: decision
  target_metric: decision_accuracy
  search_space:
    - module_name: threshold
      thresh:
        low: 0.1
        high: 0.9
        step: 0.1
    - module_name: argmax
    - module_name: tunable
"""

datasets_names = ["DeepPavlov/banking77", "DeepPavlov/minds14", "DeepPavlov/hwu64", "DeepPavlov/snips", "DeepPavlov/massive"]

if __name__ == "__main__":
    load_dotenv()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    search_space = yaml.safe_load(search_space_raw)

    setup_logging(level="INFO", log_filename="baseline-sweep")


    for dataset in datasets_names:
        logging_config = LoggingConfig(run_name=dataset.split("/")[1], clear_ram=True, dump_modules=True, report_to=["wandb"])
        embedder_config = EmbedderConfig()

        pipe = Pipeline.from_search_space(search_space)
        pipe.set_config(logging_config)
        pipe.set_config(embedder_config)

        pipe.fit(Dataset.from_hub(dataset), incompatible_search_space="filter")
