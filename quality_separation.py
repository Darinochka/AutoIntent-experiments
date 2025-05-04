from dotenv import load_dotenv
import os
import yaml
from autointent import setup_logging
from autointent import Dataset, Pipeline
from autointent.configs import LoggingConfig, EmbedderConfig, DataConfig


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
      n_trials: 20
      clf_name: [RandomForestClassifier]
      n_estimators: [200, 300, 500]
      max_depth: [100, 150, 200]
      max_features: [sqrt, log2, null]
    - module_name: bert
      n_trials: 15
      num_train_epochs: [3]
      batch_size: [8, 16, 32]
      learning_rate: [5.0e-5, 1.0e-4, 5.0e-4, 1.0e-3]
    - module_name: lora
      n_trials: 20
      num_train_epochs: [3]
      batch_size: [8, 16, 32]
      learning_rate: [5.0e-5, 1.0e-4, 5.0e-4, 1.0e-3]
      lora_alpha: [16, 32, 64]
      lora_dropout: [0.1, 0.2, 0.3]
      r: [8, 16, 32, 64]
    - module_name: ptuning
      n_trials: 20
      num_train_epochs: [3]
      batch_size: [8, 16, 32]
      learning_rate: [5.0e-5, 1.0e-4, 5.0e-4, 1.0e-3]
      num_virtual_tokens: [10, 30, 50]
      encoder_dropout: [0.1, 0.2, 0.3]
      encoder_hidden_size: [128, 256, 512]
      encoder_reparameterization_type: [MLP, LSTM]
      encoder_num_layers: [1, 2, 3]
- node_type: decision
  target_metric: decision_accuracy
  search_space:
    - module_name: threshold
      n_trials: 20
      thresh:
        low: 0.1
        high: 0.9
        step: 0.1
    - module_name: argmax
"""

# search_space_raw = """
# - node_type: scoring
#   target_metric: scoring_accuracy
#   metrics: [scoring_roc_auc, scoring_precision, scoring_recall, scoring_f1]
#   search_space:
#     - module_name: knn
#       k:
#         low: 1
#         high: 2
#       weights: [uniform]
#     - module_name: linear
# - node_type: decision
#   target_metric: decision_accuracy
#   search_space:
#     - module_name: threshold
#       thresh:
#         low: 0.1
#         high: 0.9
#         step: 0.1
#     - module_name: argmax
# """

datasets_names = ["DeepPavlov/banking77", "DeepPavlov/minds14", "DeepPavlov/hwu64", "DeepPavlov/snips", "DeepPavlov/massive"]

if __name__ == "__main__":
    from argparse import ArgumentParser
    import os
    from pathlib import Path
    import shutil

    parser = ArgumentParser(description="This is the script for measuring the quality of AutoIntent using holdout validation. This script runs multiple seeds to get confidence estimations.")
    parser.add_argument("--experiment-name", type=str, required=True, help="aka name of the wandb project")
    parser.add_argument("--embedder-name", type=str, default=None, help="Name of HF repository. Omit this param to use AutoIntent's default embedder.")
    parser.add_argument("--seeds", nargs="+")
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--n-shots", type=int, default=10)
    args = parser.parse_args()

    load_dotenv()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    search_space = yaml.safe_load(search_space_raw)

    workdir = Path("experiments-assets") / args.experiment_name

    setup_logging(level="INFO", log_filename=workdir / "logs")

    os.environ["WANDB_PROJECT"] = args.experiment_name

    for seed in args.seeds:
        for ratio in [None, 0.2, 0.5, 0.7]:
            for dataset in datasets_names:
                data_config = DataConfig(scheme="ho", separation_ratio=ratio, examples_per_intent=args.n_shots, is_few_shot_train=True)

                logging_config = LoggingConfig(
                    run_name=dataset.split("/")[1] + f"[{seed=}]" + f"[{ratio=}]",
                    clear_ram=True,
                    dump_modules=True,
                    report_to=["wandb"],
                    project_dir=workdir
                )

                if args.embedder_name is None:
                    embedder_config = EmbedderConfig(use_cache=True)
                else:
                    embedder_config = EmbedderConfig(model_name=args.embedder_name, use_cache=True)

                pipe = Pipeline.from_search_space(search_space, seed=int(seed))
                pipe.set_config(logging_config)
                pipe.set_config(embedder_config)
                pipe.set_config(data_config)

                pipe.fit(Dataset.from_hub(dataset), refit_after=True, sampler="tpe", incompatible_search_space="filter")
                shutil.rmtree(logging_config.dirpath)
