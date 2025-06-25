from dotenv import load_dotenv
import os
import yaml
from autointent import setup_logging
from autointent import Dataset, Pipeline
from autointent.configs import (
    LoggingConfig,
    DataConfig,
    HFModelConfig,
    TokenizerConfig,
    HPOConfig,
)


search_spaces = {
    "bert": """
- node_type: scoring
  target_metric: scoring_accuracy
  metrics: [scoring_roc_auc, scoring_precision, scoring_recall, scoring_f1]
  search_space:
    - module_name: bert
      num_train_epochs: [30]
      batch_size: [8, 16, 32]
      learning_rate:
        low: 5.0e-5
        high: 1.0e-3
        log: True
- node_type: decision
  target_metric: decision_accuracy
  search_space:
    - module_name: argmax
""",
    "lora": """
- node_type: scoring
  target_metric: scoring_accuracy
  metrics: [scoring_roc_auc, scoring_precision, scoring_recall, scoring_f1]
  search_space:
    - module_name: lora
      num_train_epochs: [15]
      batch_size: [8, 16, 32, 64]
      learning_rate:
        low: 5.0e-5
        high: 1.0e-3
        log: True
      lora_alpha: [16, 32, 64]
      lora_dropout: [0.02, 0.1, 0.2, 0.3]
      r: [8, 16, 32, 64, 96]
    - module_name: ptuning
      num_train_epochs: [15]
      batch_size: [8, 16, 32, 64]
      learning_rate:
        low: 5.0e-5
        high: 1.0e-3
        log: True
      num_virtual_tokens: [5, 10, 30, 50]
      encoder_dropout: [0.02, 0.1, 0.2, 0.3]
      encoder_hidden_size: [128, 256, 512]
      encoder_reparameterization_type: [MLP, LSTM]
      encoder_num_layers: [1, 2, 3]
- node_type: decision
  target_metric: decision_accuracy
  search_space:
    - module_name: argmax
""",
    "ptuning": """
- node_type: scoring
  target_metric: scoring_accuracy
  metrics: [scoring_roc_auc, scoring_precision, scoring_recall, scoring_f1]
  search_space:
    - module_name: ptuning
      num_train_epochs: [15]
      batch_size: [8, 16, 32, 64]
      learning_rate:
        low: 5.0e-5
        high: 1.0e-3
        log: True
      num_virtual_tokens: [5, 10, 30, 50]
      encoder_dropout: [0.02, 0.1, 0.2, 0.3]
      encoder_hidden_size: [128, 256, 512]
      encoder_reparameterization_type: [MLP, LSTM]
      encoder_num_layers: [1, 2, 3]
- node_type: decision
  target_metric: decision_accuracy
  search_space:
    - module_name: argmax
""",
}


datasets_names = [
    "DeepPavlov/minds14",
    "DeepPavlov/snips",
    "DeepPavlov/hwu64",
    "DeepPavlov/massive",
    "DeepPavlov/banking77",
]
# datasets_names = ["DeepPavlov/clinc150"]

if __name__ == "__main__":
    from argparse import ArgumentParser
    import os
    from pathlib import Path

    parser = ArgumentParser(
        description="This is the script for measuring the quality of AutoIntent using holdout validation. This script runs multiple seeds to get confidence estimations."
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="aka name of the wandb project",
    )
    parser.add_argument("--methods", nargs="+", type=str)
    parser.add_argument(
        "--validation-scheme", type=str, choices=["ho", "cv"], default="ho"
    )
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--n-shots", type=int, default=None)
    parser.add_argument("--hf-model", type=str, default="microsoft/deberta-v3-small")
    args = parser.parse_args()

    load_dotenv()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    workdir = Path("experiments-assets") / args.experiment_name

    setup_logging(level="INFO", log_filename=workdir / "logs")

    os.environ["WANDB_PROJECT"] = args.experiment_name
    print(f"n_shots: {args.n_shots}")

    assert all(m in search_spaces.keys() for m in args.methods)

    for method in args.methods:
        search_space = yaml.safe_load(search_spaces[method])
        for dataset in datasets_names:
            if args.n_shots is not None:
                data_config = DataConfig(
                    scheme=args.validation_scheme,
                    is_few_shot_train=True,
                    examples_per_intent=args.n_shots,
                )
            else:
                data_config = DataConfig(scheme=args.validation_scheme)

            run_name = dataset.split("/")[1] + f"[{method=}]"
            logging_config = LoggingConfig(
                run_name=run_name,
                clear_ram=True,
                dump_modules=True,
                report_to=["wandb", "codecarbon"],
                project_dir=workdir,
            )

            pipe = Pipeline.from_search_space(search_space)
            pipe.set_config(logging_config)
            pipe.set_config(data_config)
            pipe.set_config(HPOConfig(n_trials=50, n_startup_trials=20, sampler="tpe"))
            pipe.set_config(
                HFModelConfig(
                    model_name=args.hf_model,
                    tokenizer_config=TokenizerConfig(max_length=128),
                )
            )
            intents_name = (
                "intentsqwen3-32b" if dataset != "DeepPavlov/banking77" else "intents"
            )
            pipe.fit(
                Dataset.from_hub(dataset, intent_subset_name=intents_name),
                refit_after=False,
                incompatible_search_space="filter",
            )
            # pipe.dump(workdir / run_name / "pipe")
