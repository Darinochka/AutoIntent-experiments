from dotenv import load_dotenv
import os
from autointent import setup_logging
from autointent import Dataset, Pipeline
import pandas as pd
import logging
import json

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

dataset_name = "DeepPavlov/clinc150"

def evaluate_oos_accuracy(predictions: pd.Series, test_labels: pd.Series) -> tuple[float, float, float]:
    """
    Evaluate the out-of-sample accuracy of the predictions.
    """
    in_domain_mask = ~test_labels.isna()
    in_domain_predictions = predictions[in_domain_mask]
    in_domain_gold_labels = test_labels[in_domain_mask]
    in_domain_accuracy = (in_domain_predictions == in_domain_gold_labels).mean()
    out_of_domain_recall = (predictions[test_labels.isna()].isna()).mean()
    out_of_domain_precision = (test_labels[predictions.isna()].isna()).mean()
    return in_domain_accuracy, out_of_domain_recall, out_of_domain_precision


if __name__ == "__main__":
    from argparse import ArgumentParser
    import os
    from pathlib import Path

    parser = ArgumentParser(description="This is the script for measuring the quality of AutoIntent using holdout validation. This script runs multiple seeds to get confidence estimations.")
    parser.add_argument("--experiment-name", type=str, required=True, help="aka name of the wandb project")
    parser.add_argument("--seeds", nargs="+")
    parser.add_argument("--cuda", type=str, default="0")
    args = parser.parse_args()

    load_dotenv()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    workdir = Path("experiments-assets") / args.experiment_name

    setup_logging(level="INFO", log_filename=workdir / "logs")

    res = []

    for seed in args.seeds:
        logger.info(f"Evaluating {dataset_name} with seed {seed}")

        dirpath = workdir / (dataset_name.split("/")[1] + f"[{seed=}]") / "pipe"
        if not dirpath.exists():
            raise FileNotFoundError(f"Directory {dirpath} does not exist")

        pipe = Pipeline.load(dirpath)
        logger.info(f"Loaded pipeline from {dirpath}")

        dataset = Dataset.from_hub(dataset_name)
        logger.info(f"Loaded dataset {dataset_name}")

        test_utterances = dataset["test"][Dataset.utterance_feature]
        test_labels = dataset["test"][Dataset.label_feature]

        labels = pipe.predict(test_utterances)
        logger.info("Predicted labels")

        in_domain_accuracy, out_of_domain_recall, out_of_domain_precision = evaluate_oos_accuracy(pd.Series(labels), pd.Series(test_labels))
        logger.info(f"In-domain accuracy: {in_domain_accuracy}, Out-of-domain recall: {out_of_domain_recall}, Out-of-domain precision: {out_of_domain_precision}")
        res.append({
            "seed": seed,
            "in_domain_accuracy": in_domain_accuracy,
            "out_of_domain_recall": out_of_domain_recall,
            "out_of_domain_precision": out_of_domain_precision
        })

    with open(workdir / "results.json", "w") as f:
        json.dump(res, f, indent=4, ensure_ascii=False)
