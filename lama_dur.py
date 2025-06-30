from lightautoml.automl.presets.text_presets import TabularNLPAutoML
from lightautoml.tasks import Task
import pandas as pd
from datasets import load_dataset, concatenate_datasets
import numpy as np
from sklearn.metrics import classification_report
import wandb

import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["WANDB_API_KEY"] = ""


def load_data(dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a dataset from the Hugging Face datasets library.

    Args:
        dataset_name (str): The name of the dataset to load.

    Returns:
        DatasetDict: A dictionary containing the train, validation, and test splits of the dataset.
    """
    # Load the dataset
    dataset = load_dataset(dataset_name)

    if "train_0" in dataset:
        for col in ["train", "validation"]:
            dataset[col] = concatenate_datasets([dataset[f"{col}_0"], dataset[f"{col}_1"]])
            dataset.pop(f"{col}_0")
            dataset.pop(f"{col}_1")

    train_data = dataset["train"]
    test_data = dataset["test"]

    train_df = train_data.to_pandas()
    max_label = train_df["label"].max()
    train_df.loc[train_df["label"].isna(), "label"] = max_label + 1

    test_df = test_data.to_pandas()
    test_df.loc[test_df["label"].isna(), "label"] = max_label + 1
    return train_df, test_df


if __name__ == "__main__":
    # pytorch<2.7.0
    # https://github.com/sb-ai-lab/LightAutoML/issues/173
    framework = "lama"
    few_shot = None

    for dataset_name in [
        "DeepPavlov/minds14",
        # "DeepPavlov/clinc150",
        # "DeepPavlov/massive",
        # "DeepPavlov/snips",
        # "DeepPavlov/hwu64",
        # "DeepPavlov/banking77",
    ]:
        for seed in list(range(42, 43)):
            np.random.seed(seed)

            run_name = f"{dataset_name}-{framework}-{few_shot}-{seed}"
            run = wandb.init(
                project="AutoML-Eval",
                name=run_name,
                tags=[dataset_name, framework],
                config={
                    "dataset": dataset_name,
                    "framework": framework,
                    "few_shot": few_shot,
                    "seed": seed,
                },
            )

            train_df, test_df = load_data(dataset_name)

            automl = TabularNLPAutoML(
                task=Task(name="multiclass", metric="f1_macro"),
                reader_params={"random_state": seed},
                # linear_pipeline_params={"text_features": "bert"},
                text_params={"lang": "en"},
            )
            try:
                automl.fit_predict(train_df, roles={"target": "label", "Text": "utterance"})
            except Exception as e:
                logger.error("error %s", e)
                wandb.finish(1)

            test_preds = automl.predict(test_df).data
            predictions = np.argmax(test_preds, axis=-1)

            # Log the predictions
            run.log({"predictions": wandb.Table(dataframe=pd.DataFrame(predictions))})
            # Log the classification report
            report = classification_report(test_df["label"], predictions, output_dict=True)
            run.log(report)
            # Finish the run
            run.finish()
