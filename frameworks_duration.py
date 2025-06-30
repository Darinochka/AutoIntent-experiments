import logging
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import concatenate_datasets, load_dataset
from sklearn.metrics import classification_report

import wandb

logger = logging.getLogger(__name__)


def create_few_shot_split(
    df: pd.DataFrame,
    label_column: str = "label",
    examples_per_label: int = 8,
    random_seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a few-shot dataset split with a specified number of examples per label using pandas.

    Args:
        df: Input DataFrame containing features and labels.
        label_column: The name of the column containing labels (default: 'label').
        examples_per_label: Number of examples to include per label in the train split (default: 8).
        multilabel: Whether the dataset is multi-label, with labels as list-like values (default: False).
        random_seed: Random seed for reproducibility (default: None).

    Returns:
        A tuple of two DataFrames: (train_df, validation_df).
    """
    # Ensure index column exists to track rows
    df = df.copy()
    df.reset_index(drop=False, inplace=True)
    df.rename(columns={"index": "__index__"}, inplace=True)
    unique_labels = df[label_column].dropna().unique().tolist()

    selected_indices: set[Any] = set()
    train_frames: list[pd.DataFrame] = []

    # Sample per label
    for label in unique_labels:
        subset = df[df[label_column] == label]

        if random_seed is not None:
            sampled = subset.sample(
                n=min(examples_per_label, len(subset)),
                random_state=random_seed,
            )
        else:
            sampled = subset.sample(n=min(examples_per_label, len(subset)))

        count_selected = len(sampled)
        if count_selected < examples_per_label:
            logger.warning(
                "Only %d examples available for label '%s', requested %d",
                count_selected,
                label,
                examples_per_label,
            )

        train_frames.append(sampled)
        selected_indices.update(sampled["__index__"].tolist())

    # Combine train splits and drop helper columns
    train_df = pd.concat(train_frames, ignore_index=True)
    train_df.drop(columns=["__index__"], inplace=True)
    train_df = train_df.sample(frac=1).reset_index(drop=True)

    return train_df


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
            dataset[col] = concatenate_datasets(
                [dataset[f"{col}_0"], dataset[f"{col}_1"]]
            )
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


def evalute_fedot(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Train a Fedot model on the provided training and testing data.

    Args:
        train_df (pd.DataFrame): The training data.
        test_df (pd.DataFrame): The testing data.
    """
    # !pip install fedot
    from fedot.api.main import Fedot

    X_train, y_train = train_df[["utterance"]], train_df["label"].astype(int)
    X_test, y_test = test_df[["utterance"]], test_df["label"].astype(int)
    model = Fedot(problem="classification", timeout=5, preset="best_quality", n_jobs=-1)
    model.fit(features=X_train, target=y_train)
    prediction = model.predict(features=X_test)
    return prediction


def evaluate_h2o(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Train an H2O model on the provided training and testing data.

    Args:
        train_df (pd.DataFrame): The training data.
        test_df (pd.DataFrame): The testing data.
    """
    # !pip install h2o
    import h2o
    from h2o.automl import H2OAutoML
    from h2o.estimators import H2OGradientBoostingEstimator
    from h2o.estimators.word2vec import H2OWord2vecEstimator

    max_models: int = 20
    max_runtime_secs: int = 20 * 60
    seed: int = 42

    h2o.init()

    train_h2o = h2o.H2OFrame(train_df)
    test_h2o = h2o.H2OFrame(test_df)
    train_h2o["label"] = train_h2o["label"].asfactor()
    test_h2o["label"] = test_h2o["label"].asfactor()
    train, valid = train_h2o.split_frame(ratios=[0.8])
    text_col = "utterance"
    label_col = "label"
    train_tokens = train[text_col].tokenize("\\s+")
    valid_tokens = valid[text_col].tokenize("\\s+")
    test_tokens = test_h2o[text_col].tokenize(
        "\\s+"
    )  # Word2Vec needs token lists :contentReference[oaicite:0]{index=0}

    w2v_model = H2OWord2vecEstimator(sent_sample_rate=0.0, epochs=10)
    w2v_model.train(training_frame=train_tokens)

    train_vecs = w2v_model.transform(train_tokens, aggregate_method="AVERAGE")
    valid_vecs = w2v_model.transform(valid_tokens, aggregate_method="AVERAGE")
    test_vecs = w2v_model.transform(test_tokens, aggregate_method="AVERAGE")

    train_ext = train_vecs.cbind(train[label_col])
    valid_ext = valid_vecs.cbind(valid[label_col])
    test_ext = test_vecs.cbind(test_h2o[label_col])

    x_cols = train_vecs.columns
    y_col = label_col

    # 9. Run H2OAutoML
    aml = H2OAutoML(
        max_models=max_models,
        max_runtime_secs=max_runtime_secs,
        seed=seed,
        balance_classes=True,
        sort_metric="mean_per_class_error",
    )
    aml.train(
        x=x_cols,
        y=y_col,
        training_frame=train_ext,
        validation_frame=valid_ext,
        leaderboard_frame=test_ext,
    )

    preds = aml.leader.predict(test_ext).as_data_frame()
    return preds["predict"]


def evaluate_lama(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Train a LAMA model on the provided training and testing data.

    Args:
        train_df (pd.DataFrame): The training data.
        test_df (pd.DataFrame): The testing data.
    """
    # !pip install lightautoml[nlp]
    from lightautoml.automl.presets.text_presets import TabularNLPAutoML
    from lightautoml.tasks import Task
    # pytorch<2.7.0
    # https://github.com/sb-ai-lab/LightAutoML/issues/173

    automl = TabularNLPAutoML(task=Task(name="multiclass", metric="f1_macro"))
    automl.fit_predict(train_df, roles={"target": "label"})
    test_preds = automl.predict(test_df).data
    return np.argmax(test_preds, axis=-1)


def evaluate_gama(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Train a GAMA model on the provided training and testing data.

    Args:
        train_df (pd.DataFrame): The training data.
        test_df (pd.DataFrame): The testing data.
    """
    # NOT WORKING
    # ValueError: population must be at least size 3 for a pair to be selected
    raise NotImplementedError("GAMA is not working yet.")
    # !pip install gama
    from gama import GamaClassifier

    automl = GamaClassifier(max_total_time=180, store="nothing")
    automl.fit(train_df[["utterance"]], train_df[["label"]])


def evaluate_glueon(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Train a GlueOn model on the provided training and testing data.

    Args:
        train_df (pd.DataFrame): The training data.
        test_df (pd.DataFrame): The testing data.
    """
    #!pip install autogluon
    import uuid

    from autogluon.multimodal import MultiModalPredictor

    print(train_df["label"].value_counts())

    model_path = f"/tmp/{uuid.uuid4().hex}-automm_sst"
    predictor = MultiModalPredictor(
        label="label", problem_type="multiclass", eval_metric="acc", path=model_path
    )
    predictor.fit(train_df, time_limit=60 * 10)
    predictions = predictor.predict(test_df)
    return predictions


def evaluate_ludwig(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError("Ludwig is not implemented yet.")


def evaluate_tpot(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError("TPOT is not implemented yet.")


def main(dataset_name: str, framework: str, few_shot: int | None = None):
    run = wandb.init(
        project="AutoML-Eval",
        name=f"{dataset_name}-{framework}-{few_shot}",
        tags=[dataset_name, framework],
        config={
            "dataset": dataset_name,
            "framework": framework,
            "few_shot": few_shot,
        },
    )
    # Load the dataset
    train_df, test_df = load_data(dataset_name)
    if few_shot is not None:
        print(few_shot)
        train_df = create_few_shot_split(
            train_df, examples_per_label=few_shot, random_seed=42
        )
    print(train_df["label"].value_counts())
    # Evaluate the model
    if framework == "fedot":
        predictions = evalute_fedot(train_df, test_df)
    elif framework == "h2o":
        predictions = evaluate_h2o(train_df, test_df)
    elif framework == "lama":
        predictions = evaluate_lama(train_df, test_df)
    elif framework == "gama":
        predictions = evaluate_gama(train_df, test_df)
    elif framework == "gluon":
        predictions = evaluate_glueon(train_df, test_df)
    else:
        raise ValueError(f"Unknown framework: {framework}")
    # Log the predictions
    run.log({"predictions": wandb.Table(dataframe=pd.DataFrame(predictions))})
    # Log the classification report
    report = classification_report(test_df["label"], predictions, output_dict=True)
    run.log(report)
    # Finish the run
    run.finish()


if __name__ == "__main__":
    import logging
    import os
    import sys

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    handler = logging.StreamHandler(sys.stdout)


    for frame in ["h2o", "lama", "gluon"]:
        main(dataset_name="DeepPavlov/minds14", framework=frame)
