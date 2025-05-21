from datasets import load_dataset, concatenate_datasets
import pandas as pd
import logging
import numpy as np
import argparse
from sklearn.metrics import classification_report

import wandb

logging.basicConfig(level="INFO")


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


def evaluate_oos_accuracy(predictions: pd.Series, test_df: pd.DataFrame) -> tuple[float, float, float]:
    """
    Evaluate the out-of-sample accuracy of the predictions.
    """
    oos_class_id = test_df["label"].max()
    in_domain_mask = test_df["label"] != oos_class_id
    in_domain_predictions = predictions[in_domain_mask]
    in_domain_gold_labels = test_df[in_domain_mask]["label"]
    in_domain_accuracy = (in_domain_predictions == in_domain_gold_labels).mean()
    out_of_domain_recall = (predictions[test_df["label"] == oos_class_id] == oos_class_id).mean()
    out_of_domain_precision = (test_df["label"][(predictions == oos_class_id)] == oos_class_id).mean()
    return in_domain_accuracy, out_of_domain_recall, out_of_domain_precision

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
    from h2o.estimators import H2OGradientBoostingEstimator
    from h2o.estimators.word2vec import H2OWord2vecEstimator
    from h2o.automl import H2OAutoML

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
    aml.train(x=x_cols, y=y_col, training_frame=train_ext, validation_frame=valid_ext, leaderboard_frame=test_ext)

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


def evaluate_gluon(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Train a GlueOn model on the provided training and testing data.

    Args:
        train_df (pd.DataFrame): The training data.
        test_df (pd.DataFrame): The testing data.
    """
    #!pip install autogluon
    from autogluon.multimodal import MultiModalPredictor
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        predictor = MultiModalPredictor(label="label", presets="high_quality_hpo", problem_type="multiclass", eval_metric="acc", path=temp_dir)
        predictor.fit(train_df, time_limit=180, hyperparameters={"model.hf_text.checkpoint_name": "microsoft/deberta-v3-small"})
        predictions = predictor.predict(test_df)
        return predictions

def evaluate_ludwig(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError("Ludwig is not implemented yet.")

def evaluate_tpot(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError("TPOT is not implemented yet.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate AutoML models on a dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        help="The name of the dataset to evaluate.",
    )
    parser.add_argument(
        "--framework",
        type=str,
        choices=["fedot", "h2o", "lama", "gama", "gluon"],
        help="The name of the model to evaluate.",
    )
    parser.add_argument(
        "--evaluate-oos",
        action="store_true",
        help="Whether to evaluate the OOS accuracy.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="The name of the run.",
    )
    args = parser.parse_args()
    dataset_name = args.dataset
    framework = args.framework
    run = wandb.init(
        project="AutoML-Eval",
        name=f"eval-{dataset_name}-{framework}-{args.run_name}",
        tags=[dataset_name, framework],
        config={
            "dataset": dataset_name,
            "framework": framework,
        },
    )
    # Load the dataset
    train_df, test_df = load_data(dataset_name)

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
        predictions = evaluate_gluon(train_df, test_df)
    else:
        raise ValueError(f"Unknown framework: {framework}")
    # Log the predictions
    run.log({"predictions": wandb.Table(dataframe=pd.DataFrame(predictions))})
    # Log the classification report
    report = classification_report(test_df["label"], predictions, output_dict=True)
    run.log(report)
    if args.evaluate_oos:
        predictions = pd.Series(predictions).round(0).astype(int)
        in_domain_acc, oos_recall, oos_precision = evaluate_oos_accuracy(predictions, test_df)
        run.log({"in_domain_acc": in_domain_acc, "oos_recall": oos_recall, "oos_precision": oos_precision})
    # Finish the run
    run.finish()


def test_evaluate_oos_accuracy():
    """
    Test the evaluate_oos_accuracy function with synthetic data.
    """
    # Create synthetic test data
    test_df = pd.DataFrame({
        'label': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10]  # 10 is the OOS class
    })
    
    # Create synthetic predictions
    predictions = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10])  # Perfect predictions
    
    # Test perfect predictions
    in_domain_acc, oos_recall, oos_precision = evaluate_oos_accuracy(predictions, test_df)
    assert in_domain_acc == 1.0, "In-domain accuracy should be 1.0 for perfect predictions"
    assert oos_recall == 1.0, "OOS recall should be 1.0 for perfect predictions"
    assert oos_precision == 1.0, "OOS precision should be 1.0 for perfect predictions"
    
    # Test imperfect predictions
    imperfect_predictions = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10])  # Missed OOS prediction
    in_domain_acc, oos_recall, oos_precision = evaluate_oos_accuracy(imperfect_predictions, test_df)
    assert in_domain_acc == 1.0, "In-domain accuracy should still be 1.0"
    assert oos_recall == 0.5, "OOS recall should be 0.5 for missed OOS prediction"
    assert oos_precision == 1.0, "OOS precision should be 0.5 for missed OOS prediction"
    
    # Test false positive OOS predictions
    fp_predictions = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 10, 10])  # False positive OOS
    in_domain_acc, oos_recall, oos_precision = evaluate_oos_accuracy(fp_predictions, test_df)
    assert in_domain_acc == 0.9, "In-domain accuracy should still be 1.0"
    assert oos_recall == 1.0, "OOS recall should be 1.0"
    assert oos_precision == 2/3, "OOS precision should be 0.5 for one false positive"
    
    print("All tests passed!")


def show_preset():
    import json
    import yaml
    from autogluon.multimodal.utils.presets import get_presets

    hyperparameters, hyperparameter_tune_kwargs = get_presets(problem_type="default", presets="medium_quality_hpo")
    print(f"hyperparameters: {yaml.dump(hyperparameters, allow_unicode=True, default_flow_style=False)}")
    print(f"hyperparameter_tune_kwargs: {json.dumps(hyperparameter_tune_kwargs, sort_keys=True, indent=4)}")

def debug():
    from autogluon.core.utils import show_versions
    show_versions()

if __name__ == "__main__":
    # show_preset()
    # test_evaluate_oos_accuracy()
    debug()
    # main()