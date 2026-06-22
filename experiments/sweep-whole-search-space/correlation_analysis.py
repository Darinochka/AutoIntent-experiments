import pandas as pd
from scipy import stats
from typing import Dict, Tuple, List
import json


english_datasets = ["banking77", "snips"]
russian_datasets = ["banking77_ru", "snips_ru"]
oos_datasets = ["clinc150_ru"]

models = [
    "DeepSeek-V3-0324",
    "gpt-4o-mini-2024-07-18",
    "Qwen2.5-7B-Instruct-AWQ",
    "Meta-Llama-3.1-8B-Instruct-Turbo",
]

def load_results(file_path: str) -> Dict:
    """Load results from a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def prepare_correlation_data(
    results: Dict, metric: str = "decision_accuracy"
) -> pd.DataFrame:
    """Convert nested dictionary to DataFrame for correlation analysis."""
    data = []

    for model in models:
        for dataset in results[model]:
            if dataset not in english_datasets + russian_datasets + oos_datasets:
                continue
            for naug in results[model][dataset]:
                data.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "naug": int(naug),
                        metric: results[model][dataset][naug][metric],
                    }
                )

    return pd.DataFrame(data)


def calculate_correlation(
    df: pd.DataFrame, x_col: str, y_col: str
) -> Tuple[float, float, int]:
    """Calculate Pearson correlation, p-value, and number of data points."""
    return stats.pearsonr(df[x_col], df[y_col]) + (len(df),)


def format_correlation(corr: float, pval: float, n: int, alpha: float = 0.05) -> str:
    """Format correlation with statistical significance check."""
    if pval > alpha:
        return "-"
    return f"{corr:.3f}"


def get_model_correlations(
    df: pd.DataFrame,
    model: str,
    english_datasets: List[str],
    russian_datasets: List[str],
    oos_datasets: List[str],
) -> Dict:
    """Calculate correlations for a single model across different dataset groups."""
    row_data = {"Model": model}
    
    # English datasets
    model_english = df[
        (df["model"] == model) & (df["dataset"].isin(english_datasets))
    ]
    if len(model_english) >= 2:
        corr, pval, n = calculate_correlation(
            model_english, "naug", "decision_accuracy"
        )
        row_data["English"] = format_correlation(corr, pval, n)
        print("english support:", n)
    else:
        row_data["English"] = "-"
    
    # Russian datasets
    model_russian = df[
        (df["model"] == model) & (df["dataset"].isin(russian_datasets))
    ]
    if len(model_russian) >= 2:
        corr, pval, n = calculate_correlation(
            model_russian, "naug", "decision_accuracy"
        )
        print("russian support:", n)

        row_data["Russian"] = format_correlation(corr, pval, n)
    else:
        row_data["Russian"] = "-"
    
    # OOS datasets
    model_oos = df[(df["model"] == model) & (df["dataset"].isin(oos_datasets))]
    if len(model_oos) >= 2:
        corr, pval, n = calculate_correlation(model_oos, "naug", "decision_accuracy")
        row_data["OOS"] = format_correlation(corr, pval, n)
        print("oos support:", n)

    else:
        row_data["OOS"] = "-"
    
    return row_data


def get_dataset_correlations(df: pd.DataFrame) -> List[Dict]:
    """Calculate correlations for each individual dataset."""
    dataset_correlations = []
    for dataset in set(df["dataset"]):
        dataset_data = df[df["dataset"] == dataset]
        if len(dataset_data) >= 2:
            corr, pval, n = calculate_correlation(dataset_data, "naug", "decision_accuracy")
            dataset_correlations.append({
                "Dataset": dataset,
                "Correlation": format_correlation(corr, pval, n),
                "Support": n
            })
        else:
            dataset_correlations.append({
                "Dataset": dataset,
                "Correlation": "-",
                "Support": n
            })
    return dataset_correlations


def get_group_correlations(
    df: pd.DataFrame,
    english_datasets: List[str],
    russian_datasets: List[str],
    oos_datasets: List[str],
) -> List[Dict]:
    """Calculate correlations for dataset groups (English, Russian, OOS)."""
    group_correlations = []
    
    # English datasets
    english_data = df[df["dataset"].isin(english_datasets)]
    if len(english_data) >= 2:
        corr, pval, n = calculate_correlation(english_data, "naug", "decision_accuracy")
        group_correlations.append({
            "Group": "English",
            "Correlation": format_correlation(corr, pval, n),
            "Support": n
        })
    else:
        group_correlations.append({
            "Group": "English",
            "Correlation": "-",
            "Support": 0
        })
    
    # Russian datasets
    russian_data = df[df["dataset"].isin(russian_datasets)]
    if len(russian_data) >= 2:
        corr, pval, n = calculate_correlation(russian_data, "naug", "decision_accuracy")
        group_correlations.append({
            "Group": "Russian",
            "Correlation": format_correlation(corr, pval, n),
            "Support": n
        })
    else:
        group_correlations.append({
            "Group": "Russian",
            "Correlation": "-",
            "Support": 0
        })
    
    # OOS datasets
    oos_data = df[df["dataset"].isin(oos_datasets)]
    if len(oos_data) >= 2:
        corr, pval, n = calculate_correlation(oos_data, "naug", "decision_accuracy")
        group_correlations.append({
            "Group": "OOS",
            "Correlation": format_correlation(corr, pval, n),
            "Support": n
        })
    else:
        group_correlations.append({
            "Group": "OOS",
            "Correlation": "-",
            "Support": 0
        })
    
    return group_correlations


def analyze_before_after(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze before and after effects for each augmentation level.
    
    Returns:
        List of dictionaries containing statistical analysis for each naug value.
        Each dictionary contains:
        - naug: number of augmentations
        - t_stat: t-statistic from paired t-test
        - pval: p-value from paired t-test
        - effect_size: Cohen's d effect size
        - mean_improvement: mean improvement in accuracy
        - n_comparisons: number of comparisons made
    """
    results = []
    for naug in sorted(df["naug"].unique()):
        if naug == 0:
            continue

        # Get data for current naug and baseline (naug=0)
        current_data = df[df["naug"] == naug]
        baseline_data = df[df["naug"] == 0]

        # Merge the data to ensure we're comparing the same model-dataset pairs
        merged_data = pd.merge(
            current_data,
            baseline_data,
            on=["model", "dataset"],
            suffixes=("_current", "_baseline"),
        )

        # Perform paired t-test
        t_stat, pval = stats.ttest_rel(
            merged_data["decision_accuracy_current"],
            merged_data["decision_accuracy_baseline"],
        )

        # Calculate effect size (Cohen's d)
        d = (
            merged_data["decision_accuracy_current"].mean()
            - merged_data["decision_accuracy_baseline"].mean()
        ) / merged_data["decision_accuracy_current"].std()

        results.append({
            "naug": int(naug),
            "t_stat": float(t_stat),
            "pval": float(pval),
            "effect_size": float(d),
            "mean_improvement": float(
                merged_data["decision_accuracy_current"].mean() - 
                merged_data["decision_accuracy_baseline"].mean()
            ) * 100,
            "n_comparisons": len(merged_data)
        })
    
    return pd.DataFrame(results)


def analyze_correlations(results: Dict):
    """Calculate all required correlations."""
    df = prepare_correlation_data(results)
    
    # Define dataset groups
    

    # Prepare and display main correlation table
    correlation_data = [
        get_model_correlations(df, model, english_datasets, russian_datasets, oos_datasets)
        for model in models
    ]
    correlation_df = pd.DataFrame(correlation_data)
    print("\nCorrelation Analysis (r values, non-significant correlations omitted):")
    print(correlation_df.to_string(index=False))
    
    # Calculate and display dataset-specific correlations
    dataset_correlations = get_dataset_correlations(df)
    dataset_df = pd.DataFrame(dataset_correlations)
    print("\nDataset-specific correlations across all models:")
    print(dataset_df.to_string(index=False))
    
    # Calculate and display group correlations
    group_correlations = get_group_correlations(
        df, english_datasets, russian_datasets, oos_datasets
    )
    group_df = pd.DataFrame(group_correlations)
    print("\nGroup correlations across all models:")
    print(group_df.to_string(index=False))
    
    # Analyze before and after effects
    before_after_results = analyze_before_after(df[df["dataset"] != "clinc150_ru"])
    print("\nBefore and after analysis results:")
    print(before_after_results.to_string(index=False))
    before_after_results.to_json("before_after.json", index=False)
    # print(json.dumps(before_after_results, indent=2))


if __name__ == "__main__":
    # Load your results
    results = load_results(
        "wandb_aug_exps/results.json"
    )  # Update with your actual file path
    analyze_correlations(results)
