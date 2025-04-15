import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

def calculate_metrics_stats(metrics_list):
    """
    Calculate mean for each metric across all configurations
    for each module.
    
    Args:
        metrics_list (list): List of metric dictionaries loaded from YAML
        
    Returns:
        pd.DataFrame: DataFrame with mean for each metric per module
    """
    # Create a dictionary to store all entries for each module_name
    module_entries = {}
    
    for entry in metrics_list:
        module_name = entry['module_name']
        if module_name not in module_entries:
            module_entries[module_name] = []
        module_entries[module_name].append(entry['metrics'])
    
    # Calculate statistics for each module
    rows = []
    for module_name, metrics_list in module_entries.items():
        # Create a dictionary to store all values for each metric
        metric_values = {
            'scoring_accuracy': [],
            'scoring_f1': [],
            'scoring_precision': [],
            'scoring_recall': [],
            'scoring_roc_auc': []
        }
        
        # Collect all values for each metric
        for metrics in metrics_list:
            for metric_name in metric_values.keys():
                metric_values[metric_name].append(metrics[metric_name])
        
        # Calculate mean for each metric
        row = {'module_name': module_name}
        for metric_name, values in metric_values.items():
            row[metric_name] = np.std(values)
        
        rows.append(row)
    
    # Create DataFrame and reorder columns
    df = pd.DataFrame(rows)
    columns = ["module_name", "scoring_accuracy", "scoring_f1", "scoring_precision", "scoring_recall", "scoring_roc_auc"]
    df = df[columns]
    return df

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--experiment-name", type=str, required=True, help="Name of wandb project")
    parser.add_argument("--results-dir", type=str, default="scoring_results")
    parser.add_argument("--output-dir", type=str, default="scoring_stats")
    args = parser.parse_args()

    results_dir = Path(args.results_dir) / args.experiment_name
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(exist_ok=True, parents=True)

    for dataset_results in results_dir.iterdir():
        results = yaml.safe_load(dataset_results.read_text())
        df = calculate_metrics_stats(results)
        df.to_json(output_dir / f"{dataset_results.stem}.json", orient='records') 