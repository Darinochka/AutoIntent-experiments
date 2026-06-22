import yaml
import pandas as pd

def process_metrics_to_dataframe(metrics_list, scoring_metric='scoring_accuracy'):
    """
    Process a list of metric dictionaries into a pandas DataFrame, keeping only the best
    entry for each module_name based on the specified scoring metric.
    
    Args:
        metrics_list (list): List of metric dictionaries loaded from YAML
        scoring_metric (str): Metric name to use for determining the best module (default: 'scoring_accuracy')
        
    Returns:
        pd.DataFrame: Processed DataFrame with one row per module_name
    """
    # Create a dictionary to store the best entry for each module_name
    best_entries = {}
    
    for entry in metrics_list:
        module_name = entry['module_name']
        current_score = entry['metrics'][scoring_metric]
        
        # If we haven't seen this module or this one is better, store it
        if module_name not in best_entries or current_score > best_entries[module_name]['metrics'][scoring_metric]:
            best_entries[module_name] = entry
    
    # Convert the best entries to a list of dictionaries for DataFrame creation
    rows = []
    for module_name, entry in best_entries.items():
        row = {
            'module_name': module_name,
            'scoring_accuracy': entry['metrics']['scoring_accuracy'],
            'scoring_f1': entry['metrics']['scoring_f1'],
            'scoring_precision': entry['metrics']['scoring_precision'],
            'scoring_recall': entry['metrics']['scoring_recall'],
            'scoring_roc_auc': entry['metrics']['scoring_roc_auc'],
            'module_params': yaml.dump(entry['module_params']),
            'module_dump_dir': entry['module_dump_dir']
        }
        rows.append(row)
    
    # Create DataFrame and reorder columns
    df = pd.DataFrame(rows)
    column_order = ['module_name', 'scoring_accuracy', 'scoring_f1', 
                   'scoring_precision', 'scoring_recall', 'scoring_roc_auc',
                   'module_params', 'module_dump_dir']
    df = df[column_order]
    
    return df

if __name__ == "__main__":
    from pathlib import Path
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--experiment-name", type=str, required=True, help="Name of wandb project")
    parser.add_argument("--results-dir", type=str, default="scoring_results")
    parser.add_argument("--output-dir", type=str, default="all_best_scorers")
    parser.add_argument("--scoring-metric", type=str, default="scoring_accuracy")
    args = parser.parse_args()

    results_dir = Path(args.results_dir) / args.experiment_name
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(exist_ok=True, parents=True)

    for dataset_results in results_dir.iterdir():
        results = yaml.safe_load(dataset_results.read_text())
        df = process_metrics_to_dataframe(results, scoring_metric=args.scoring_metric)
        df.to_json(output_dir / f"{dataset_results.stem}.json")