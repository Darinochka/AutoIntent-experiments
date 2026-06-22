import json
import pandas as pd
import numpy as np
from pathlib import Path
import re
from collections import defaultdict

def extract_dataset_name(filename):
    """Extract dataset name from filename pattern: <dataset>[seed='<seed>'].json"""
    match = re.match(r"(.+?)\[seed='\d+'\]\.json", filename)
    if match:
        return match.group(1)
    return None

def aggregate_results(input_dir, output_dir):
    """
    Aggregate results across different seeds for each dataset.
    
    Args:
        input_dir (str): Directory containing the JSON files
        output_dir (str): Directory to save aggregated results
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print(f"Input directory: {input_path}")
    print(f"Output directory: {output_path}")
    
    # Group files by dataset
    dataset_files = defaultdict(list)
    json_files = list(input_path.glob("*.json"))
    print(f"Found {len(json_files)} JSON files")
    
    for json_file in json_files:
        print(f"Processing file: {json_file}")
        dataset_name = extract_dataset_name(json_file.name)
        if dataset_name:
            print(f"Found dataset: {dataset_name}")
            dataset_files[dataset_name].append(json_file)
    
    print(f"Found {len(dataset_files)} datasets")
    
    # Process each dataset
    for dataset_name, files in dataset_files.items():
        print(f"\nProcessing dataset: {dataset_name}")
        print(f"Number of files: {len(files)}")
        
        # Read all JSON files for this dataset
        dfs = []
        for file in files:
            print(f"Reading file: {file}")
            with open(file, 'r') as f:
                data = json.load(f)
                print(f"Data shape: {len(data)} records")
                dfs.append(pd.DataFrame(data))
        
        # Calculate mean and std for each metric
        metrics = ['scoring_accuracy', 'scoring_f1', 'scoring_precision', 'scoring_recall', 'scoring_roc_auc']
        results = []
        
        for metric in metrics:
            # Stack all values for this metric
            values = np.array([df[metric].values for df in dfs])
            
            # Calculate mean and std
            mean_values = np.mean(values, axis=0)
            std_values = np.std(values, axis=0)
            
            # Create result rows
            for i, (mean, std) in enumerate(zip(mean_values, std_values)):
                results.append({
                    'module_name': dfs[0]['module_name'][i],
                    'metric': metric,
                    'mean': mean,
                    'std': std
                })
        
        # Convert to DataFrame
        result_df = pd.DataFrame(results)
        
        # Save pivot table version for better readability
        pivot_df = result_df.pivot(index='module_name', columns='metric', values=['mean', 'std'])
        pivot_df.columns = ['_'.join(col).strip('_') for col in pivot_df.columns.values]
        pivot_df = pivot_df.reset_index()
        
        output_file = output_path / f"{dataset_name}.csv"
        print(f"Saving to: {output_file}")
        pivot_df.to_csv(output_file)

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", type=str, required=True, help="Name of wandb project")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing experiment results")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir) / args.experiment_name
    output_dir = Path(f"{args.input_dir}_aggregated") / args.experiment_name
    
    aggregate_results(input_dir, output_dir)