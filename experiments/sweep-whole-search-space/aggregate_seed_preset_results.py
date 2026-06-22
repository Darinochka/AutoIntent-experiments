import numpy as np
from pathlib import Path
import re
from collections import defaultdict
import yaml

def extract_dataset_and_preset(filename):
    """Extract dataset name and preset from filename pattern: <dataset>[seed='<seed>'][preset='<preset>'].yaml"""
    match = re.match(r"(.+?)\[seed='\d+'\].*?\[preset='(.+?)'\]\.yaml", filename)
    if match:
        return match.group(1), match.group(2)
    return None, None

def aggregate_results(input_dir, output_dir):
    """
    Aggregate results across different seeds for each dataset and preset combination.
    
    Args:
        input_dir (str): Directory containing the YAML files
        output_dir (str): Directory to save aggregated results
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print(f"Input directory: {input_path}")
    print(f"Output directory: {output_path}")
    
    # Group files by dataset and preset
    dataset_preset_files = defaultdict(list)
    yaml_files = list(input_path.glob("*.yaml"))
    print(f"Found {len(yaml_files)} YAML files")
    
    for yaml_file in yaml_files:
        print(f"Processing file: {yaml_file}")
        dataset_name, preset = extract_dataset_and_preset(yaml_file.name)
        if dataset_name and preset:
            print(f"Found dataset: {dataset_name}, preset: {preset}")
            dataset_preset_files[(dataset_name, preset)].append(yaml_file)
    
    print(f"Found {len(dataset_preset_files)} dataset-preset combinations")
    
    # Process each dataset-preset combination
    for (dataset_name, preset), files in dataset_preset_files.items():
        print(f"\nProcessing dataset: {dataset_name}, preset: {preset}")
        print(f"Number of files: {len(files)}")
        
        # Initialize lists to store metrics across seeds
        metrics = defaultdict(list)
        
        # Read metrics from each file
        for file in files:
            with open(file, 'r') as f:
                data = yaml.safe_load(f)
                for metric, value in data.items():
                    metrics[metric].append(value)
        
        # Calculate mean and std for each metric
        results = {}
        for metric, values in metrics.items():
            results[f"{metric}_mean"] = float(np.mean(values))
            results[f"{metric}_std"] = float(np.std(values))
        
        # Save aggregated results
        output_file = output_path / f"{dataset_name}[preset='{preset}'].yaml"
        with open(output_file, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        
        print(f"Saved aggregated results to {output_file}")
    
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