if __name__ == "__main__":
    from pathlib import Path
    import yaml
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--expriment-name", type=str, required=True, help="Name of wandb project")
    parser.add_argument("--configs-dir", type=str, default="wandb_downloads")
    parser.add_argument("--output-dir", type=str, default="scoring_results")
    args = parser.parse_args()

    configs_dir = Path(args.configs_dir) / args.experiment_name
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(exist_ok=True)

    for dataset_dir in configs_dir.iterdir():
        config_file = dataset_dir / "config.yaml"
        config = yaml.safe_load(config_file.read_text())
        scoring_modules = config["configs"]["value"]["scoring"]
        with (output_dir / f"{dataset_dir.name}.yaml").open("w") as file:
            yaml.dump(scoring_modules, file)