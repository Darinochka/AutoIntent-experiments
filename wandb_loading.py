if __name__ == "__main__":
    # script to get all wandb run data, built from their api examples
    import pandas as pd
    import wandb
    from pathlib import Path
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("projects", nargs="+")
    args = parser.parse_args()

    api = wandb.Api()

    projects = api.projects()

    savedir = Path("wandb_downloads")

    for a_project in projects:
        if a_project.name not in args.projects:
            continue
        project_savedir = savedir / a_project.name
        run_list, config_list, name_list = [], [], []
        runs = api.runs(f"ilya_alekseev_2016/{a_project.name}")
        for i, run in enumerate(runs):
            if run.name != "final_metrics":
                continue
            
            print(f"Processing run {run.name} in group {run.group} from experiment {a_project.name}")

            run_list.append(run.summary._json_dict)

            config_list.append(
                {k: v for k, v in run.config.items() if not k.startswith("_")}
            )
            name_list.append(run.name)

            run_save_path = project_savedir / str(run.group)
            run_save_path.mkdir(parents=True, exist_ok=True)
            final_config = next(file for file in run.files() if Path(file.name).name == "config.yaml")
            final_config.download(root=run_save_path)

        runs_df = pd.DataFrame(
            {"summary": run_list, "config": config_list, "name": name_list}
        )

        csv_savedir = project_savedir / "csv"
        csv_savedir.mkdir(parents=True)
        runs_df.to_csv(csv_savedir / f"{a_project.name}.csv")