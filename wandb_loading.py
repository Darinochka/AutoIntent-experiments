if __name__ == "__main__":
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
        runs = api.runs(f"ilya_alekseev_2016/{a_project.name}")
        for i, run in enumerate(runs):
            if run.name != "final_metrics":
                continue
            
            print(f"Processing run {run.name} in group {run.group} from experiment {a_project.name}")

            run_save_path = project_savedir / str(run.group)
            run_save_path.mkdir(parents=True, exist_ok=True)
            final_config = next(file for file in run.files() if Path(file.name).name == "config.yaml")
            final_config.download(root=run_save_path)
