if __name__ == "__main__":
    import logging
    from argparse import ArgumentParser
    from pathlib import Path

    import wandb

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
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
        already_processed = list(
            directory.name
            for directory in project_savedir.iterdir()
            if directory.is_dir()
        )
        for i, run in enumerate(runs):
            if run.name != "final_metrics":
                continue
            if run.group in already_processed:
                logger.info(
                    f"Skipping run {run.name} in group {run.group} from experiment {a_project.name} because it has already been processed"
                )
                continue

            logger.info(
                f"Processing run {run.name} in group {run.group} from experiment {a_project.name}"
            )

            run_save_path = project_savedir / str(run.group)
            run_save_path.mkdir(parents=True, exist_ok=True)
            final_config = next(
                file for file in run.files() if Path(file.name).name == "config.yaml"
            )
            final_config.download(root=run_save_path)
