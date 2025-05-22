if __name__ == "__main__":
    import json
    import logging
    import tempfile
    from argparse import ArgumentParser
    from collections import defaultdict
    from io import TextIOWrapper
    from pathlib import Path

    import yaml

    import wandb

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    parser = ArgumentParser()
    parser.add_argument("projects", nargs="+")
    args = parser.parse_args()

    api = wandb.Api()

    projects = api.projects()

    savedir = Path("wandb_oos")

    for a_project in projects:
        if a_project.name not in args.projects:
            continue
        project_savepath = savedir / a_project.name / "results.json"
        runs = api.runs(f"ilya_alekseev_2016/{a_project.name}")
        project_savepath.parent.mkdir(parents=True, exist_ok=True)
        if project_savepath.exists():
            logger.info(f"Loading results from {project_savepath}")
            with project_savepath.open("r") as f:
                groupwise_results = json.load(f)
        else:
            groupwise_results = defaultdict(list)
        for i, run in enumerate(runs):

            logger.info(
                f"Processing run {run.name} in group {run.group} from experiment {a_project.name}"
            )

            # retrieve config and metrics summary
            files_to_download = ["config.yaml", "wandb-summary.json"]

            # config may be not present, i.e. no hyperparameters were passed
            run_results = {
                "config": {},
            }

            with tempfile.TemporaryDirectory() as tempdir:
                for file_name in files_to_download:
                    file_to_download = next(
                        (file for file in run.files() if Path(file.name).name == file_name),
                        None,
                    )
                    if file_to_download is None:
                        logger.warning(f"File {file_name} not found in run {run.name}")
                        continue

                    wrapper: TextIOWrapper = file_to_download.download(
                        root=str(Path(tempdir) / file_name)
                    )
                    if file_name == "config.yaml":
                        loaded = yaml.safe_load(wrapper)
                        loaded.pop("_wandb")
                    else:
                        loaded = json.load(wrapper)
                    run_results[file_name.split(".")[0]] = loaded

            groupwise_results[run.group].append(
                {
                    "module_name": run.name,
                    **run_results,
                }
            )
            with project_savepath.open("w") as f:
                json.dump(groupwise_results, f, indent=4, ensure_ascii=False)
