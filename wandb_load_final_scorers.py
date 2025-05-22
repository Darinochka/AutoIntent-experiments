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

    projects = api.projects(entity="samoed-roman")

    savedir = Path("wandb_encoders")

    for a_project in projects:
        if a_project.name not in args.projects:
            continue
        project_savepath = savedir / a_project.name / "results.json"
        runs = api.runs(f"samoed-roman/{a_project.name}", filters={"displayName": "final_metrics"})

        project_savepath.parent.mkdir(parents=True, exist_ok=True)
        cur_res = []
        for i, run in enumerate(runs):
            if run.name != "final_metrics":
                continue

            logger.info(
                f"Processing run {run.name} in group {run.group} from experiment {a_project.name}"
            )

            # retrieve config and metrics summary
            files_to_download = [ "wandb-summary.json"]

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
                    else:
                        loaded = json.load(wrapper)["pipeline_metrics"]
                    run_results[file_name.split(".")[0]] = loaded
            try:
                parts = run.group.split("_")
                dataset = parts[0]
                encoder = "_".join(parts[1:-1])
                if parts[-1] == "True":
                    continue
            except ValueError:
                logger.warning(f"Failed to parse group {run.group}")
                raise
            res = {
                "dataset": dataset,
                "encoder": encoder,
                "metrics": run_results["wandb-summary"],
            }
            cur_res.append(res)
            with project_savepath.open("w") as f:
                json.dump(cur_res, f, indent=4, ensure_ascii=False)
