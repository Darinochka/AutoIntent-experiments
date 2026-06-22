if __name__ == "__main__":
    import logging
    from argparse import ArgumentParser
    from io import TextIOWrapper
    from pathlib import Path
    import yaml
    import tempfile
    import json

    import wandb

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    parser = ArgumentParser()
    parser.add_argument("projects", nargs="+")
    args = parser.parse_args()

    api = wandb.Api()

    projects = api.projects()

    savedir = Path("wandb_compute_exps")
    savedir.mkdir(parents=True, exist_ok=True)
    savepath = savedir / "results.json"

    # results[scorer_name] = [res1, res2, ...]
    results = {}

    if savepath.exists():
        with savepath.open("r") as file:
            results = json.load(file)

        logger.info(f"Loaded results from {savepath} with keys: {results.keys()}")

    for a_project in projects:
        if a_project.name not in args.projects:
            continue
        runs = api.runs(f"ilya_alekseev_2016/{a_project.name}")

        for i, run in enumerate(runs):
            if run.name in ["final_metrics", "argmax_0"]:
                continue

            logger.info(
                f"Processing run {run.name} in group {run.group} from experiment {a_project.name}"
            )

            files_to_download = ["config.yaml", "wandb-summary.json"]
            run_results = {
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
                        run_results["config"] = yaml.safe_load(wrapper)
                        run_results["config"].pop("_wandb")
                    if file_name == "wandb-summary.json":
                        run_results["metrics"] = json.load(wrapper)

                scorer_name = run.name.split("_")[0]
                logger.info(f"Scorer: {scorer_name}")

                if scorer_name not in results:
                    results[scorer_name] = []
                results[scorer_name].append(run_results)

            with savepath.open("w") as file:
                json.dump(results, file)
