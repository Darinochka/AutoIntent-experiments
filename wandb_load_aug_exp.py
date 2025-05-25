from cycler import V


def parse_group_name(name: str):
    parts = name.split("_")
    llm_name = parts[0]
    naug = parts[-1]
    dataset = parts[1]
    if parts[2] == "ru":
        dataset += "_ru"
    return llm_name, naug, dataset

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

    projects = api.projects(entity="darinka")

    savedir = Path("wandb_aug_exps")
    savedir.mkdir(parents=True, exist_ok=True)

    # llm / dataset / n-aug
    results = {}

    for a_project in projects:
        if a_project.name not in args.projects:
            continue
        runs = api.runs(f"darinka/{a_project.name}", filters={"displayName": "final_metrics"})

        for i, run in enumerate(runs):
            if run.name != "final_metrics":
                continue

            logger.info(
                f"Processing run {run.name} in group {run.group} from experiment {a_project.name}"
            )
            with tempfile.TemporaryDirectory() as tempdir:
                llm, naug, dataset = parse_group_name(run.group)
                if llm not in results:
                    results[llm] = {}
                if dataset not in results[llm]:
                    results[llm][dataset] = {}
                if naug not in results[llm][dataset]:
                    results[llm][dataset][naug] = {}
                results[llm][dataset][naug] = {
                    k: v
                    for k, v in run.summary.items() if k.startswith("decision")
                }

            with (savedir / "results.json").open("w") as file:
                json.dump(results, file)
