from dotenv import load_dotenv
import os
from autointent import setup_logging
from autointent import Dataset, Pipeline
from autointent.configs import LoggingConfig, DataConfig


datasets_names = ["DeepPavlov/banking77", "DeepPavlov/minds14", "DeepPavlov/hwu64", "DeepPavlov/snips", "DeepPavlov/massive", "DeepPavlov/clinc150"]

if __name__ == "__main__":
    from argparse import ArgumentParser
    import os
    from pathlib import Path
    import shutil
    from autointent.custom_types import SearchSpacePreset

    parser = ArgumentParser(description="This is the script for measuring the quality of AutoIntent using holdout validation. This script runs multiple seeds to get confidence estimations.")
    parser.add_argument("--experiment-name", type=str, required=True, help="aka name of the wandb project")
    parser.add_argument("--seeds", nargs="+")
    parser.add_argument("--validation-scheme", type=str, choices=["ho", "cv"], default="ho")
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--presets", nargs="+", type=str)
    args = parser.parse_args()

    load_dotenv()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    workdir = Path("experiments-assets") / args.experiment_name

    setup_logging(level="INFO", log_filename=workdir / "logs")

    os.environ["WANDB_PROJECT"] = args.experiment_name

    assert all(p in SearchSpacePreset.__args__ for p in args.presets)

    for seed in args.seeds:
        for preset in args.presets:
            for dataset in datasets_names:
                data_config = DataConfig(scheme=args.validation_scheme)

                logging_config = LoggingConfig(
                    run_name=dataset.split("/")[1] + f"[{seed=}]" + f"[{preset=}]",
                    clear_ram=True,
                    dump_modules=True,
                    report_to=["wandb", "codecarbon"],
                    project_dir=workdir
                )

                pipe = Pipeline.from_preset(preset, seed=seed)
                pipe.set_config(logging_config)
                pipe.set_config(data_config)
                intents_name = (
                    "intentsqwen3-32b" if dataset != "DeepPavlov/banking77" else "intents"
                )
                pipe.fit(Dataset.from_hub(dataset, intent_subset_name=intents_name), incompatible_search_space="filter")
                shutil.rmtree(logging_config.dirpath)
