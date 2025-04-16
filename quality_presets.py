from dotenv import load_dotenv
import os
from autointent import setup_logging
from autointent import Dataset, Pipeline
from autointent.configs import LoggingConfig, EmbedderConfig, DataConfig


datasets_names = ["DeepPavlov/banking77", "DeepPavlov/minds14", "DeepPavlov/hwu64", "DeepPavlov/snips", "DeepPavlov/massive"]

if __name__ == "__main__":
    from argparse import ArgumentParser
    import os
    from pathlib import Path
    import shutil
    from autointent.custom_types import SearchSpacePreset

    parser = ArgumentParser(description="This is the script for measuring the quality of AutoIntent using holdout validation. This script runs multiple seeds to get confidence estimations.")
    parser.add_argument("--experiment-name", type=str, required=True, help="aka name of the wandb project")
    parser.add_argument("--embedder-name", type=str, default=None, help="Name of HF repository. Omit this param to use AutoIntent's default embedder.")
    parser.add_argument("--seeds", nargs="+")
    parser.add_argument("--validation-scheme", type=str, choices=["ho", "cv"])
    parser.add_argument("--cuda", type=str, default="0")
    args = parser.parse_args()

    load_dotenv()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    workdir = Path("experiments-assets") / args.experiment_name

    setup_logging(level="INFO", log_filename=workdir / "logs")

    os.environ["WANDB_PROJECT"] = args.experiment_name

    for seed in args.seeds:
        for preset in SearchSpacePreset.__args__:
            for dataset in datasets_names:
                data_config = DataConfig(scheme=args.validation_scheme)

                logging_config = LoggingConfig(
                    run_name=dataset.split("/")[1] + f"[{seed=}]" + f"[{preset=}]",
                    clear_ram=True,
                    dump_modules=True,
                    report_to=["wandb"],
                    project_dir=workdir
                )
                
                if args.embedder_name is None:
                    embedder_config = EmbedderConfig()
                else:
                    embedder_config = EmbedderConfig(model_name=args.embedder_name)

                pipe = Pipeline.from_preset(preset, seed=seed)
                pipe.set_config(logging_config)
                pipe.set_config(embedder_config)
                pipe.set_config(data_config)

                pipe.fit(Dataset.from_hub(dataset), incompatible_search_space="filter")
                shutil.rmtree(logging_config.dirpath)
