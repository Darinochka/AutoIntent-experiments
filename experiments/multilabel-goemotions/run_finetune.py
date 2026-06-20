"""CLI: Phase-4 fine-tuning runner — AutoIntent `transformers` preset (bert scoring node) on MPS.

Phase 1-3 used `classic-light` (a frozen e5 embedder + shallow head) and showed a hard macro-F1 ceiling
~0.35 that more data does not lift (REPORT §9), implying the frozen representation — not data — is the
wall. Phase 4 tests that directly by *fine-tuning* a transformer end-to-end on the same balanced caps.

It starts from the `transformers-no-hpo` preset (a single `bert` scoring config so the transformer trains
once, not once-per-HPO-trial) and overrides:
  * model -> `bert-base-uncased` (cached, so it runs offline, and it is the exact model of the published
    GoEmotions BERT baseline = 0.46 macro-F1 -> apples-to-apples),
  * device -> mps, epochs / batch_size / learning_rate -> a small-data BERT-FT recipe,
  * scoring target -> scoring_f1, decision target -> decision_f1 (matches the Phase-3 headline),
  * hpo n_trials -> small (the single scoring config trains once; trials only tune the decision threshold).

Usage:
    uv run run_finetune.py --data data/ge_classwise_100_s1_test.json --exp-name ft-cw100-s1 \
        --device mps --epochs 15 --batch-size 32 --learning-rate 3e-5 --n-trials 5 --eval-split test
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Any, Literal

from autointent import Pipeline
from autointent.configs import LoggingConfig
from autointent.utils import load_preset
from cyclopts import App, Parameter
from loguru import logger

from src.naming import ensure_absent, metrics_path
from src.pipeline import load_multilabel_dataset, selected_modules

EXP_DIR = Path(__file__).resolve().parent
app = App(help="Phase-4: fine-tune a transformer (bert) scoring node on GoEmotions multilabel.")


@dataclass(frozen=True)
class FinetuneConfig:
    """Configuration for one fine-tuning run."""

    data: Annotated[Path, Parameter(help="Dataset JSON (built by sweep/prepare).")] = (
        EXP_DIR / "data" / "ge_classwise_100_s1_test.json"
    )
    exp_name: Annotated[str, Parameter(help="Experiment name (logs/<name>_metrics.json).")] = "ft-cw100-s1"
    base_preset: Annotated[str, Parameter(help="AutoIntent preset to start from.")] = "transformers-no-hpo"
    model_name: Annotated[str, Parameter(help="HF model to fine-tune (cached for offline).")] = "bert-base-uncased"
    device: Annotated[Literal["cpu", "cuda", "mps"], Parameter(help="Torch device.")] = "mps"
    epochs: Annotated[int, Parameter(help="Train epochs (full, since early stopping is disabled by default).")] = 15
    batch_size: Annotated[int, Parameter(help="Train batch size (kept modest for MPS memory).")] = 32
    learning_rate: Annotated[float, Parameter(help="AdamW LR. 2e-5 learns; 3e-5/1e-5 collapse (see diag).")] = 2e-5
    disable_early_stopping: Annotated[
        bool,
        Parameter(help="Disable the preset's scoring_f1@0.5 early stop (broken for sparse multilabel: probs<0.5)."),
    ] = True
    n_trials: Annotated[int, Parameter(help="HPO trials (decision threshold tuning; scoring is fixed).")] = 5
    scoring_metric: Annotated[str, Parameter(help="Scoring-node target metric.")] = "scoring_f1"
    decision_metric: Annotated[str, Parameter(help="Decision-node target metric (matches Phase-3 headline).")] = (
        "decision_f1"
    )
    eval_split: Annotated[Literal["validation", "test"], Parameter(help="Provenance of the held-out eval.")] = "test"
    seed: Annotated[int, Parameter(help="Random seed.")] = 1
    logs_dir: Annotated[Path, Parameter(help="Directory for logs/metrics.")] = EXP_DIR / "logs"
    overwrite: Annotated[bool, Parameter(help="Replace existing metrics/run outputs.")] = False


_DEFAULTS = FinetuneConfig()


def build_finetune_config(cfg: FinetuneConfig) -> dict[str, Any]:
    """Load the transformers preset and inject the Phase-4 overrides; return an optimization config dict."""
    space = load_preset(cfg.base_preset)  # type: ignore[arg-type]
    for node in space["search_space"]:
        if node["node_type"] == "scoring":
            node["target_metric"] = cfg.scoring_metric
            for module in node["search_space"]:
                if module.get("module_name") == "bert":
                    module["classification_model_config"] = [{"model_name": cfg.model_name, "device": cfg.device}]
                    module["num_train_epochs"] = [cfg.epochs]
                    module["batch_size"] = [cfg.batch_size]
                    module["learning_rate"] = [cfg.learning_rate]
                    if cfg.disable_early_stopping:
                        # The preset early-stops on scoring_f1@0.5; sparse-multilabel probs sit <0.5 so the
                        # metric is ~0 from step 1 -> it stops immediately and restores a near-random model.
                        module["early_stopping_config"] = [{"metric": None}]
        elif node["node_type"] == "decision":
            node["target_metric"] = cfg.decision_metric
    space.setdefault("hpo_config", {})
    space["hpo_config"]["n_trials"] = cfg.n_trials
    return {**space, "seed": cfg.seed}


@app.default
def main(cfg: Annotated[FinetuneConfig, Parameter(name="*")] = _DEFAULTS) -> None:
    """Fine-tune and evaluate one (dataset, recipe) cell; write a metrics JSON compatible with the sweep logs."""
    out_path = metrics_path(cfg.logs_dir, cfg.exp_name)
    ensure_absent(out_path, cfg.overwrite, label="Metrics file")
    ensure_absent(Path(cfg.logs_dir) / cfg.exp_name, cfg.overwrite, label="Run directory")
    if not cfg.data.exists():
        raise SystemExit(f"Dataset not found at {cfg.data}. Build it with sweep.py --sizes ... first.")

    dataset = load_multilabel_dataset(cfg.data)
    logger.info("Loaded {}: {} classes, multilabel={}", cfg.data.name, dataset.n_classes, dataset.multilabel)
    fed_split_sizes = {split: len(dataset[split]) for split in dataset}

    config = build_finetune_config(cfg)
    pipeline = Pipeline.from_optimization_config(config)
    pipeline.set_config(LoggingConfig(project_dir=Path(cfg.logs_dir), run_name=cfg.exp_name, dump_modules=False))

    logger.info(
        "Fine-tuning {} on {} (epochs<={}, bs={}, lr={}, n_trials={}, device={}) ...",
        cfg.model_name,
        cfg.data.name,
        cfg.epochs,
        cfg.batch_size,
        cfg.learning_rate,
        cfg.n_trials,
        cfg.device,
    )
    context = pipeline.fit(dataset)
    metrics = dict(context.optimization_info.pipeline_metrics)

    report: dict[str, Any] = {
        "phase": 4,
        "preset": cfg.base_preset,
        "autointent_version": version("autointent"),
        "exp_name": cfg.exp_name,
        "finetune": {
            "model_name": cfg.model_name,
            "device": cfg.device,
            "epochs_max": cfg.epochs,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "disable_early_stopping": cfg.disable_early_stopping,
            "n_trials": cfg.n_trials,
        },
        "n_classes": dataset.n_classes,
        "fed_split_sizes": fed_split_sizes,
        "eval_split": cfg.eval_split,
        "target_metrics": {"scoring": cfg.scoring_metric, "decision": cfg.decision_metric},
        "test_metrics": metrics,
        "selected_modules": selected_modules(context),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Test metrics (multilabel):")
    for name, value in metrics.items():
        logger.info("  {}: {}", name, value)
    logger.info("Wrote {}", out_path)


if __name__ == "__main__":
    app()
