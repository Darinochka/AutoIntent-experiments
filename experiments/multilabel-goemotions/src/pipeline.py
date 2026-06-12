"""Build, fit, and report an AutoIntent pipeline for the GoEmotions multilabel task."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autointent import Dataset, Pipeline
from autointent.configs import LoggingConfig
from autointent.metrics import DICISION_METRICS_MULTILABEL, SCORING_METRICS_MULTILABEL
from autointent.utils import load_preset


def validate_metrics(scoring_metric: str | None, decision_metric: str | None) -> None:
    """Reject unknown target-metric names with the valid multilabel options."""
    if scoring_metric and scoring_metric not in SCORING_METRICS_MULTILABEL:
        raise SystemExit(f"Invalid --scoring-metric '{scoring_metric}'. Choose from: {sorted(SCORING_METRICS_MULTILABEL)}")
    if decision_metric and decision_metric not in DICISION_METRICS_MULTILABEL:
        raise SystemExit(
            f"Invalid --decision-metric '{decision_metric}'. Choose from: {sorted(DICISION_METRICS_MULTILABEL)}"
        )


def build_pipeline(preset: str, seed: int, scoring_metric: str | None, decision_metric: str | None) -> Pipeline:
    """Load a preset and optionally override the scoring/decision target metrics before building."""
    validate_metrics(scoring_metric, decision_metric)
    cfg = load_preset(preset)
    for node in cfg["search_space"]:
        if node["node_type"] == "scoring" and scoring_metric:
            node["target_metric"] = scoring_metric
        elif node["node_type"] == "decision" and decision_metric:
            node["target_metric"] = decision_metric
    return Pipeline.from_optimization_config({**cfg, "seed": seed})


def load_multilabel_dataset(data_path: str | Path) -> Dataset:
    """Load the dataset JSON and assert it is multilabel."""
    dataset = Dataset.from_json(data_path)
    if not dataset.multilabel:
        raise SystemExit("Loaded dataset is not multilabel; check prepare_data.py output.")
    return dataset


def selected_modules(context: Any) -> list[dict]:
    """Best-effort extraction of the chosen module configs (never fatal)."""
    try:
        return [cfg.asdict() for cfg in context.optimization_info.get_inference_nodes_config()]
    except Exception as exc:  # noqa: BLE001
        return [{"error": f"could not serialize node configs: {exc}"}]


def run_experiment(
    *,
    data_path: str | Path,
    preset: str,
    exp_name: str,
    logs_dir: str | Path,
    out_path: str | Path,
    embedder_model: str | None,
    device: str | None,
    scoring_metric: str | None,
    decision_metric: str | None,
    seed: int,
    dump_modules: bool = True,
) -> dict:
    """Optimize the pipeline, evaluate on test, write the report, and return it.

    dump_modules persists fitted modules to disk for reuse; final test metrics are computed regardless
    (clear_ram is left False). Sweeps pass dump_modules=False to avoid writing a module dump per run.
    """
    dataset = load_multilabel_dataset(data_path)
    print(f"Loaded {Path(data_path).name}: {dataset.n_classes} classes, multilabel={dataset.multilabel}")
    for split in dataset:
        print(f"  {split}: {len(dataset[split])} samples")

    pipeline = build_pipeline(preset, seed, scoring_metric, decision_metric)
    pipeline.set_config(LoggingConfig(project_dir=Path(logs_dir), run_name=exp_name, dump_modules=dump_modules))

    emb_updates: dict[str, str] = {}
    if embedder_model:
        emb_updates["model_name"] = embedder_model
    if device:
        emb_updates["device"] = device
    if emb_updates:
        pipeline.set_config(pipeline.embedder_config.model_copy(update=emb_updates))
        print(f"Embedder overrides: {emb_updates}")

    # Snapshot the splits we feed before fit() (AutoIntent carves an HPO-validation out of train in place).
    fed_split_sizes = {split: len(dataset[split]) for split in dataset}

    print(f"Optimizing preset '{preset}' (experiment '{exp_name}') ...")
    context = pipeline.fit(dataset)

    metrics = dict(context.optimization_info.pipeline_metrics)
    report = {
        "preset": preset,
        "exp_name": exp_name,
        "n_classes": dataset.n_classes,
        "fed_split_sizes": fed_split_sizes,
        "eval_on": "ai-test (= GoEmotions validation); HPO-validation carved from train by AutoIntent",
        "target_metrics": {
            "scoring": scoring_metric or "preset-default",
            "decision": decision_metric or "preset-default",
        },
        "test_metrics": metrics,
        "selected_modules": selected_modules(context),
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== Test metrics (multilabel) ===")
    for name, value in metrics.items():
        print(f"  {name}: {value}")
    print(f"\nWrote {out_path}")
    return report
