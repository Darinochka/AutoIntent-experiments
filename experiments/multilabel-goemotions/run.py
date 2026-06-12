"""Optimize an AutoIntent pipeline on the GoEmotions multilabel dataset and dump metrics.

Expects the dataset JSON produced by ``prepare_data.py``. Builds a pipeline from a preset,
fits it (which also evaluates on the ``test`` split), and writes the final multilabel decision
metrics plus the selected module configs to ``<logs-dir>/<run-name>_metrics.json``.

Usage:
    uv run run.py                                   # classic-light on data/go_emotions.json
    uv run run.py --preset classic-medium --run-name gm-medium
    uv run run.py --scoring-metric scoring_map --decision-metric decision_f1  # override target metrics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autointent import Dataset, Pipeline
from autointent.configs import LoggingConfig
from autointent.metrics import DICISION_METRICS_MULTILABEL, SCORING_METRICS_MULTILABEL
from autointent.utils import load_preset

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=SCRIPT_DIR / "data" / "go_emotions.json", help="Dataset JSON.")
    parser.add_argument("--preset", default="classic-light", help="AutoIntent search-space preset.")
    parser.add_argument("--logs-dir", type=Path, default=SCRIPT_DIR / "logs", help="Directory for run logs/metrics.")
    parser.add_argument("--run-name", default=None, help="Run name (default: goemotions-<preset>).")
    parser.add_argument(
        "--embedder-model",
        default=None,
        help="Override the preset's embedder (e.g. sentence-transformers/all-MiniLM-L6-v2 for a fast CPU run).",
    )
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None, help="Torch device for the embedder.")
    parser.add_argument(
        "--scoring-metric",
        default=None,
        help="Override the scoring node's target_metric (selects the best scorer). E.g. scoring_map.",
    )
    parser.add_argument(
        "--decision-metric",
        default=None,
        help="Override the decision node's target_metric (selects the best decisioner). E.g. decision_f1.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def build_pipeline(preset: str, seed: int, scoring_metric: str | None, decision_metric: str | None) -> Pipeline:
    """Load a preset and optionally override the scoring/decision target metrics before building."""
    if scoring_metric and scoring_metric not in SCORING_METRICS_MULTILABEL:
        raise SystemExit(f"Invalid --scoring-metric '{scoring_metric}'. Choose from: {sorted(SCORING_METRICS_MULTILABEL)}")
    if decision_metric and decision_metric not in DICISION_METRICS_MULTILABEL:
        raise SystemExit(
            f"Invalid --decision-metric '{decision_metric}'. Choose from: {sorted(DICISION_METRICS_MULTILABEL)}"
        )

    cfg = load_preset(preset)
    for node in cfg["search_space"]:
        if node["node_type"] == "scoring" and scoring_metric:
            node["target_metric"] = scoring_metric
        elif node["node_type"] == "decision" and decision_metric:
            node["target_metric"] = decision_metric
    return Pipeline.from_optimization_config({**cfg, "seed": seed})


def selected_modules(context) -> list[dict]:  # noqa: ANN001
    """Best-effort extraction of the chosen module configs (never fatal)."""
    try:
        return [cfg.asdict() for cfg in context.optimization_info.get_inference_nodes_config()]
    except Exception as exc:  # noqa: BLE001
        return [{"error": f"could not serialize node configs: {exc}"}]


def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"goemotions-{args.preset}"

    if not args.data.exists():
        raise SystemExit(f"Dataset not found at {args.data}. Run prepare_data.py first.")

    dataset = Dataset.from_json(args.data)
    if not dataset.multilabel:
        raise SystemExit("Loaded dataset is not multilabel; check prepare_data.py output.")
    print(f"Loaded {args.data.name}: {dataset.n_classes} classes, multilabel={dataset.multilabel}")
    for split in dataset:
        print(f"  {split}: {len(dataset[split])} samples")

    pipeline = build_pipeline(args.preset, args.seed, args.scoring_metric, args.decision_metric)
    # dump_modules=True keeps fitted modules so final test metrics are computed and saved.
    pipeline.set_config(LoggingConfig(project_dir=args.logs_dir, run_name=run_name, dump_modules=True))
    emb_updates: dict[str, str] = {}
    if args.embedder_model:
        emb_updates["model_name"] = args.embedder_model
    if args.device:
        emb_updates["device"] = args.device
    if emb_updates:
        pipeline.set_config(pipeline.embedder_config.model_copy(update=emb_updates))
        print(f"Embedder overrides: {emb_updates}")

    print(f"Optimizing preset '{args.preset}' (run '{run_name}') ...")
    context = pipeline.fit(dataset)

    metrics = dict(context.optimization_info.pipeline_metrics)
    report = {
        "preset": args.preset,
        "run_name": run_name,
        "n_classes": dataset.n_classes,
        "split_sizes": {split: len(dataset[split]) for split in dataset},
        "target_metrics": {
            "scoring": args.scoring_metric or "preset-default",
            "decision": args.decision_metric or "preset-default",
        },
        "test_metrics": metrics,
        "selected_modules": selected_modules(context),
    }

    args.logs_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.logs_dir / f"{run_name}_metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== Test metrics (multilabel) ===")
    for name, value in metrics.items():
        print(f"  {name}: {value}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
