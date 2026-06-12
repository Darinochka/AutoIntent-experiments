"""CLI: optimize an AutoIntent pipeline on GoEmotions and dump metrics. Logic lives in src/pipeline.py.

Usage:
    uv run run.py                                   # classic-light on data/go_emotions.json
    uv run run.py --preset classic-medium --exp-name gm-medium
    uv run run.py --scoring-metric scoring_map --decision-metric decision_f1  # override target metrics
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.naming import ensure_absent, metrics_path
from src.pipeline import run_experiment

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=SCRIPT_DIR / "data" / "go_emotions.json", help="Dataset JSON.")
    parser.add_argument("--preset", default="classic-light", help="AutoIntent search-space preset.")
    parser.add_argument("--logs-dir", type=Path, default=SCRIPT_DIR / "logs", help="Directory for run logs/metrics.")
    parser.add_argument("--exp-name", default=None, help="Experiment name (default: goemotions-<preset>).")
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
    parser.add_argument("--overwrite", action="store_true", help="Replace existing metrics/run outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_name = args.exp_name or f"goemotions-{args.preset}"

    out_path = ensure_absent(metrics_path(args.logs_dir, exp_name), args.overwrite, label="Metrics file")
    ensure_absent(Path(args.logs_dir) / exp_name, args.overwrite, label="Run directory")

    if not args.data.exists():
        raise SystemExit(f"Dataset not found at {args.data}. Run prepare_data.py first.")

    run_experiment(
        data_path=args.data,
        preset=args.preset,
        exp_name=exp_name,
        logs_dir=args.logs_dir,
        out_path=out_path,
        embedder_model=args.embedder_model,
        device=args.device,
        scoring_metric=args.scoring_metric,
        decision_metric=args.decision_metric,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
