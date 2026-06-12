"""CLI: sweep scoring/decision target metrics across dataset sizes and balance modes.

Logic lives in src/sweep.py. Each (size, balance, scoring_metric, decision_metric) is one AutoIntent
optimization; results land in <logs-dir>/sweep_summary.csv plus per-run <exp>_metrics.json.

Usage:
    uv run sweep.py --dry-run                       # print the run plan only
    uv run sweep.py --device mps                    # full grid on mps (default 3 sizes x 2 balances x 9x4)
    uv run sweep.py --device mps \\
        --sizes 100 --balances classwise stratified \\
        --scoring-metrics scoring_f1 scoring_map \\
        --decision-metrics decision_f1 decision_accuracy   # trimmed grid

Resumable: re-running skips cells whose <exp>_metrics.json already exists (use --overwrite to force).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.sweep import ALL_DECISION, ALL_SCORING, run_sweep

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 100, 500], help="Shot counts (per-class budget).")
    parser.add_argument(
        "--balances",
        nargs="+",
        default=["classwise", "stratified"],
        choices=["classwise", "natural", "stratified"],
        help=(
            "classwise=balanced; stratified=imbalanced floor (collapses to full at high N); "
            "natural=size-matched imbalanced (opt-in; cells missing a class are skipped)."
        ),
    )
    parser.add_argument("--scoring-metrics", nargs="+", default=ALL_SCORING, help="Scoring-node target metrics to try.")
    parser.add_argument(
        "--decision-metrics", nargs="+", default=ALL_DECISION, help="Decision-node target metrics to try."
    )
    parser.add_argument("--preset", default="classic-light", help="AutoIntent search-space preset.")
    parser.add_argument("--embedder-model", default=None, help="Override the preset's embedder.")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None, help="Torch device for the embedder.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--logs-dir", type=Path, default=SCRIPT_DIR / "logs", help="Directory for logs/metrics/summary.")
    parser.add_argument("--data-dir", type=Path, default=SCRIPT_DIR / "data", help="Directory for prepared datasets.")
    parser.add_argument("--overwrite", action="store_true", help="Rerun cells even if their metrics already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print the run plan and exit (no datasets, no fits).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_sweep(
        sizes=args.sizes,
        balances=args.balances,
        scoring_metrics=args.scoring_metrics,
        decision_metrics=args.decision_metrics,
        preset=args.preset,
        embedder_model=args.embedder_model,
        device=args.device,
        seed=args.seed,
        logs_dir=args.logs_dir,
        data_dir=args.data_dir,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
