"""CLI: build the GoEmotions multilabel JSON for AutoIntent. Logic lives in src/data.py.

Usage:
    uv run prepare_data.py                                       # full dataset -> data/go_emotions.json
    uv run prepare_data.py --min-samples-per-class 50            # stratified subsample (floor 50/class)
    uv run prepare_data.py --min-samples-per-class 50 --balance classwise  # flattened (cap ~50/class)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data import DEFAULT_CONFIG, DEFAULT_REPO, prepare_mapping, save_mapping
from src.naming import ensure_absent

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="HuggingFace dataset repo.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Dataset config name.")
    parser.add_argument("--out", type=Path, default=SCRIPT_DIR / "data" / "go_emotions.json", help="Output JSON path.")
    parser.add_argument(
        "--min-samples-per-class",
        type=int,
        default=None,
        help=(
            "Train subsample size control (default: full train). With --balance stratified it is a floor; "
            "with --balance classwise it is the per-class target/cap."
        ),
    )
    parser.add_argument(
        "--balance",
        choices=["stratified", "classwise"],
        default="stratified",
        help=(
            "stratified: proportion-preserving subsample (iterative-stratification). "
            "classwise: flatten the distribution by capping each class near the target (multilabel undersampling)."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the output file if it already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = ensure_absent(args.out, args.overwrite, label="Dataset file")
    mapping = prepare_mapping(args.repo, args.config, args.min_samples_per_class, args.balance, args.seed)
    save_mapping(mapping, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
