"""Convert the GoEmotions dataset into AutoIntent's multilabel JSON format.

GoEmotions (``google-research-datasets/go_emotions``, ``simplified`` config) ships each
example as ``text`` plus ``labels`` (a list of emotion class indices). AutoIntent expects
multilabel samples as one-hot vectors: ``{"utterance": ..., "label": [0, 1, 0, ...]}`` with an
explicit ``intents`` list so the label space is sized correctly.

Usage:
    uv run prepare_data.py                                       # full dataset -> data/go_emotions.json
    uv run prepare_data.py --min-samples-per-class 50            # stratified subsample (floor 50/class)
    uv run prepare_data.py --min-samples-per-class 50 --balance classwise  # flattened (cap ~50/class)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from datasets import load_dataset
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

SCRIPT_DIR = Path(__file__).resolve().parent

# GoEmotions split name -> AutoIntent split name.
SPLIT_MAP = {"train": "train", "validation": "validation", "test": "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="google-research-datasets/go_emotions", help="HuggingFace dataset repo.")
    parser.add_argument("--config", default="simplified", help="Dataset config name.")
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
    return parser.parse_args()


def label_matrix(rows: list[dict], n_classes: int) -> np.ndarray:
    """Build a (n_samples, n_classes) one-hot matrix from GoEmotions label-index lists."""
    y = np.zeros((len(rows), n_classes), dtype=int)
    for i, ex in enumerate(rows):
        for idx in ex["labels"]:
            y[i, idx] = 1
    return y


def balanced_subsample(rows: list[dict], n_classes: int, min_per_class: int, seed: int) -> list[dict]:
    """Smallest label-stratified subsample (iterative-stratification) where each class clears a floor.

    The subset size is chosen so the rarest present class reaches ~min_per_class samples while the
    overall label distribution is preserved. If the floor cannot be met (a class has too few total
    examples), all rows are kept.
    """
    y = label_matrix(rows, n_classes)
    class_counts = y.sum(axis=0)
    present = class_counts > 0
    # Proportion-preserving sampling keeps each class at count * (target / n); invert for the rarest.
    needed = (min_per_class * len(rows) / class_counts[present]).max()
    target = math.ceil(needed)
    if target >= len(rows):
        print(f"Cannot reduce train below {len(rows)} rows for min-samples-per-class={min_per_class}; keeping all.")
        return rows

    # test_size is passed explicitly: iterstrat 0.1.9's "default" sentinel is rejected by modern sklearn.
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, train_size=target, test_size=len(rows) - target, random_state=seed
    )
    train_idx, _ = next(splitter.split(np.zeros((len(rows), 1)), y))
    subset = [rows[i] for i in train_idx]

    rarest = int(label_matrix(subset, n_classes).sum(axis=0)[present].min())
    print(f"Balanced-subsampled train to {len(subset)} rows (rarest class has {rarest}, floor {min_per_class}).")
    if rarest < min_per_class:
        print(f"  Warning: rarest class below floor ({rarest} < {min_per_class}); data too imbalanced to guarantee it.")
    return subset


def classwise_subsample(rows: list[dict], n_classes: int, target_per_class: int, seed: int) -> list[dict]:
    """Flatten the label distribution by capping each class near target_per_class (multilabel undersampling).

    Shuffles, then keeps a row only while at least one of its labels is still below the target. Majority
    classes settle near the target; rarer classes keep all available samples. Exact equality is impossible
    in multilabel data since each row carries several labels, so capped classes may slightly overshoot.
    """
    order = np.random.default_rng(seed).permutation(len(rows))
    counts = np.zeros(n_classes, dtype=int)
    keep = []
    for i in order:
        labels = rows[i]["labels"]
        if labels and any(counts[label] < target_per_class for label in labels):
            keep.append(int(i))
            for label in labels:
                counts[label] += 1
    subset = [rows[i] for i in keep]

    present = label_matrix(rows, n_classes).sum(axis=0) > 0
    sub_counts = label_matrix(subset, n_classes).sum(axis=0)[present]
    print(
        f"Classwise-subsampled train to {len(subset)} rows "
        f"(per-class min/median/max = {int(sub_counts.min())}/{int(np.median(sub_counts))}/{int(sub_counts.max())}, "
        f"target {target_per_class})."
    )
    return subset


def to_autointent_split(examples: list[dict], n_classes: int) -> list[dict]:
    """Map GoEmotions rows to AutoIntent one-hot multilabel samples (dropping label-less rows)."""
    samples = []
    for ex in examples:
        label_ids = ex["labels"]
        if not label_ids:  # multilabel samples must have at least one active class
            continue
        one_hot = [0] * n_classes
        for idx in label_ids:
            one_hot[idx] = 1
        samples.append({"utterance": ex["text"], "label": one_hot})
    return samples


def main() -> None:
    args = parse_args()

    print(f"Loading {args.repo} ({args.config}) ...")
    ds = load_dataset(args.repo, args.config)

    names = ds["train"].features["labels"].feature.names
    n_classes = len(names)
    intents = [{"id": i, "name": name} for i, name in enumerate(names)]
    print(f"{n_classes} emotion classes: {', '.join(names)}")

    mapping: dict[str, list[dict]] = {"intents": intents}
    for hf_split, ai_split in SPLIT_MAP.items():
        if hf_split not in ds:
            continue
        rows = list(ds[hf_split])
        if ai_split == "train" and args.min_samples_per_class is not None:
            if args.balance == "classwise":
                rows = classwise_subsample(rows, n_classes, args.min_samples_per_class, args.seed)
            else:
                rows = balanced_subsample(rows, n_classes, args.min_samples_per_class, args.seed)
        samples = to_autointent_split(rows, n_classes)
        mapping[ai_split] = samples
        print(f"  {ai_split}: {len(samples)} samples (dropped {len(rows) - len(samples)} label-less)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
