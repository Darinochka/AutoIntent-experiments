"""GoEmotions loading, multilabel one-hot conversion, and train-subsampling helpers.

GoEmotions (``google-research-datasets/go_emotions``, ``simplified`` config) ships each example as
``text`` plus ``labels`` (a list of emotion class indices). AutoIntent expects multilabel samples as
one-hot vectors with an explicit ``intents`` list so the label space is sized correctly.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from datasets import load_dataset
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

DEFAULT_REPO = "google-research-datasets/go_emotions"
DEFAULT_CONFIG = "simplified"

# GoEmotions split name -> AutoIntent split name.
SPLIT_MAP = {"train": "train", "validation": "validation", "test": "test"}


def load_goemotions(repo: str = DEFAULT_REPO, config: str = DEFAULT_CONFIG) -> tuple[dict, list[str]]:
    """Load the dataset and return it alongside the ordered emotion class names."""
    ds = load_dataset(repo, config)
    names = ds["train"].features["labels"].feature.names
    return ds, names


def label_matrix(rows: list[dict], n_classes: int) -> np.ndarray:
    """Build a (n_samples, n_classes) one-hot matrix from GoEmotions label-index lists."""
    y = np.zeros((len(rows), n_classes), dtype=int)
    for i, ex in enumerate(rows):
        for idx in ex["labels"]:
            y[i, idx] = 1
    return y


def stratified_subsample(rows: list[dict], n_classes: int, min_per_class: int, seed: int) -> list[dict]:
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
    print(f"Stratified-subsampled train to {len(subset)} rows (rarest class has {rarest}, floor {min_per_class}).")
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


def subsample_train(rows: list[dict], n_classes: int, min_per_class: int | None, balance: str, seed: int) -> list[dict]:
    """Dispatch to the requested subsampler, or keep all rows when min_per_class is None."""
    if min_per_class is None:
        return rows
    if balance == "classwise":
        return classwise_subsample(rows, n_classes, min_per_class, seed)
    return stratified_subsample(rows, n_classes, min_per_class, seed)


def to_onehot_samples(rows: list[dict], n_classes: int) -> list[dict]:
    """Map GoEmotions rows to AutoIntent one-hot multilabel samples (dropping label-less rows)."""
    samples = []
    for ex in rows:
        label_ids = ex["labels"]
        if not label_ids:  # multilabel samples must have at least one active class
            continue
        one_hot = [0] * n_classes
        for idx in label_ids:
            one_hot[idx] = 1
        samples.append({"utterance": ex["text"], "label": one_hot})
    return samples


def prepare_mapping(
    repo: str, config: str, min_per_class: int | None, balance: str, seed: int
) -> dict[str, list[dict]]:
    """Build the full AutoIntent dataset mapping (intents + splits) from GoEmotions."""
    print(f"Loading {repo} ({config}) ...")
    ds, names = load_goemotions(repo, config)
    n_classes = len(names)
    print(f"{n_classes} emotion classes: {', '.join(names)}")

    mapping: dict[str, list[dict]] = {"intents": [{"id": i, "name": name} for i, name in enumerate(names)]}
    for hf_split, ai_split in SPLIT_MAP.items():
        if hf_split not in ds:
            continue
        rows = list(ds[hf_split])
        if ai_split == "train":
            rows = subsample_train(rows, n_classes, min_per_class, balance, seed)
        samples = to_onehot_samples(rows, n_classes)
        mapping[ai_split] = samples
        print(f"  {ai_split}: {len(samples)} samples (dropped {len(rows) - len(samples)} label-less)")
    return mapping


def save_mapping(mapping: dict, out_path: str | Path) -> None:
    """Write the dataset mapping to JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
