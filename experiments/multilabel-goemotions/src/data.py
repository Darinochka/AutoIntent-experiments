"""GoEmotions loading, multilabel one-hot conversion, and train-subsampling helpers.

GoEmotions (``google-research-datasets/go_emotions``, ``simplified`` config) ships each example as
``text`` plus ``labels`` (a list of emotion class indices). AutoIntent expects multilabel samples as
one-hot vectors with an explicit ``intents`` list so the label space is sized correctly.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from loguru import logger

DEFAULT_REPO = "google-research-datasets/go_emotions"
DEFAULT_CONFIG = "simplified"

# Feeding scheme: we give AutoIntent only train + test (no validation split), so AutoIntent carves its
# own HPO-validation out of the (subsampled) train via DataConfig.validation_size. This makes the train
# balance treatment shape the validation set too -- an imbalanced train yields an imbalanced HPO val, so
# model selection is affected by balance, not just model fitting. GoEmotions' validation becomes the held-out
# eval set (AutoIntent's test); GoEmotions' own test split is intentionally unused.
# GoEmotions split name -> AutoIntent split name.
SPLIT_MAP = {"train": "train", "validation": "test"}


def load_goemotions(repo: str = DEFAULT_REPO, config: str = DEFAULT_CONFIG) -> tuple[Any, list[str]]:
    """Load the dataset and return it alongside the ordered emotion class names."""
    ds = load_dataset(repo, config)
    names = ds["train"].features["labels"].feature.names
    return ds, names


def label_matrix(rows: list[dict[str, Any]], n_classes: int) -> np.ndarray:
    """Build a (n_samples, n_classes) one-hot matrix from GoEmotions label-index lists."""
    y = np.zeros((len(rows), n_classes), dtype=int)
    for i, ex in enumerate(rows):
        for idx in ex["labels"]:
            y[i, idx] = 1
    return y


def stratified_subsample(
    rows: list[dict[str, Any]], n_classes: int, min_per_class: int, seed: int
) -> list[dict[str, Any]]:
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
        logger.warning(
            "Cannot reduce train below {} rows for min-samples-per-class={}; keeping all.", len(rows), min_per_class
        )
        return rows

    # test_size is passed explicitly: iterstrat 0.1.9's "default" sentinel is rejected by modern sklearn.
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, train_size=target, test_size=len(rows) - target, random_state=seed
    )
    train_idx, _ = next(splitter.split(np.zeros((len(rows), 1)), y))
    subset = [rows[i] for i in train_idx]

    rarest = int(label_matrix(subset, n_classes).sum(axis=0)[present].min())
    logger.info(
        "Stratified-subsampled train to {} rows (rarest class has {}, floor {}).", len(subset), rarest, min_per_class
    )
    if rarest < min_per_class:
        logger.warning(
            "Rarest class below floor ({} < {}); data too imbalanced to guarantee it.", rarest, min_per_class
        )
    return subset


def classwise_subsample(
    rows: list[dict[str, Any]], n_classes: int, target_per_class: int, seed: int
) -> list[dict[str, Any]]:
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
    logger.info(
        "Classwise-subsampled train to {} rows (per-class min/median/max = {}/{}/{}, target {}).",
        len(subset),
        int(sub_counts.min()),
        int(np.median(sub_counts)),
        int(sub_counts.max()),
        target_per_class,
    )
    return subset


def natural_subsample(rows: list[dict[str, Any]], n_classes: int, total_size: int, seed: int) -> list[dict[str, Any]]:
    """Proportion-preserving sample of ``total_size`` rows (keeps the natural, imbalanced distribution).

    Used to build a size-matched imbalanced counterpart to a balanced N-shot set: same total size, but
    drawn at natural label proportions instead of capped per class.
    """
    if total_size >= len(rows):
        return rows
    y = label_matrix(rows, n_classes)
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, train_size=total_size, test_size=len(rows) - total_size, random_state=seed
    )
    idx, _ = next(splitter.split(np.zeros((len(rows), 1)), y))
    subset = [rows[i] for i in idx]

    present = y.sum(axis=0) > 0
    sub = label_matrix(subset, n_classes).sum(axis=0)[present]
    logger.info(
        "Natural-subsampled train to {} rows (per-class min/median/max = {}/{}/{}).",
        len(subset),
        int(sub.min()),
        int(np.median(sub)),
        int(sub.max()),
    )
    return subset


def subsample_train(
    rows: list[dict[str, Any]], n_classes: int, min_per_class: int | None, balance: str, seed: int
) -> list[dict[str, Any]]:
    """Dispatch to the requested subsampler, or keep all rows when min_per_class is None."""
    if min_per_class is None:
        return rows
    if balance == "classwise":
        return classwise_subsample(rows, n_classes, min_per_class, seed)
    return stratified_subsample(rows, n_classes, min_per_class, seed)


def to_onehot_samples(rows: list[dict[str, Any]], n_classes: int) -> list[dict[str, Any]]:
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
) -> dict[str, list[dict[str, Any]]]:
    """Build the full AutoIntent dataset mapping (intents + splits) from GoEmotions."""
    logger.info("Loading {} ({}) ...", repo, config)
    ds, names = load_goemotions(repo, config)
    n_classes = len(names)
    logger.info("{} emotion classes: {}", n_classes, ", ".join(names))

    mapping: dict[str, list[dict[str, Any]]] = {"intents": [{"id": i, "name": name} for i, name in enumerate(names)]}
    for src_split, ai_split in SPLIT_MAP.items():
        if src_split not in ds:
            continue
        rows = list(ds[src_split])
        if ai_split == "train":
            rows = subsample_train(rows, n_classes, min_per_class, balance, seed)
        samples = to_onehot_samples(rows, n_classes)
        mapping[ai_split] = samples
        dropped = len(rows) - len(samples)
        logger.info(
            "ai-{} (from source {}): {} samples (dropped {} label-less)", ai_split, src_split, len(samples), dropped
        )
    return mapping


def assemble_mapping(
    names: list[str], train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Build an AutoIntent mapping from explicit train/eval rows.

    Mirrors the prepare_mapping feeding scheme: eval_rows become AutoIntent's ``test`` split, and
    AutoIntent carves its HPO-validation out of ``train``. Used by the sweep to vary the train subsample
    while keeping the held-out eval set fixed.
    """
    n_classes = len(names)
    return {
        "intents": [{"id": i, "name": name} for i, name in enumerate(names)],
        "train": to_onehot_samples(train_rows, n_classes),
        "test": to_onehot_samples(eval_rows, n_classes),
    }


def save_mapping(mapping: dict[str, Any], out_path: str | Path) -> None:
    """Write the dataset mapping to JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
