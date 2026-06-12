"""Convert the GoEmotions dataset into AutoIntent's multilabel JSON format.

GoEmotions (``google-research-datasets/go_emotions``, ``simplified`` config) ships each
example as ``text`` plus ``labels`` (a list of emotion class indices). AutoIntent expects
multilabel samples as one-hot vectors: ``{"utterance": ..., "label": [0, 1, 0, ...]}`` with an
explicit ``intents`` list so the label space is sized correctly.

Usage:
    uv run prepare_data.py                 # full dataset -> data/go_emotions.json
    uv run prepare_data.py --max-train 2000  # subsample train for a fast pass
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

SCRIPT_DIR = Path(__file__).resolve().parent

# GoEmotions split name -> AutoIntent split name.
SPLIT_MAP = {"train": "train", "validation": "validation", "test": "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="google-research-datasets/go_emotions", help="HuggingFace dataset repo.")
    parser.add_argument("--config", default="simplified", help="Dataset config name.")
    parser.add_argument("--out", type=Path, default=SCRIPT_DIR / "data" / "go_emotions.json", help="Output JSON path.")
    parser.add_argument("--max-train", type=int, default=None, help="Subsample the train split to this many samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling.")
    return parser.parse_args()


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
        if ai_split == "train" and args.max_train is not None and args.max_train < len(rows):
            rows = random.Random(args.seed).sample(rows, args.max_train)
            print(f"Subsampled train to {args.max_train} rows.")
        samples = to_autointent_split(rows, n_classes)
        mapping[ai_split] = samples
        print(f"  {ai_split}: {len(samples)} samples (dropped {len(rows) - len(samples)} label-less)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
