"""Render the Phase-3 capability curve: macro-F1 and class imbalance vs training-set size.

Reads the aggregated test sweep (``logs/sweep_summary_test.csv``) for the classwise curve and the saved
seed-1 datasets (``data/ge_classwise_<cap>_s1_test.json``) for the per-cap imbalance ratio, then draws a
twin-axis figure: macro-F1 (left, with ±std band) and max/min class imbalance (right, log) against
training rows (log x). The picture shows macro-F1 plateauing (~0.35) exactly as imbalance explodes —
i.e. the gap to fine-tuned BERT (0.46) is not a data-volume problem.

Usage:
    uv run plot_curve.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Annotated

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cyclopts import App, Parameter

EXP_DIR = Path(__file__).resolve().parent
app = App(help="Plot the classwise capability curve vs imbalance.")
BERT_MACRO_F1 = 0.46  # GoEmotions BERT-base baseline (Demszky et al. 2020), full-data fine-tuned


def _imbalance_ratio(path: Path) -> float | None:
    """max/min per-class support over present classes, from a saved one-hot dataset (None if absent)."""
    if not path.exists():
        return None
    train = json.loads(path.read_text(encoding="utf-8"))["train"]
    n_classes = len(train[0]["label"])
    counts = [0] * n_classes
    for row in train:
        for cls, on in enumerate(row["label"]):
            if on:
                counts[cls] += 1
    present = [c for c in counts if c > 0]
    return max(present) / min(present)


@app.default
def main(
    summary_csv: Annotated[Path, Parameter(help="Aggregated test sweep.")] = EXP_DIR / "logs" / "sweep_summary_test.csv",
    data_dir: Annotated[Path, Parameter(help="Prepared datasets.")] = EXP_DIR / "data",
    out: Annotated[Path, Parameter(help="Output PNG.")] = EXP_DIR / "figures" / "phase3_capability_curve.png",
) -> None:
    """Draw and save the twin-axis capability/imbalance figure."""
    rows = [
        r for r in csv.DictReader(summary_csv.open(encoding="utf-8")) if r["balance"] == "classwise"
    ]
    rows.sort(key=lambda r: int(r["train_size"]))
    caps = [int(r["size"]) for r in rows]
    train = [int(r["train_size"]) for r in rows]
    f1 = [float(r["f1_mean"]) for r in rows]
    f1_std = [float(r["f1_std"]) for r in rows]
    imb = [_imbalance_ratio(data_dir / f"ge_classwise_{c}_s1_test.json") for c in caps]

    fig, ax1 = plt.subplots(figsize=(8.5, 5.2))
    ax1.set_xscale("log")
    ax1.set_xlabel("training rows (log scale)")
    ax1.set_ylabel("macro decision-F1 (test)", color="tab:blue")
    lo = [v - s for v, s in zip(f1, f1_std)]
    hi = [v + s for v, s in zip(f1, f1_std)]
    ax1.fill_between(train, lo, hi, color="tab:blue", alpha=0.15)
    ax1.plot(train, f1, "o-", color="tab:blue", label="macro-F1 (classwise)")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim(0.10, 0.50)
    ax1.axhline(BERT_MACRO_F1, ls="--", color="tab:red", lw=1.2)
    ax1.text(train[0], BERT_MACRO_F1 + 0.006, "fine-tuned BERT (full data) = 0.46", color="tab:red", fontsize=9)

    # annotate the full-data plateau point
    ax1.annotate(
        f"full data\n{f1[-1]:.3f}",
        xy=(train[-1], f1[-1]),
        xytext=(train[-1] * 0.30, f1[-1] - 0.075),
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "tab:blue"},
    )

    ax2 = ax1.twinx()
    ax2.set_yscale("log")
    ax2.set_ylabel("class imbalance  max/min  (log)", color="tab:green")
    ax2.plot(train, imb, "s--", color="tab:green", alpha=0.8, label="imbalance max/min")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    ax2.axvline(1918, ls=":", color="gray", lw=1)
    ax2.text(1918, ax2.get_ylim()[1] * 0.55, " balance saturates\n (grief pinned @77)", fontsize=8, color="gray")

    ax1.set_title("AutoIntent classic-light (frozen e5): macro-F1 plateaus as imbalance explodes")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    app()
