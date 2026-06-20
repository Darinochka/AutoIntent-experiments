"""Phase-4 figure: fine-tuned bert-base vs the frozen-e5 capability curve.

Plots the frozen-e5 classwise curve (from sweep_summary_test.csv) and overlays the fine-tuned bert-base
points (from logs/ft-cw<cap>-s1_metrics.json) on the same macro-F1 vs training-rows axes, with reference
lines for frozen-e5's full-data ceiling and the published BERT-base full-data baseline. The picture shows
FT starting BELOW frozen at small data, then crossing over and breaking the frozen ceiling by cap-300.

Usage:  uv run plot_ft.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_DIR = Path(__file__).resolve().parent
BERT_FULL = 0.46  # GoEmotions BERT-base, full-data fine-tuned (Demszky et al. 2020)
FROZEN_CEILING = 0.353  # frozen-e5 classic-light on full 43k (REPORT §9)


def _frozen_curve(summary_csv: Path) -> tuple[list[int], list[float]]:
    rows = [r for r in csv.DictReader(summary_csv.open(encoding="utf-8")) if r["balance"] == "classwise"]
    rows.sort(key=lambda r: int(r["train_size"]))
    return [int(r["train_size"]) for r in rows], [float(r["f1_mean"]) for r in rows]


def _ft_points(logs_dir: Path) -> tuple[list[int], list[float]]:
    pts = []
    for p in sorted(logs_dir.glob("ft-cw*-s1_metrics.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        pts.append((r["fed_split_sizes"]["train"], r["test_metrics"]["decision_f1"]))
    pts.sort()
    return [x for x, _ in pts], [y for _, y in pts]


def main() -> None:
    fx, fy = _frozen_curve(EXP_DIR / "logs" / "sweep_summary_test.csv")
    tx, ty = _ft_points(EXP_DIR / "logs")

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.set_xscale("log")
    ax.set_xlabel("training rows (log scale)")
    ax.set_ylabel("macro decision-F1 (test)")
    ax.set_ylim(0.10, 0.50)

    ax.plot(fx, fy, "o-", color="tab:blue", label="frozen e5 + linear (classic-light)")
    ax.plot(tx, ty, "D-", color="tab:purple", markersize=8, label="fine-tuned bert-base (transformers)")
    for x, y in zip(tx, ty):
        ax.annotate(f"{y:.3f}", xy=(x, y), xytext=(x, y + 0.012), fontsize=8, ha="center", color="tab:purple")

    ax.axhline(BERT_FULL, ls="--", color="tab:red", lw=1.2)
    ax.text(fx[0], BERT_FULL + 0.006, "fine-tuned BERT-base, full data = 0.46", color="tab:red", fontsize=9)
    ax.axhline(FROZEN_CEILING, ls=":", color="tab:blue", lw=1.2, alpha=0.7)
    ax.text(fx[0], FROZEN_CEILING - 0.018, "frozen-e5 ceiling (full data) = 0.353", color="tab:blue", fontsize=8)

    ax.set_title("Fine-tuning breaks the frozen ceiling — but only with enough data")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = EXP_DIR / "figures" / "phase4_ft_vs_frozen.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
