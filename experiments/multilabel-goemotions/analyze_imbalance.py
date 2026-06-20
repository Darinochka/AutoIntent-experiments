"""Compute per-class imbalance for the prepared classwise datasets (for the Phase-3 report).

Each ``data/ge_classwise_<cap>_s<seed>_test.json`` stores train rows as one-hot ``label`` vectors,
so per-class support is just the column sum. We report, per (cap, seed): realized train rows, the
imbalance ratio (max/min over present classes), the coefficient of variation (std/mean), and how many
classes are "pinned" below the cap (could not reach it because they are rarer than the cap).

Usage:
    uv run analyze_imbalance.py                       # all data/ge_classwise_*_test.json
    uv run analyze_imbalance.py --seed 1              # only seed-1 datasets
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

EXP_DIR = Path(__file__).resolve().parent
app = App(help="Per-class imbalance of the prepared classwise datasets.")

_NAME_RE = re.compile(r"ge_classwise_(\d+)_s(\d+)_test\.json$")


def _class_counts(path: Path) -> list[int]:
    """Sum the one-hot train labels into per-class support counts."""
    train = json.loads(path.read_text(encoding="utf-8"))["train"]
    n_classes = len(train[0]["label"])
    counts = [0] * n_classes
    for row in train:
        for cls, on in enumerate(row["label"]):
            if on:
                counts[cls] += 1
    return counts


@app.default
def main(
    data_dir: Annotated[Path, Parameter(help="Directory with prepared datasets.")] = EXP_DIR / "data",
    seed: Annotated[int | None, Parameter(help="Only this seed (default: all).")] = None,
) -> None:
    """Print a per-(cap, seed) imbalance table sorted by cap."""
    rows = []
    for path in sorted(data_dir.glob("ge_classwise_*_test.json")):
        m = _NAME_RE.search(path.name)
        if not m:
            continue
        cap, s = int(m.group(1)), int(m.group(2))
        if seed is not None and s != seed:
            continue
        counts = _class_counts(path)
        present = [c for c in counts if c > 0]
        train_rows = sum(counts)  # label occurrences; rows differ slightly (multilabel) but track support
        n_rows = len(json.loads(path.read_text(encoding="utf-8"))["train"])
        cmax, cmin = max(present), min(present)
        cv = statistics.pstdev(present) / statistics.mean(present)
        pinned = sum(1 for c in present if c < cap)
        rows.append((cap, s, n_rows, cmin, cmax, cmax / cmin, cv, pinned, len(present)))

    rows.sort(key=lambda r: (r[0], r[1]))
    print(f"{'cap':>6} {'seed':>4} {'rows':>7} {'min':>5} {'max':>6} {'max/min':>8} {'CV':>6} {'pinned':>7} {'present':>7}")
    for cap, s, n_rows, cmin, cmax, ratio, cv, pinned, present in rows:
        print(f"{cap:>6} {s:>4} {n_rows:>7} {cmin:>5} {cmax:>6} {ratio:>8.1f} {cv:>6.2f} {pinned:>7} {present:>7}")


if __name__ == "__main__":
    app()
