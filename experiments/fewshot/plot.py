"""Plot few-shot accuracy curves: one subplot per dataset, frameworks overlaid.

Reads  ./data/comparison_few_shot.csv
Writes ./figures/few_shot_accuracy.png and .svg

Only accuracy is plotted. The "full" point is excluded on purpose: this is a
few-shot study, so behaviour on the full training set is out of scope.

Run:  uv run --no-project --with pandas --with matplotlib --with seaborn python plot.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGS = HERE / "figures"

HUE_ORDER = ["AutoIntent", "AutoGluon", "H2O"]
XTICKS = [4, 8, 16, 32, 64, 128]


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA / "comparison_few_shot.csv")

    # few-shot study: drop the full-training-set point entirely
    fs = df["few_shot"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    df = df[fs != "full"].copy()
    df["x"] = pd.to_numeric(fs[fs != "full"], errors="coerce")
    df = df.dropna(subset=["x"])
    df["x"] = df["x"].astype(int)

    datasets = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(15, 4.5), sharey=True)
    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds]
        sns.lineplot(
            data=sub, x="x", y="accuracy", hue="framework",
            hue_order=HUE_ORDER, marker="o", ax=ax, legend=(ax is axes[0]),
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks(XTICKS)
        ax.set_xticklabels(XTICKS)
        ax.set_title(ds.replace("DeepPavlov/", ""), fontsize=13)
        ax.set_xlabel("Few-shot examples")
        ax.set_ylabel("")
        ax.grid(True, color="lightgray", linewidth=0.5, alpha=0.7)

    axes[0].set_ylabel("Accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    if axes[0].legend_ is not None:
        axes[0].legend_.remove()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    for ext in ("png", "svg"):
        out = FIGS / f"few_shot_accuracy.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
