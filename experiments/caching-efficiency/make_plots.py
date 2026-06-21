"""Render a few figures from the benchmark CSVs into ``results/*.png`` for the report."""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from cache_bench.harness import RESULTS_DIR

_IDEAL_REUSE = 1.01  # redundancy at/under this counts as "no waste"


def plot_embed_perf() -> None:
    """Cold vs warm embedding time, and the per-call key-build overhead, vs list size."""
    frame = pd.read_csv(RESULTS_DIR / "embed_perf.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(frame["n_utterances"], frame["cold_s"], "o-", label="cold (compute + save)")
    ax.plot(frame["n_utterances"], frame["warm_s"], "s-", label="warm (hash + np.load)")
    ax.plot(frame["n_utterances"], frame["key_build_s"], "^--", label="key build (hash whole list)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of utterances in the list")
    ax.set_ylabel("seconds (log)")
    ax.set_title("Embedding cache: cold vs warm vs key-build overhead")
    ax.legend()
    ax.grid(visible=True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "embed_perf.png", dpi=130)
    plt.close(fig)


def plot_embed_reuse() -> None:
    """Redundant recomputation factor under realistic access patterns."""
    frame = pd.read_csv(RESULTS_DIR / "embed_reuse.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["tab:green" if v <= _IDEAL_REUSE else "tab:red" for v in frame["redundancy_factor"]]
    ax.bar(frame["pattern"], frame["redundancy_factor"], color=colors)
    ax.axhline(1.0, color="black", linestyle=":", label="ideal (per-utterance cache)")
    ax.set_ylabel("redundancy factor (computed / ideal)")
    ax.set_title("Embedding cache reuse by access pattern (lower is better)")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "embed_reuse.png", dpi=130)
    plt.close(fig)


def plot_struct_preload() -> None:
    """Entries on disk vs entries actually preloaded into RAM by __init__."""
    frame = pd.read_csv(RESULTS_DIR / "struct_preload.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(frame["entries_written"], frame["on_disk_dirs"], "o-", label="entries on disk")
    ax.plot(frame["entries_written"], frame["preloaded_into_memory"], "s-", label="preloaded into RAM")
    ax.set_xlabel("entries written")
    ax.set_ylabel("count")
    ax.set_title("StructuredOutputCache.__init__ preloads nothing (is_file vs directory)")
    ax.legend()
    ax.grid(visible=True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "struct_preload.png", dpi=130)
    plt.close(fig)


def main() -> None:
    """Render all figures."""
    plot_embed_perf()
    plot_embed_reuse()
    plot_struct_preload()
    print(f"Wrote figures to {RESULTS_DIR}/*.png")


if __name__ == "__main__":
    main()
