"""Run the AutoIntent cache benchmarks and write results to ``results/``.

Usage:
    uv run python run_all.py                 # run everything
    uv run python run_all.py embeddings      # only the embeddings-cache scenarios
    uv run python run_all.py structured      # only the structured-output scenarios
    uv run python run_all.py concurrency     # only the robustness scenarios
"""

from __future__ import annotations

import argparse
import logging

from cache_bench.scenarios import concurrency, embeddings, structured

GROUPS = {
    "embeddings": embeddings.run,
    "structured": structured.run,
    "concurrency": concurrency.run,
}


def main() -> None:
    """Parse args and run the selected benchmark group(s)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", nargs="?", default="all", choices=["all", *GROUPS])
    args = parser.parse_args()

    # AutoIntent's caches log at DEBUG; keep the benchmark output readable.
    logging.getLogger("autointent").setLevel(logging.ERROR)

    selected = GROUPS if args.group == "all" else {args.group: GROUPS[args.group]}
    for name, fn in selected.items():
        print(f"\n{'=' * 70}\n# {name}\n{'=' * 70}")
        fn()

    print("\nDone. Raw results in results/*.json and results/*.csv")


if __name__ == "__main__":
    main()
