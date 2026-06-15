"""CLI: sweep scoring/decision target metrics across dataset sizes, balance modes, and seeds.

Logic and config live in src/sweep.py (SweepConfig + run_sweep), so the same sweep can be driven from
Python: ``run_sweep(SweepConfig(sizes=[10], balances=["classwise"], seeds=[1, 2, 3], device="mps"))``.

Usage:
    uv run sweep.py --dry-run                                  # print the run plan only
    uv run sweep.py --device mps --seeds 1 2 3                 # full grid, 3 seeds
    uv run sweep.py --device mps --sizes 10 100 \\
        --balances classwise stratified \\
        --scoring-metrics scoring_f1 --decision-metrics decision_f1 --seeds 1 2 3   # capability study

Resumable: re-running skips cells whose <exp>_metrics.json already exists (use --overwrite to force).
"""

from typing import Annotated

from cyclopts import App, Parameter

from src.sweep import SweepConfig, run_sweep

app = App(help="Sweep AutoIntent scoring/decision target metrics on GoEmotions multilabel.")


@app.default
def main(cfg: Annotated[SweepConfig, Parameter(name="*")] = SweepConfig()) -> None:
    """Run the sweep described by the flattened SweepConfig options."""
    run_sweep(cfg)


if __name__ == "__main__":
    app()
