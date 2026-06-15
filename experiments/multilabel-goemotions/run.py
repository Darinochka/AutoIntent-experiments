"""CLI: optimize an AutoIntent pipeline on GoEmotions and dump metrics. Logic lives in src/pipeline.py.

Usage:
    uv run run.py                                   # classic-light on data/go_emotions.json
    uv run run.py --preset classic-medium --exp-name gm-medium
    uv run run.py --scoring-metric scoring_map --decision-metric decision_f1  # override target metrics
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import App, Parameter

from src.naming import ensure_absent, metrics_path
from src.pipeline import run_experiment

EXP_DIR = Path(__file__).resolve().parent
app = App(help="Optimize an AutoIntent pipeline on GoEmotions multilabel.")


@dataclass(frozen=True)
class RunConfig:
    """Configuration for a single optimization run."""

    data: Annotated[Path, Parameter(help="Dataset JSON.")] = EXP_DIR / "data" / "go_emotions.json"
    preset: Annotated[str, Parameter(help="AutoIntent search-space preset.")] = "classic-light"
    logs_dir: Annotated[Path, Parameter(help="Directory for run logs/metrics.")] = EXP_DIR / "logs"
    exp_name: Annotated[str | None, Parameter(help="Experiment name (default: goemotions-<preset>).")] = None
    embedder_model: Annotated[str | None, Parameter(help="Override the preset's embedder.")] = None
    device: Annotated[Literal["cpu", "cuda", "mps"] | None, Parameter(help="Torch device for the embedder.")] = None
    scoring_metric: Annotated[str | None, Parameter(help="Override the scoring node's target_metric.")] = None
    decision_metric: Annotated[str | None, Parameter(help="Override the decision node's target_metric.")] = None
    seed: Annotated[int, Parameter(help="Random seed.")] = 42
    overwrite: Annotated[bool, Parameter(help="Replace existing metrics/run outputs.")] = False


_DEFAULTS = RunConfig()


@app.default
def main(cfg: Annotated[RunConfig, Parameter(name="*")] = _DEFAULTS) -> None:
    """Run one optimization described by the flattened RunConfig options."""
    exp_name = cfg.exp_name or f"goemotions-{cfg.preset}"

    ensure_absent(metrics_path(cfg.logs_dir, exp_name), cfg.overwrite, label="Metrics file")
    ensure_absent(Path(cfg.logs_dir) / exp_name, cfg.overwrite, label="Run directory")

    if not cfg.data.exists():
        msg = f"Dataset not found at {cfg.data}. Run prepare_data.py first."
        raise SystemExit(msg)

    run_experiment(
        data_path=cfg.data,
        preset=cfg.preset,
        exp_name=exp_name,
        logs_dir=cfg.logs_dir,
        embedder_model=cfg.embedder_model,
        device=cfg.device,
        scoring_metric=cfg.scoring_metric,
        decision_metric=cfg.decision_metric,
        seed=cfg.seed,
    )


if __name__ == "__main__":
    app()
