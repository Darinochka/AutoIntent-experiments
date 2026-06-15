"""CLI: build the GoEmotions multilabel JSON for AutoIntent. Logic lives in src/data.py.

Usage:
    uv run prepare_data.py                                       # full dataset -> data/go_emotions.json
    uv run prepare_data.py --min-samples-per-class 50            # stratified subsample (floor 50/class)
    uv run prepare_data.py --min-samples-per-class 50 --balance classwise  # flattened (cap ~50/class)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import App, Parameter
from loguru import logger

from src.data import DEFAULT_CONFIG, DEFAULT_REPO, prepare_mapping, save_mapping
from src.naming import ensure_absent

EXP_DIR = Path(__file__).resolve().parent
app = App(help="Build the GoEmotions multilabel dataset JSON for AutoIntent.")


@dataclass(frozen=True)
class PrepareConfig:
    """Configuration for building the dataset JSON."""

    repo: Annotated[str, Parameter(help="HuggingFace dataset repo.")] = DEFAULT_REPO
    config: Annotated[str, Parameter(help="Dataset config name.")] = DEFAULT_CONFIG
    out: Annotated[Path, Parameter(help="Output JSON path.")] = EXP_DIR / "data" / "go_emotions.json"
    min_samples_per_class: Annotated[
        int | None,
        Parameter(help="Train subsample control (floor for stratified, per-class cap for classwise)."),
    ] = None
    balance: Annotated[
        Literal["stratified", "classwise"],
        Parameter(help="stratified=proportion-preserving; classwise=flatten the distribution."),
    ] = "stratified"
    seed: Annotated[int, Parameter(help="Random seed for subsampling.")] = 42
    overwrite: Annotated[bool, Parameter(help="Replace the output file if it already exists.")] = False


_DEFAULTS = PrepareConfig()


@app.default
def main(cfg: Annotated[PrepareConfig, Parameter(name="*")] = _DEFAULTS) -> None:
    """Build and save the dataset described by the flattened PrepareConfig options."""
    out = ensure_absent(cfg.out, cfg.overwrite, label="Dataset file")
    mapping = prepare_mapping(cfg.repo, cfg.config, cfg.min_samples_per_class, cfg.balance, cfg.seed)
    save_mapping(mapping, out)
    logger.info("Wrote {}", out)


if __name__ == "__main__":
    app()
