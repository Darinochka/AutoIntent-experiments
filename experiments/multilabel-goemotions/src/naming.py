"""Output-path resolution and collision checks shared by the CLI scripts."""

from __future__ import annotations

from pathlib import Path


def ensure_absent(path: str | Path, overwrite: bool, label: str = "Output") -> Path:
    """Fail before doing any work if path already exists (unless overwrite is set)."""
    path = Path(path)
    if path.exists() and not overwrite:
        msg = (
            f"{label} already exists: {path}\n"
            "Choose a different name (--exp-name / --out) or pass --overwrite to replace it."
        )
        raise SystemExit(msg)
    return path


def metrics_path(logs_dir: str | Path, exp_name: str) -> Path:
    """Path of the metrics JSON for a given experiment."""
    return Path(logs_dir) / f"{exp_name}_metrics.json"
