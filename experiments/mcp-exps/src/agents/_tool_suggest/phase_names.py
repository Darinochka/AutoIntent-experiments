"""Phase / collection naming for tool-suggest storage."""

import re


def sanitize_phase_name(name: str) -> str:
    """Make phase name safe for filenames and remote collection names."""
    return re.sub(r"[^\w\-]", "_", name) or "phase"


def namespaced_collection_name(experiment_name: str, phase_name: str) -> str:
    """Remote collection id: experiment then phase so one server can host many benchmark runs."""
    exp = sanitize_phase_name(experiment_name)
    phase = sanitize_phase_name(phase_name)
    return f"{exp}__{phase}"
