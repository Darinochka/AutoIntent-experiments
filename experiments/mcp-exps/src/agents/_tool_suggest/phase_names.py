"""Phase / collection naming for tool-suggest storage."""

import re


def sanitize_phase_name(name: str) -> str:
    """Make phase name safe for filenames and remote collection names."""
    return re.sub(r"[^\w\-]", "_", name) or "phase"
