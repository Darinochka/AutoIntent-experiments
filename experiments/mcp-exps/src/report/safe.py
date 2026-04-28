"""Utility for float deserializing."""


def safe_float(v: object) -> float:
    """Best-effort float conversion for metrics/scores coming from Logfire."""
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v))
    except ValueError:
        return 0.0
