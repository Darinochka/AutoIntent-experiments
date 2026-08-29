"""Shared infrastructure for the cache benchmarks.

Provides:
- an isolated temp cache dir (so the user's real ``~/Library/Caches/autointent`` is untouched),
- deterministic synthetic utterance/message generation,
- small timing helpers,
- result writers (JSON + CSV).
"""

from __future__ import annotations

import json
import random
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Any

import autointent._wrappers.embedder.utils as _emb_utils
import autointent.generation._cache as _gen_cache
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

# A small, fast, CPU-friendly model. It is also AutoIntent's default embedder.
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# The two modules that resolve the cache root via ``user_cache_dir`` at call time.
_PATCH_TARGETS = (_emb_utils, _gen_cache)


@contextmanager
def temp_cache() -> Iterator[Path]:
    """Redirect both AutoIntent caches to a throwaway dir for the duration of the block.

    AutoIntent resolves its cache root via ``appdirs.user_cache_dir("autointent")`` at call
    time, in two modules. We monkeypatch that symbol in both so benchmarks never touch (or
    benefit from) the user's real cache, and so each scenario starts from an empty cache.

    Yields:
        Path to the temporary cache root (contains ``embeddings/`` and ``structured_outputs/``).
    """
    base = Path(tempfile.mkdtemp(prefix="ai_cache_bench_"))
    originals = [mod.user_cache_dir for mod in _PATCH_TARGETS]

    def fake_cache_dir(*_args: object, **_kwargs: object) -> str:
        return str(base)

    try:
        for mod in _PATCH_TARGETS:
            mod.user_cache_dir = fake_cache_dir  # type: ignore[attr-defined]
        yield base
    finally:
        for mod, original in zip(_PATCH_TARGETS, originals, strict=True):
            mod.user_cache_dir = original  # type: ignore[attr-defined]
        shutil.rmtree(base, ignore_errors=True)


def embeddings_dir(base: Path) -> Path:
    """Path to the embeddings cache under a cache root."""
    return base / "embeddings"


def structured_dir(base: Path) -> Path:
    """Path to the structured-output cache under a cache root."""
    return base / "structured_outputs"


_WORDS = [
    "account",
    "login",
    "password",
    "reset",
    "transfer",
    "balance",
    "card",
    "payment",
    "refund",
    "order",
    "delivery",
    "cancel",
    "subscription",
    "upgrade",
    "downgrade",
    "invoice",
    "billing",
    "support",
    "agent",
    "human",
    "schedule",
    "appointment",
    "reminder",
    "weather",
    "flight",
    "hotel",
    "booking",
    "restaurant",
    "reservation",
    "menu",
    "price",
    "status",
    "update",
    "tracking",
    "package",
    "return",
    "exchange",
    "warranty",
    "repair",
    "install",
    "setup",
    "configure",
]


def gen_utterances(n: int, seed: int = 0) -> list[str]:
    """Generate ``n`` deterministic, varied-length synthetic utterances.

    Args:
        n: Number of utterances.
        seed: RNG seed for reproducibility.

    Returns:
        A list of ``n`` unique-ish strings.
    """
    rng = random.Random(seed)  # noqa: S311 - synthetic benchmark data, not security-sensitive
    out: list[str] = []
    for i in range(n):
        length = rng.randint(4, 16)
        words = rng.choices(_WORDS, k=length)
        out.append(f"{i:06d} " + " ".join(words))
    return out


def timeit(fn: Callable[[], object], repeats: int = 5) -> float:
    """Return the median wall-clock time (seconds) over ``repeats`` runs of ``fn``."""
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return median(samples)


def time_once(fn: Callable[[], object]) -> float:
    """Return the wall-clock time (seconds) for a single run of ``fn``."""
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def dir_size_bytes(path: Path) -> int:
    """Total size in bytes of all files under ``path`` (recursive)."""
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def count_entries(path: Path, *, files: bool, dirs: bool) -> int:
    """Count direct children of ``path`` that are files and/or directories."""
    if not path.exists():
        return 0
    total = 0
    for child in path.iterdir():
        if files and child.is_file():
            total += 1
        if dirs and child.is_dir():
            total += 1
    return total


def count_inodes(path: Path) -> int:
    """Count every filesystem entry (files + dirs) under ``path``, recursively."""
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob("*"))


def write_results(name: str, rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Write ``rows`` to ``results/<name>.json`` and ``results/<name>.csv``.

    Args:
        name: Base filename (no extension).
        rows: Records to persist.

    Returns:
        The DataFrame that was written (for convenient printing by callers).
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS_DIR / f"{name}.csv", index=False)
    return frame
