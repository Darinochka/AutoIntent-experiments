"""Robustness scenarios: writes are not atomic, so an interrupted/concurrent write poisons the cache.

Neither cache writes via a temp file + atomic rename:
- embeddings: ``np.save(final_path, matrix)`` writes in place;
- structured outputs: ``PydanticModelDumper.dump`` ``mkdir``s then writes two JSON files in sequence.

If a writer is interrupted (crash, OOM, or a reader observing the file mid-write from another
process), the cache is left with a truncated/partial entry. The read paths do not guard against
this, so the *next* read raises and keeps raising until the bad entry is deleted by hand.
"""

from __future__ import annotations

from autointent import Embedder
from autointent._hash import Hasher
from autointent.configs._embedder import SentenceTransformerEmbeddingConfig
from autointent.generation._cache import StructuredOutputCache

from cache_bench.harness import (
    DEFAULT_MODEL,
    embeddings_dir,
    gen_utterances,
    temp_cache,
    write_results,
)
from cache_bench.models import IntentPrediction, build_messages, make_prediction

GEN_PARAMS: dict[str, object] = {"temperature": 0.0}


def _embedding_poisoning() -> dict[str, object]:
    utts = gen_utterances(300, seed=42)
    with temp_cache() as base:
        embedder = Embedder(SentenceTransformerEmbeddingConfig(model_name=DEFAULT_MODEL))
        embedder.embed(utts)  # valid cold write

        hasher = Hasher()
        hasher.update(embedder._backend.get_hash())
        hasher.update(utts)
        path = embeddings_dir(base) / f"{hasher.hexdigest()}.npy"

        raw = path.read_bytes()
        path.write_bytes(raw[: len(raw) // 2])  # simulate an interrupted / concurrent partial write

        error_type = "none"
        recovered = True
        try:
            embedder.embed(utts)  # read path: np.load on a truncated file
        except Exception as exc:  # noqa: BLE001 - we want to record whatever it raises
            error_type = type(exc).__name__
            recovered = False

    return {
        "cache": "embeddings",
        "scenario": "truncated .npy then re-read",
        "raised_on_next_read": not recovered,
        "error_type": error_type,
        "auto_recovers_by_recompute": recovered,
    }


def _structured_poisoning() -> dict[str, object]:
    messages = build_messages("classify me")
    with temp_cache() as base:
        writer = StructuredOutputCache(use_cache=True)
        writer.set(messages, IntentPrediction, GEN_PARAMS, make_prediction(0))

        key = writer._get_cache_key(messages, IntentPrediction, GEN_PARAMS)
        entry_dir = base / "structured_outputs" / key
        (entry_dir / "model_dump.json").unlink()  # simulate crash between the two file writes

        reader = StructuredOutputCache(use_cache=True)  # fresh: empty memory cache
        error_type = "none"
        recovered = True
        try:
            reader.get(messages, IntentPrediction, GEN_PARAMS)  # dir exists, model_dump.json missing
        except Exception as exc:  # noqa: BLE001 - record whatever it raises
            error_type = type(exc).__name__
            recovered = False

    return {
        "cache": "structured_outputs",
        "scenario": "entry dir missing model_dump.json then re-read",
        "raised_on_next_read": not recovered,
        "error_type": error_type,
        "auto_recovers_by_recompute": recovered,
    }


def run() -> None:
    """Run robustness/atomicity demonstrations for both caches."""
    rows = [_embedding_poisoning(), _structured_poisoning()]
    frame = write_results("concurrency", rows)
    print("\n[concurrency] non-atomic write poisoning (interrupted/concurrent writes)")
    print(frame.to_string(index=False))
