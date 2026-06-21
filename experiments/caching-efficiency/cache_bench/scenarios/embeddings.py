"""Embeddings-cache scenarios: keying correctness, perf decomposition, reuse, storage."""

from __future__ import annotations

import tempfile
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from autointent import Embedder
from autointent._hash import Hasher
from autointent._wrappers.embedder.sentence_transformers import (
    SentenceTransformerEmbeddingBackend,
    _get_latest_commit_hash,
)
from autointent.configs._embedder import SentenceTransformerEmbeddingConfig

from cache_bench.harness import (
    DEFAULT_MODEL,
    count_entries,
    embeddings_dir,
    gen_utterances,
    temp_cache,
    time_once,
    timeit,
    write_results,
)

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

PERF_SIZES = (100, 500, 1_000, 5_000, 10_000)


def _make_embedder(*, use_cache: bool = True, model_name: str = DEFAULT_MODEL) -> Embedder:
    config = SentenceTransformerEmbeddingConfig(model_name=model_name, use_cache=use_cache)
    return Embedder(config)


def _n_cached(base: Path) -> int:
    return count_entries(embeddings_dir(base), files=True, dirs=False)


def run_keying() -> None:
    """Verify the embedding cache hits only on byte-identical (model, list, prompt) keys."""
    base_list = gen_utterances(200, seed=1)
    rows: list[dict[str, object]] = []

    with temp_cache() as base:
        embedder = _make_embedder()
        first = embedder.embed(base_list)  # cold: writes 1 file
        baseline_files = _n_cached(base)

        edited = [*base_list]
        edited[0] = edited[0] + "?"  # one character changed

        variants: list[tuple[str, list[str], bool]] = [
            ("identical", base_list, True),
            ("reordered", list(reversed(base_list)), False),
            ("subset_drop_last", base_list[:-1], False),
            ("superset_add_one", [*base_list, "a brand new utterance"], False),
            ("one_char_edit", edited, False),
        ]

        for name, utts, expect_hit in variants:
            before = _n_cached(base)
            result = embedder.embed(utts)
            after = _n_cached(base)
            observed_hit = after == before
            arrays_equal = bool(np.array_equal(result, first)) if name == "identical" else None
            rows.append(
                {
                    "variant": name,
                    "expected_hit": expect_hit,
                    "observed_hit": observed_hit,
                    "correct": observed_hit == expect_hit,
                    "files_before": before,
                    "files_after": after,
                    "arrays_equal_to_first": arrays_equal,
                }
            )

        # Prompt sensitivity: same list, but a configured prompt changes the key.
        prompt_embedder = Embedder(
            SentenceTransformerEmbeddingConfig(model_name=DEFAULT_MODEL, default_prompt="query: ")
        )
        before = _n_cached(base)
        prompt_embedder.embed(base_list)
        after = _n_cached(base)
        rows.append(
            {
                "variant": "prompt_added",
                "expected_hit": False,
                "observed_hit": after == before,
                "correct": (after == before) is False,
                "files_before": before,
                "files_after": after,
                "arrays_equal_to_first": None,
            }
        )
        _ = baseline_files

    frame = write_results("embed_keying", rows)
    print("\n[embed_keying] hit/miss correctness")
    print(frame.to_string(index=False))


def run_perf() -> None:
    """Cold vs warm timing per list size, decomposed into key-build / get_hash / np.load."""
    rows: list[dict[str, object]] = []

    with temp_cache() as base:
        embedder = _make_embedder()
        embedder.embed(["warmup utterance to load the model"])  # load model outside timings
        backend = embedder._backend
        ghash = backend.get_hash()  # warms the lru_cache for the commit hash

        for n in PERF_SIZES:
            utts = gen_utterances(n, seed=n)

            cold_s = time_once(partial(embedder.embed, utts))  # compute + hash + save
            warm_s = timeit(partial(embedder.embed, utts), repeats=3)  # hash + np.load

            def build_key(u: list[str] = utts) -> str:
                hasher = Hasher()
                hasher.update(ghash)
                hasher.update(u)
                return hasher.hexdigest()

            key_build_s = timeit(build_key, repeats=5)
            path = embeddings_dir(base) / f"{build_key()}.npy"
            np_load_s = timeit(partial(np.load, path), repeats=5)
            get_hash_cached_s = timeit(backend.get_hash, repeats=5)

            file_mb = path.stat().st_size / 1e6
            rows.append(
                {
                    "n_utterances": n,
                    "cold_s": round(cold_s, 5),
                    "warm_s": round(warm_s, 6),
                    "speedup_cold_over_warm": round(cold_s / warm_s, 1) if warm_s else None,
                    "key_build_s": round(key_build_s, 6),
                    "get_hash_cached_s": round(get_hash_cached_s, 6),
                    "np_load_s": round(np_load_s, 6),
                    "file_mb": round(file_mb, 3),
                    "bytes_per_utterance": round(path.stat().st_size / n, 1),
                }
            )

    frame = write_results("embed_perf", rows)
    print("\n[embed_perf] cold vs warm, with overhead decomposition")
    print(frame.to_string(index=False))


def run_get_hash() -> None:
    """Cost of get_hash() for HF-name (first vs lru-cached) and for a local-path model."""
    rows: list[dict[str, object]] = []

    with temp_cache():
        embedder = _make_embedder()
        backend_hf = cast("SentenceTransformerEmbeddingBackend", embedder._backend)
        local_model: SentenceTransformer = backend_hf._load_model()

        _get_latest_commit_hash.cache_clear()
        try:
            first_s = time_once(backend_hf.get_hash)
            network_ok = True
        except Exception as exc:  # noqa: BLE001 - record offline/rate-limit and continue
            first_s = float("nan")
            network_ok = False
            print(f"[embed_get_hash] HF commit-hash lookup failed ({exc!s}); continuing.")
        cached_s = timeit(backend_hf.get_hash, repeats=5)

        with tempfile.TemporaryDirectory() as tmp:
            local_dir = Path(tmp) / "local_model"
            local_model.save(str(local_dir))
            backend_local = SentenceTransformerEmbeddingBackend(
                SentenceTransformerEmbeddingConfig(model_name=str(local_dir))
            )
            local_first_s = time_once(backend_local.get_hash)
            local_cached_s = timeit(backend_local.get_hash, repeats=3)

        rows.append(
            {
                "case": "hf_name_first_call",
                "seconds": round(first_s, 5),
                "note": "queries HF Hub for commit SHA" + ("" if network_ok else " (FAILED - offline)"),
            }
        )
        rows.append(
            {"case": "hf_name_lru_cached", "seconds": round(cached_s, 6), "note": "subsequent calls hit lru_cache"}
        )
        rows.append(
            {
                "case": "local_path_first_call",
                "seconds": round(local_first_s, 5),
                "note": "pickles + hashes EVERY model parameter tensor",
            }
        )
        rows.append(
            {
                "case": "local_path_repeat",
                "seconds": round(local_cached_s, 5),
                "note": "no memoization: re-hashes all params each call",
            }
        )

    frame = write_results("embed_get_hash", rows)
    print("\n[embed_get_hash] get_hash() cost by model-identity source")
    print(frame.to_string(index=False))


def run_reuse() -> None:
    """Analytic reuse/redundancy under realistic access patterns (whole-list keying)."""
    rows: list[dict[str, object]] = []

    def evaluate(name: str, lists: list[list[str]], description: str) -> None:
        seen: set[tuple[str, ...]] = set()
        hits = 0
        computed = 0
        unique_utts: set[str] = set()
        for lst in lists:
            key = tuple(lst)
            unique_utts.update(lst)
            if key in seen:
                hits += 1
            else:
                seen.add(key)
                computed += len(lst)
        n_calls = len(lists)
        ideal = len(unique_utts)
        rows.append(
            {
                "pattern": name,
                "description": description,
                "n_calls": n_calls,
                "hits": hits,
                "misses": n_calls - hits,
                "hit_rate": round(hits / n_calls, 3),
                "utterances_computed": computed,
                "utterances_ideal": ideal,
                "redundancy_factor": round(computed / ideal, 2) if ideal else None,
            }
        )

    pool = gen_utterances(1_400, seed=7)

    fixed = pool[:1_000]
    evaluate(
        "fixed_data_across_trials",
        [fixed for _ in range(20)],
        "same train set embedded across 20 hyperparameter trials (ideal case)",
    )
    evaluate(
        "growing_dataset",
        [pool[: 500 + 100 * step] for step in range(10)],
        "dataset grows by 100 utterances over 10 runs",
    )
    rng = np.random.default_rng(0)
    folds = []
    base = pool[:1_000]
    for _ in range(5):
        idx = rng.permutation(1_000)
        train_idx = sorted(idx[:800].tolist())
        folds.append([base[i] for i in train_idx])
    evaluate("kfold_cv", folds, "5-fold CV, each fold embeds its 800-utterance train split")
    reshuffled = []
    for s in range(5):
        order = np.random.default_rng(s).permutation(1_000)
        reshuffled.append([base[i] for i in order.tolist()])
    evaluate("reshuffled_reruns", reshuffled, "same 1000 utterances re-embedded in 5 different orders")

    frame = write_results("embed_reuse", rows)
    print("\n[embed_reuse] achieved reuse under access patterns")
    print(frame.to_string(index=False))


def run_storage() -> None:
    """Projected on-disk footprint and duplication under the reuse patterns."""
    rows: list[dict[str, object]] = []

    with temp_cache() as cache_base:
        embedder = _make_embedder()
        embedder.embed(gen_utterances(1_000, seed=999))
        measured = embeddings_dir(cache_base)
        sample_file = next(measured.glob("*.npy"))
        bytes_per_utt = sample_file.stat().st_size / 1_000

    patterns: list[tuple[str, list[int], int]] = [
        ("fixed_data_across_trials", [1_000], 1_000),
        ("growing_dataset", [500 + 100 * step for step in range(10)], 1_400),
        ("kfold_cv", [800] * 5, 1_000),
        ("reshuffled_reruns", [1_000] * 5, 1_000),
    ]
    for name, stored_lists, unique_utts in patterns:
        stored_utts = sum(stored_lists)
        rows.append(
            {
                "pattern": name,
                "files_on_disk": len(stored_lists),
                "stored_utterance_vectors": stored_utts,
                "unique_utterance_vectors": unique_utts,
                "duplication_factor": round(stored_utts / unique_utts, 2),
                "disk_mb": round(stored_utts * bytes_per_utt / 1e6, 2),
                "ideal_mb": round(unique_utts * bytes_per_utt / 1e6, 2),
            }
        )

    frame = write_results("embed_storage", rows)
    print(f"\n[embed_storage] footprint (measured {bytes_per_utt:.0f} bytes/vector)")
    print(frame.to_string(index=False))


def run() -> None:
    """Run all embedding-cache scenarios."""
    run_keying()
    run_perf()
    run_get_hash()
    run_reuse()
    run_storage()
