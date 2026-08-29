"""Structured-output (LLM) cache scenarios: preload, ops, cross-model collision, storage."""

from __future__ import annotations

import os
from functools import partial
from statistics import median
from types import SimpleNamespace
from typing import Any

from autointent.generation import Generator
from autointent.generation._cache import StructuredOutputCache

from cache_bench.harness import (
    count_entries,
    count_inodes,
    dir_size_bytes,
    structured_dir,
    temp_cache,
    time_once,
    timeit,
    write_results,
)
from cache_bench.models import IntentPrediction, build_messages, make_prediction

GEN_PARAMS: dict[str, Any] = {"temperature": 0.0}


def _populate(cache: StructuredOutputCache, k: int) -> None:
    for i in range(k):
        cache.set(build_messages(f"utterance number {i}"), IntentPrediction, GEN_PARAMS, make_prediction(i))


def run_preload() -> None:
    """Check whether StructuredOutputCache.__init__ actually preloads entries into RAM."""
    rows: list[dict[str, object]] = []

    for k in (0, 50, 200, 1_000):
        with temp_cache() as base:
            writer = StructuredOutputCache(use_cache=True)
            _populate(writer, k)
            sdir = structured_dir(base)
            n_dirs = count_entries(sdir, files=False, dirs=True)
            n_files = count_entries(sdir, files=True, dirs=False)

            fresh = StructuredOutputCache(use_cache=True)  # triggers _load_existing_cache
            init_s = timeit(lambda: StructuredOutputCache(use_cache=True), repeats=3)
            memory_loaded = len(fresh._memory_cache)

            # functional correctness despite preload behaviour: a disk hit still works
            disk_hit = None
            if k > 0:
                got = fresh.get(build_messages("utterance number 0"), IntentPrediction, GEN_PARAMS)
                disk_hit = got is not None and got.label == make_prediction(0).label

            rows.append(
                {
                    "entries_written": k,
                    "on_disk_dirs": n_dirs,
                    "on_disk_files": n_files,
                    "preloaded_into_memory": memory_loaded,
                    "init_seconds": round(init_s, 5),
                    "disk_hit_still_works": disk_hit,
                }
            )

    frame = write_results("struct_preload", rows)
    print("\n[struct_preload] does __init__ preload? (entries_written vs preloaded_into_memory)")
    print(frame.to_string(index=False))


def run_ops() -> None:
    """Per-operation latency: set (write), cold get (disk), warm get (memory)."""
    rows: list[dict[str, object]] = []
    n_ops = 200

    with temp_cache() as base:
        cache = StructuredOutputCache(use_cache=True)

        set_samples: list[float] = []
        for i in range(n_ops):
            msgs = build_messages(f"set op {i}")
            result = make_prediction(i)
            set_samples.append(time_once(partial(cache.set, msgs, IntentPrediction, GEN_PARAMS, result)))

        get_disk_samples: list[float] = []
        get_mem_samples: list[float] = []
        for i in range(n_ops):
            msgs = build_messages(f"set op {i}")
            cache._memory_cache.clear()  # force a disk read
            get_disk_samples.append(time_once(partial(cache.get, msgs, IntentPrediction, GEN_PARAMS)))
            get_mem_samples.append(time_once(partial(cache.get, msgs, IntentPrediction, GEN_PARAMS)))

        sdir = structured_dir(base)
        n_entries = count_entries(sdir, files=False, dirs=True)
        total_bytes = dir_size_bytes(sdir)
        rows.append(
            {
                "set_median_ms": round(median(set_samples) * 1e3, 4),
                "get_disk_median_ms": round(median(get_disk_samples) * 1e3, 4),
                "get_memory_median_ms": round(median(get_mem_samples) * 1e3, 5),
                "entries": n_entries,
                "bytes_per_entry": round(total_bytes / n_entries, 1) if n_entries else None,
            }
        )

    frame = write_results("struct_ops", rows)
    print("\n[struct_ops] per-operation latency")
    print(frame.to_string(index=False))


def _fake_parse_factory(label: str, counter: list[int]) -> object:
    """Build a stand-in for ``client.beta.chat.completions.parse`` that never hits the network."""

    def fake_parse(**_kwargs: object) -> object:
        counter[0] += 1
        prediction = IntentPrediction(label=label, confidence=0.99, rationale=f"from {label}")
        message = SimpleNamespace(parsed=prediction, content=prediction.model_dump_json())
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    return fake_parse


def _stub_generator(model_name: str, label: str, counter: list[int]) -> Generator:
    """Construct a Generator whose network call is stubbed out."""
    generator = Generator(model_name=model_name, use_cache=True, **GEN_PARAMS)
    completions = SimpleNamespace(parse=_fake_parse_factory(label, counter))
    fake_client = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=completions)))
    generator.client = fake_client  # type: ignore[assignment]
    return generator


def run_collision() -> None:
    """The structured-output key ignores model identity -> different models collide."""
    rows: list[dict[str, object]] = []
    os.environ.setdefault("OPENAI_API_KEY", "sk-dummy-no-network")
    messages = build_messages("please classify this utterance")

    with temp_cache() as base:
        cache = StructuredOutputCache(use_cache=True)
        key_a = cache._get_cache_key(messages, IntentPrediction, GEN_PARAMS)
        key_b = cache._get_cache_key(messages, IntentPrediction, GEN_PARAMS)
        cache.set(messages, IntentPrediction, GEN_PARAMS, IntentPrediction(label="A", confidence=1.0, rationale="A"))
        served_to_b = cache.get(messages, IntentPrediction, GEN_PARAMS)
        rows.append(
            {
                "level": "cache",
                "detail": "same messages/schema/params, two different models",
                "key_identical_across_models": key_a == key_b,
                "model_b_served_model_a_value": served_to_b is not None and served_to_b.label == "A",
                "second_backend_call_made": None,
            }
        )

        # End-to-end via two stubbed Generators sharing the (patched) cache dir.
        calls_a = [0]
        calls_b = [0]
        gen_a = _stub_generator("model-A", "A", calls_a)
        gen_b = _stub_generator("model-B", "B", calls_b)
        out_a = gen_a.get_structured_output_sync(messages, IntentPrediction)
        out_b = gen_b.get_structured_output_sync(messages, IntentPrediction)
        rows.append(
            {
                "level": "generator",
                "detail": "model-A then model-B with identical messages/params",
                "key_identical_across_models": None,
                "model_b_served_model_a_value": out_b.label == "A" and out_a.label == "A",
                "second_backend_call_made": bool(calls_b[0]),
            }
        )
        _ = base

    frame = write_results("struct_collision", rows)
    print("\n[struct_collision] model identity is NOT part of the key")
    print(frame.to_string(index=False))


def run_storage() -> None:
    """Inode and byte footprint of the structured-output cache vs entry count."""
    rows: list[dict[str, object]] = []

    for k in (100, 1_000):
        with temp_cache() as base:
            cache = StructuredOutputCache(use_cache=True)
            _populate(cache, k)
            sdir = structured_dir(base)
            total_bytes = dir_size_bytes(sdir)
            inodes = count_inodes(sdir)
            rows.append(
                {
                    "entries": k,
                    "filesystem_inodes": inodes,
                    "inodes_per_entry": round(inodes / k, 2),
                    "total_kb": round(total_bytes / 1e3, 1),
                    "bytes_per_entry": round(total_bytes / k, 1),
                }
            )

    frame = write_results("struct_storage", rows)
    print("\n[struct_storage] footprint vs entry count")
    print(frame.to_string(index=False))


def run() -> None:
    """Run all structured-output cache scenarios."""
    run_preload()
    run_ops()
    run_collision()
    run_storage()
