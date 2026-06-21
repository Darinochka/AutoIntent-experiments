# Evaluation design — AutoIntent 0.3.1 caching efficiency

This document describes (1) how the cache works in AutoIntent 0.3.1, (2) what we measure
and why, and (3) the concrete scenarios. The findings/numbers live in `REPORT.md`; this is
the methodology.

## 1. How the cache works (0.3.1)

AutoIntent ships **two independent, file-based caches**, both keyed by `xxhash.xxh64` digests
and both rooted at `appdirs.user_cache_dir("autointent")` (on macOS: `~/Library/Caches/autointent`).

### 1a. Embeddings cache
Files: `autointent/_wrappers/embedder/{sentence_transformers,openai,vllm}.py`,
`autointent/_wrappers/embedder/utils.py`, `autointent/_hash.py`.

- `EmbedderConfig.use_cache` defaults to **True** (`configs/_embedder.py`).
- On `embed(utterances, task_type)`, when `use_cache` is on, the backend builds a key:
  ```
  h = Hasher()
  h.update(self.get_hash())          # model identity + max_length
  h.update(utterances)               # the WHOLE list, via pickle.dumps(list)
  if prompt: h.update(prompt)
  path = user_cache_dir/embeddings/{h.hexdigest()}.npy
  ```
- If `path` exists → `np.load(path)` and return. Else compute, then `np.save(path, matrix)`.
- The cached value is the **entire embedding matrix for that exact list** in one `.npy`.
- `get_hash()` (sentence-transformers): for a HF model name, hashes the **HF commit SHA**
  (resolved via `huggingface_hub.model_info`, memoized with `lru_cache`) plus `max_length`.
  For a **local path** model it instead pickles **every model parameter tensor**. The model
  *name string itself is never hashed* — only the resolved SHA/params.
  - 0.3.1 change: on HF Hub error (offline/rate-limit) it falls back to the **revision string
    `"main"`** (0.3.0 fell back to the model name).

### 1b. LLM structured-output cache
Files: `autointent/generation/_cache.py` (`StructuredOutputCache`), `_generator.py`,
`_dump_tools/unit_dumpers.py` (`PydanticModelDumper`).

- `Generator(use_cache=True)` (default) constructs `StructuredOutputCache(use_cache=True)`,
  whose `__init__` calls `_load_existing_cache()` — intended to preload **all** entries into RAM.
- Key = `xxh64(json(messages) + json(output_model.model_json_schema()) + json(generation_params))`.
  The **model name and base_url are NOT part of the key** (only `generation_params` is, and the
  model name is stored separately on the `Generator`).
- `get()` → memory cache → else `_load_from_disk()`. `set()` → memory + `_save_to_disk()`.
- `_save_to_disk` uses `PydanticModelDumper.dump`, which **creates a directory per entry**
  (`…/structured_outputs/<key>/`) holding `class_info.json` + `model_dump.json`.
- `_load_existing_cache` lists `cache_dir.iterdir()` and keeps only `f.is_file()` — but entries
  are **directories**, so this filter excludes them (a mismatch worth verifying empirically).

## 2. What we measure and why

The cache exists "to avoid costful recomputations". So efficiency = *how much compute/cost it
actually saves*, *how much overhead it adds*, *how it scales*, and *whether it is correct/safe*.

| Aspect | Question | Why it matters |
|---|---|---|
| Hit/miss correctness | Does the key hit when it should and miss when it should? | A cache that misses on trivially-equivalent inputs wastes the savings; one that hits when it shouldn't returns wrong data. |
| Hit speedup | How much faster is a warm call vs a cold call? | The headline benefit. |
| Lookup/write overhead | What does the cache cost on every call (key build, `get_hash`, disk IO)? | Overhead is paid even on hits and on misses; if large it erodes the benefit. |
| Scaling | How do warm-load time, key-build time, startup time, storage grow with N (list size) and K (number of entries)? | Determines behaviour at realistic benchmark scale. |
| Reuse / key granularity | Under realistic access patterns, what fraction of work is actually reused? | Whole-list keys may defeat reuse across overlapping inputs. |
| Storage footprint | Bytes and inodes per entry; growth over time | File-based caches grow unbounded with no eviction. |
| Concurrency / robustness | Are writes atomic? Safe under parallel workers? | Optuna/HTTP server can hit the same cache concurrently. |

## 3. Scenarios

Everything runs **without real API calls** (per request): the embeddings path uses a small
**local** sentence-transformers model (real cache code path), and the OpenAI/LLM path is
**simulated** — `StructuredOutputCache` is exercised directly, and where a `Generator` is needed
its network method is stubbed (a dummy `OPENAI_API_KEY` is set; no request is sent). All cache IO
is redirected to a temp scratch dir by monkeypatching `user_cache_dir` in the two cache modules,
so the user's real cache is never touched.

**Embeddings**
- `embed_keying` — for a base list, confirm hit on identical input and miss on: reorder, subset,
  superset, one-char edit, different prompt. Verify a hit returns the bit-identical matrix.
- `embed_perf` — for N ∈ {100, 500, 1k, 5k, 10k}: cold vs warm `embed()` time; decompose warm
  cost into `get_hash()`, list-hashing (`Hasher.update(utterances)`), and `np.load`. Measure
  `get_hash()` for HF-name (first vs lru-cached) and for a local-path model (param pickling).
- `embed_reuse` — analytic reuse/redundancy under access patterns: fixed-data-across-trials
  (ideal), growing dataset, k-fold CV, reshuffled re-runs. Report achieved hit-rate and
  redundant-utterances-embedded vs an idealized per-utterance cache.
- `embed_storage` — measured bytes/file for several N; projected footprint and duplication under
  the patterns above.

**LLM structured outputs**
- `struct_preload` — write K entries, inspect on-disk layout (dir vs file), then construct a fresh
  `StructuredOutputCache` and measure init time and resulting `_memory_cache` size vs K
  (tests whether the preload actually loads anything).
- `struct_ops` — per-entry `set()` time, cold `get()` (disk) time, warm `get()` (memory) time, vs
  payload size.
- `struct_collision` — show the key ignores model identity: two different "models" with identical
  messages/schema/params collide (cache returns the first model's output). Demonstrated at cache
  level and via two stubbed `Generator`s.
- `struct_storage` — files/inodes and bytes per entry; growth vs K.

**Robustness**
- `concurrency` — characterize write atomicity: code uses `np.save`/multi-file JSON dump straight
  to the final path (no tmp+rename), so a concurrent reader can observe a partial file. Demonstrate
  a torn read and measure parallel-write throughput. We are explicit about what is *demonstrated*
  vs *argued*.

Each scenario writes raw results to `results/<name>.{json,csv}`; `run_all.py` runs them and
prints a summary. Plots (where useful) go to `results/*.png`.
