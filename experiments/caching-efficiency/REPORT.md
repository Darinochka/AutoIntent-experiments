# AutoIntent 0.3.1 caching efficiency — report

**TL;DR.** AutoIntent's two file-based caches are **very effective for the workload they were
designed for** (re-embedding/re-querying the *same* fixed inputs across many hyperparameter
trials): warm reads are **1000–4400× faster** than recompute, with negligible key overhead.
But outside exact-input-match they degrade or misbehave, and there are **four real correctness/
robustness bugs**. None of the problems require a tech-stack change to fix; a SQLite/LMDB move is
*optional* and mostly buys robustness/operability, not hot-path speed.

- Methodology & cache internals: [DESIGN.md](DESIGN.md)
- Raw numbers: [`results/*.csv`](results/) and `results/*.json`
- Figures: `results/embed_perf.png`, `results/embed_reuse.png`, `results/struct_preload.png`
- Reproduce: `uv run python run_all.py && uv run python make_plots.py`

All measurements: macOS (Darwin 25.2), Python 3.12, CPU, `sentence-transformers/all-MiniLM-L6-v2`
(384-dim). The OpenAI/LLM path is **simulated** (no API calls): the structured-output cache is
exercised directly and the `Generator`'s network call is stubbed. All cache IO is redirected to a
temp dir, so the user's real cache was never touched.

---

## 1. What the cache does (one paragraph each)

**Embeddings cache** (`use_cache=True` by default). On `embed(list_of_utterances)` it builds a key
`xxh64(pickle(model_identity) + pickle(whole_list) + prompt)` and stores/loads the **entire
embedding matrix for that exact list** as one `.npy` under `~/Library/Caches/autointent/embeddings/`.
`model_identity` is the HF commit SHA (network call, `lru_cache`d) — or, for a local-path model,
**a hash of every parameter tensor**. The model *name* is never part of the key.

**LLM structured-output cache** (`use_cache=True` by default). `Generator.get_structured_output_*`
keys on `xxh64(json(messages) + json(output_schema) + json(generation_params))` and stores each
result as a **directory** (`structured_outputs/<key>/` with `class_info.json` + `model_dump.json`).
The **model name / base_url are not part of the key**. `__init__` tries to preload all entries into
RAM.

---

## 2. The good: huge wins on the designed workload

### 2.1 Warm reads are ~1000–4400× faster than recompute
`results/embed_perf.csv` (see `results/embed_perf.png`):

| utterances | cold (compute+save) | warm (hash+load) | speedup | key-build | np.load | file |
|-----------:|--------------------:|-----------------:|--------:|----------:|--------:|-----:|
| 100   | 0.110 s | 0.088 ms | **1248×** | 5 µs | 39 µs | 0.15 MB |
| 1 000 | 0.804 s | 0.181 ms | **4441×** | 34 µs | 77 µs | 1.54 MB |
| 10 000 | 6.021 s | 5.40 ms | **1116×** | 555 µs | 2.5 ms | 15.4 MB |

For a typical AutoIntent run — one embedder, a fixed train set, dozens of optuna trials over
KNN/linear/retrieval modules that all embed that same set — this is exactly the hit path, and the
cache turns repeated multi-second encodes into sub-millisecond loads. **This is the cache's reason
to exist and it delivers.** Reuse on that pattern is 95–100% (`embed_reuse.csv`, `fixed_data`),
with **no** storage waste (duplication 1.0×).

### 2.2 Key/lookup overhead is small (for HF models)
The per-call bookkeeping paid even on hits — `get_hash()` + hashing the list — is ~5 µs at N=100
and ~0.56 ms at N=10 000, i.e. <0.1% of the cold cost and comparable to `np.load`. For HF-name
models `get_hash()` is ~32 µs after the first (lru-cached) call. Fine.

### 2.3 The structured-output cache is cheap per op
`results/struct_ops.csv`: `set` ≈ 0.28 ms, cold `get` (disk) ≈ 0.14 ms, warm `get` (memory)
≈ 0.076 ms. All sub-millisecond — trivially worth it against a multi-second LLM call.

---

## 3. The bad: where efficiency breaks down

### 3.1 Whole-list keying gives **zero** reuse outside exact match (biggest efficiency issue)
The embedding key hashes the *entire ordered list*. Any change — reorder, add one item, drop one,
edit one character, change the prompt — is a full miss (`embed_keying.csv`: all correct, all
misses). Consequently (`embed_reuse.csv`, `embed_reuse.png`):

| access pattern | hit rate | redundant compute | redundant storage |
|---|---:|---:|---:|
| fixed data across 20 trials | 0.95 | **1.0×** | 1.0× |
| growing dataset (+100 ×10) | 0.00 | **6.8×** | 6.8× (14.6 MB vs 2.2 MB ideal) |
| 5-fold CV (overlapping splits) | 0.00 | **4.0×** | 4.0× |
| same set, reshuffled ×5 | 0.00 | **5.0×** | 5.0× |

So incremental data, cross-validation, or any reordering recompute and re-store everything. A
per-utterance cache would have reused the overlap and cut both compute and disk by 4–7×.

### 3.2 Local/fine-tuned models pay a huge per-call key cost
`results/embed_get_hash.csv`: for a **local-path** model, `get_hash()` pickles+hashes *every
parameter tensor* — **171 ms first call, 57 ms every subsequent call** (MiniLM, 22 M params; larger
models scale worse). It is recomputed on **every** `embed()`, including cache hits, and is **not
memoized**. After fine-tuning an embedder, this overhead is added to each embed call.

### 3.3 Unbounded growth, no eviction
Neither cache has a size cap, TTL, or eviction. The structured-output cache uses **3 inodes per
entry** (1 dir + 2 JSON files; `struct_storage.csv`), so 100 k cached LLM calls ⇒ ~300 k files.
Embeddings accumulate one uncompressed float32 `.npy` per unique list (1 536 B/vector) forever.

---

## 4. The ugly: correctness & robustness bugs

| # | Bug | Severity | Evidence |
|---|---|---|---|
| B1 | **Structured-output key ignores model identity.** Same messages+schema+params on two different LLMs collide; the second model is served the first model's cached answer. | **High** | `struct_collision.csv`: key identical across models = True; model-B served model-A's value = True; **no second backend call made**. |
| B2 | **Embedding cache can collide across models when offline.** The key uses the HF commit SHA, not the model name; in 0.3.1 the SHA lookup falls back to the literal `"main"` on any HF Hub error. Two different models, same utterance list, same `max_length`, both offline ⇒ same key ⇒ wrong embeddings returned. (0.3.0 fell back to the model name, avoiding this.) | **Medium** | Code: `sentence_transformers.py::_get_latest_commit_hash` returns `rev`; `get_hash()` never hashes `model_name`. |
| B3 | **Non-atomic writes poison the cache.** Both caches write in place (no temp-file + atomic rename); the multi-file JSON dump isn't atomic either. An interrupted or concurrently-observed partial write leaves a corrupt entry, and the read path doesn't guard against it, so the next read **raises and keeps raising** until the entry is deleted by hand. | **Medium-High** (parallel optuna workers / the HTTP/MCP server) | `concurrency.csv`: truncated `.npy` ⇒ `ValueError` on next read, no auto-recovery; entry dir missing `model_dump.json` ⇒ `FileNotFoundError`, no auto-recovery. |
| B4 | **The eager preload loads nothing.** `StructuredOutputCache.__init__ → _load_existing_cache` keeps only `iterdir()` children where `is_file()`, but entries are **directories**, so the in-RAM preload is always empty. | **Low** (currently harmless; latent) | `struct_preload.csv`: entries_written ∈ {50,200,1000} ⇒ preloaded_into_memory = 0; disk hits still work. |

Notes:
- B4 is currently *benign* (lazy disk loads work, and skipping the preload actually avoids an O(K)
  RAM load). But it's a latent trap: "fixing" the dumper to single files would silently enable an
  O(K) full-cache-into-RAM load on **every** `Generator()` construction. Init already does an O(K)
  directory scan (0.003 s at 1 000 entries; grows linearly).

---

## 5. Root causes

- **Coarse key granularity (3.1).** The unit of caching is "an exact list", not "an utterance".
  Reuse therefore requires the caller to present byte-identical, identically-ordered lists.
- **Identity via heavy/incomplete signals (3.2, B2).** Model identity is derived by hashing all
  parameters (expensive, local case) or a network SHA with a constant offline fallback, and the
  cheap, unambiguous discriminator — the model name/path — is omitted from the key.
- **Missing dimension in the LLM key (B1).** The key was built from request *content* but not from
  *which model* produced it; one global namespace is shared across all models.
- **File-as-database (3.3, B3, B4).** Using the filesystem directly (one file/dir per entry, in-place
  writes, `iterdir` scans) gives no atomicity, no transactions, no indexing, no eviction, and an
  inode explosion — all things a real key-value store provides for free.

---

## 6. Proposals (propose-only; not implemented here)

Ordered by value/effort. Items in §6.1 are cheap and fix the actual bugs; §6.2 is the biggest
*efficiency* win; §6.3 is the optional tech-stack change.

### 6.1 Small hacks on the current implementation (do these first)
1. **Atomic writes + self-healing reads** (fixes B3). Write to `path.tmp` then `os.replace()`
   (atomic on POSIX); wrap `np.load`/`PydanticModelDumper.load` in `try/except` and on corruption
   delete + recompute instead of raising. ~30 lines, removes a whole class of failures.
2. **Put the model in the key** (fixes B1, B2). Add `model_name` (+ `base_url`) to the
   structured-output key; add `model_name` to embedding `get_hash()`. Trivial and removes silent
   wrong-data hits.
3. **Memoize `get_hash()` per backend instance** (fixes 3.2). Compute once, cache the int. Turns the
   57–171 ms/call local-model cost into a one-off. ~5 lines.
4. **Decide what the preload is for** (B4). Either make `_load_existing_cache` read directories, or
   delete it and rely on the (working) lazy disk loads. Removes a latent O(K)-RAM trap.
5. **Bounded growth.** Add an opt-in max size / TTL (or at least a documented `clear_cache()` and a
   logged cache-size warning). Optionally `np.savez_compressed` or a float16 option for embeddings.

### 6.2 Per-utterance keying for embeddings (biggest efficiency win)
Switch the embedding cache unit from "exact list" to "single utterance": key each utterance by
`hash(model_identity + utterance + prompt)`. On `embed(list)`, split into cached/uncached, compute
only the misses, and reassemble in order. This converts the 4–7× redundancy in §3.1 into ~1× and
deduplicates storage automatically (a shared utterance is stored once). Medium effort (batched
lookup + order reassembly). **Caveat:** naively this means one tiny file per utterance, which makes
the small-files/inode problem worse — so it pairs naturally with §6.3.

### 6.3 Replace file-per-entry with an embedded key-value store (optional, robustness/operability)
Move both caches behind **SQLite** (WAL mode) or **LMDB**:
- *Structured outputs* → a single SQLite DB: `key TEXT PRIMARY KEY, model TEXT, schema_hash TEXT,
  value JSON, created_at`. Wins: atomic transactions (kills B3), one file instead of 3·K inodes,
  indexed point lookups (the preload becomes unnecessary), trivial eviction/TTL/size queries, safe
  concurrent access. Clean fit — entries are small JSON. Medium-high effort.
- *Embeddings* → store vectors as blobs keyed per-utterance (this is where §6.2 lives), giving reuse
  + dedup + atomicity + eviction in one move. SQLite blobs work; **LMDB** (memory-mapped) is likely
  faster for many small vector reads. For very large matrices a memmap/Parquet sidecar can beat both.
  High effort.

**Honest scoping of §6.3:** the warm read path is *already* sub-millisecond (§2), so SQLite/LMDB
will **not** make hits meaningfully faster. Its value is correctness (atomic writes), operability
(one file, eviction, concurrency), and enabling per-utterance keys without an inode explosion.
Justify it by operational pain (inode counts, parallel workers, unbounded size), not by hit latency.

### Recommended sequence
§6.1 (1→4) immediately — they're small and fix real bugs. Then §6.2 for the embedding efficiency
win. Adopt §6.3 (start with the structured-output SQLite DB) when cache size / concurrency /
inode pressure becomes a real operational problem.

---

## 7. Reproducing

```bash
cd experiments/caching-efficiency
uv sync
uv run python run_all.py            # all groups -> results/*.{json,csv}
uv run python run_all.py embeddings # or: structured | concurrency
uv run python make_plots.py         # results/*.png
uv run ruff check . && uv run mypy . # checks pass clean
```
