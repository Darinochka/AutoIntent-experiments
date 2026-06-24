# caching-efficiency

Benchmarks for the efficiency of **AutoIntent 0.3.1**'s two file-based caches:

- the **embeddings** cache (`Embedder`, `.npy` per exact utterance-list), and
- the **LLM structured-output** cache (`Generator` / `StructuredOutputCache`, a directory per entry).

The goal: measure how much the caches actually save, what overhead/limits they have, and whether
they are correct and safe — then propose improvements (from small fixes to a SQLite/LMDB rewrite).

## Read these
- **[REPORT.md](REPORT.md)** — findings, root causes, and proposals (the thing to review).
- **[DESIGN.md](DESIGN.md)** — how the cache works and the evaluation methodology.

## Run
```bash
uv sync
uv run python run_all.py             # all scenarios -> results/*.{json,csv}
uv run python run_all.py structured  # one group: embeddings | structured | concurrency
uv run python make_plots.py          # results/*.png
uv run ruff check . && uv run mypy .  # lint + types (clean)
```

No API keys and no network calls to OpenAI are needed: the OpenAI/LLM path is simulated and the
embeddings path uses a small local sentence-transformers model. Cache IO is redirected to a temp
dir, so your real `~/Library/Caches/autointent` is untouched.

## Layout
```
cache_bench/
  harness.py            # temp-cache isolation, timing, dataset gen, result writers
  models.py             # Pydantic output model + message builders
  scenarios/
    embeddings.py       # keying correctness, cold/warm perf, get_hash cost, reuse, storage
    structured.py       # preload behaviour, op latency, cross-model collision, storage
    concurrency.py      # non-atomic-write poisoning
run_all.py              # orchestrator
make_plots.py           # figures from the CSVs
results/                # CSV + JSON + PNG outputs
```
