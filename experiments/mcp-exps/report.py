"""Download and visualize experiment results.

This module contains CLI commands:

1. Default command: download experiment spans from Logfire and write a JSONL report.
2. `table` command: read a JSONL report and print a Rich summary.
3. `from-link` command: resolve experiment name from a public URL (`spanId`) and load it like `load`.
4. `aggregate-links` command: merge several public trace URLs into one JSONL (e.g. CV folds).

## Usage examples

### 1) Download a report from Logfire
```bash
export LOGFIRE_API_KEY="..."
uv run report.py --experiment basic-fs --output-dir ./reports
```

The output will be saved as `./reports/basic-fs.jsonl` (one JSON object per line):
- first line: experiment header (trace id + aggregated metrics)
- following lines: per-case rows (evaluator scores)

Header **token and cost** totals are the **sum** over leaf LLM spans whose names match `chat <model>`
(e.g. `chat gpt-5.4-mini`), using each span's `logfire.metrics` rollup. Parent evaluate-span metrics are
not used (they reflect pydantic-evals **averages** across tasks, not experiment-wide totals).

### 2) Pretty-print the report
```bash
uv run report.py table --report-path ./reports/basic-fs.jsonl
```

### 3) Load a report using a public trace URL
Uses ``spanId`` from the URL to resolve the pydantic-evals experiment name (same string as
``evaluate <name>``), then runs the same ``query`` + JSONL write path as ``load``.
Assumes ``experiment-name`` is unique per run so ``trace_id`` filtering is unnecessary.

```bash
uv run report.py from-link "https://logfire-eu.pydantic.dev/public-trace/…?spanId=…"
```
Writes ``./<experiment>_<trace-prefix>.jsonl`` by default (``trace-prefix`` from the resolved trace for
filenames; see ``--help`` for ``--output``).

### 4) Aggregate several public trace URLs (e.g. cross-validation folds)
Each ``--link`` is resolved like ``from-link``; one trace per link is narrowed and merged. Case rows are
written as ``{case_name}__{trace_prefix}`` so identical task names across folds stay distinct. Header
``trace_id`` lists all trace UUIDs separated by ``;``.

Optional ``--inter-link-delay`` (default 20s) spaces Logfire queries when merging many folds. Each link
runs two SQL queries; ``query`` also waits 10s between them.

```bash
uv run report.py aggregate-links --name ts-fs-cv-gpt54 \
  --link "https://logfire-eu.pydantic.dev/public-trace/…?spanId=…" \
  --link "https://logfire-eu.pydantic.dev/public-trace/…?spanId=…"
```
"""

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import cyclopts
from dotenv import load_dotenv
from logfire.query_client import AsyncLogfireQueryClient
from loguru import logger
from rich import box
from rich.console import Console
from rich.table import Table

from src.report import (
    CaseRow,
    ExperimentHeader,
    merge_logfire_eval_fetch_results,
    narrow_eval_fetch_to_trace,
    parse_span_id_from_public_trace_url,
    query,
    resolve_experiment_for_span,
    trace_prefix,
    write_experiment_jsonl,
)

if TYPE_CHECKING:
    from src.report.models import LogfireEvalFetchResult

load_dotenv()

app = cyclopts.App(name="Report maker")


@app.default
async def load(
    experiment: Annotated[str, cyclopts.Parameter(help="Experiment name (like basic-fs)")],
    output_dir: Annotated[
        Path | None,
        cyclopts.Parameter(help="Where to save jsonl file. CWD by default"),
    ] = None,
    timeout: Annotated[  # noqa: ASYNC109
        int,
        cyclopts.Parameter(help="Query timeout in seconds"),
    ] = 10,
) -> None:
    """Load raw data from Logfire and save to JSONL file.

    This script finds all runs of the experiment with this name, collects cases and saved them
    to a JSONL file. The file starts with a header which is followed by per-case rows.
    """
    output_path = ((output_dir or Path.cwd()).resolve() / experiment).with_suffix(".jsonl")
    logger.info(f"Result will be saved to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    async with AsyncLogfireQueryClient(
        read_token=os.getenv("LOGFIRE_API_KEY") or "fake",
        timeout=timeout,  # type: ignore[arg-type]
    ) as client:
        bundle = await query(client, experiment)

    if bundle is None:
        logger.warning("Logfire query returned no rows.")
        return

    write_experiment_jsonl(output_path, experiment, bundle)
    logger.success("Done!")


@app.command(name="from-link")
async def from_link(
    url: Annotated[
        str,
        cyclopts.Parameter(help="Logfire public trace URL including ?spanId=… (path UUID is ignored)"),
    ],
    output_dir: Annotated[
        Path | None,
        cyclopts.Parameter(help="Directory for the JSONL file (default: current working directory)"),
    ] = None,
    output: Annotated[
        Path | None,
        cyclopts.Parameter(help="Explicit output JSONL path (overrides default `<experiment>_<trace-prefix>.jsonl`)"),
    ] = None,
    timeout: Annotated[  # noqa: ASYNC109
        int,
        cyclopts.Parameter(help="Query timeout in seconds"),
    ] = 10,
) -> None:
    """Load using a public Logfire link.

    Parses ``spanId``, resolves the pydantic-evals experiment name (``evaluate <name>``) via SQL, then runs the same
    pipeline as ``load``. Requires ``experiment-name`` to be unique for your runs (same assumption as ``load``).
    """
    span_id = parse_span_id_from_public_trace_url(url)
    logger.info(f"Using span_id={span_id}")

    base_dir = (output_dir or Path.cwd()).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    async with AsyncLogfireQueryClient(
        read_token=os.getenv("LOGFIRE_API_KEY") or "fake",
        timeout=timeout,  # type: ignore[arg-type]
    ) as client:
        trace_id, experiment = await resolve_experiment_for_span(client, span_id)
        logger.info(f"Resolved experiment={experiment!r} trace_id={trace_id}")
        bundle = await query(client, experiment)

    if bundle is None:
        logger.warning("Logfire query returned no rows.")
        return

    if output is not None:
        output_path = output.resolve()
    else:
        stem = f"{experiment}_{trace_prefix(trace_id)}"
        output_path = (base_dir / stem).with_suffix(".jsonl")

    logger.info(f"Result will be saved to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_experiment_jsonl(output_path, experiment, bundle)
    logger.success("Done!")


@app.command(name="aggregate-links")
async def aggregate_links(
    name: Annotated[
        str,
        cyclopts.Parameter("--name", help="Label for header.experiment_name and default output stem"),
    ],
    links: Annotated[
        list[str],
        cyclopts.Parameter(
            "--link",
            help="Public Logfire trace URL with ?spanId= (repeat per trace to merge)",
            allow_repeating=True,
        ),
    ],
    output_dir: Annotated[
        Path | None,
        cyclopts.Parameter(help="Directory for the JSONL file (default: current working directory)"),
    ] = None,
    output: Annotated[
        Path | None,
        cyclopts.Parameter(help="Explicit output JSONL path (overrides default `<name>.jsonl`)"),
    ] = None,
    timeout: Annotated[  # noqa: ASYNC109
        int,
        cyclopts.Parameter(help="Query timeout in seconds"),
    ] = 10,
    inter_link_delay: Annotated[
        float,
        cyclopts.Parameter(
            help="Seconds to wait before each link after the first (reduces Logfire rate limits on many CV folds)",
        ),
    ] = 20.0,
) -> None:
    """Merge several public trace URLs into one JSONL report.

    Resolves each link to ``(trace_id, experiment)``, loads Logfire data with the same pipeline as
    ``from-link`` (query then narrow to that trace), concatenates bundles, and writes one file. Per-case
    names are suffixed with ``__<trace-prefix>`` so CV folds do not collide.
    """
    if not links:
        msg = "Provide at least one --link URL"
        raise ValueError(msg)

    base_dir = (output_dir or Path.cwd()).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    narrowed_parts: list[LogfireEvalFetchResult] = []
    async with AsyncLogfireQueryClient(
        read_token=os.getenv("LOGFIRE_API_KEY") or "fake",
        timeout=timeout,  # type: ignore[arg-type]
    ) as client:
        for i, url in enumerate(links):
            if i > 0 and inter_link_delay > 0:
                logger.info(f"Waiting {inter_link_delay}s before next link (rate limit backoff)")
                await asyncio.sleep(inter_link_delay)
            span_id = parse_span_id_from_public_trace_url(url)
            trace_id, experiment = await resolve_experiment_for_span(client, span_id)
            logger.info(f"[{i + 1}/{len(links)}] Resolved experiment={experiment!r} trace_id={trace_id}")
            bundle = await query(client, experiment)
            if bundle is None:
                msg = f"Logfire returned no rows for experiment={experiment!r} (link index {i})"
                raise RuntimeError(msg)
            narrowed = narrow_eval_fetch_to_trace(bundle, trace_id)
            if narrowed is None:
                msg = f"No case rows for trace_id={trace_id!r} after query (link index {i})"
                raise RuntimeError(msg)
            narrowed_parts.append(narrowed)

    merged = merge_logfire_eval_fetch_results(narrowed_parts)

    output_path = output.resolve() if output is not None else (base_dir / name).with_suffix(".jsonl")

    logger.info(
        f"Writing merged report ({len(merged.trace_order)} traces, {len(merged.case_rows)} case rows) to {output_path}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_experiment_jsonl(output_path, name, merged, disambiguate_cases_by_trace=True)
    logger.success("Done!")


@app.command(name="table", help="Read a JSONL report and print a rich summary table.")
def print_table(
    report_path: Annotated[Path, cyclopts.Parameter(help="Path to JSONL report file")],
    sort_cases: Annotated[
        Literal["passed", "name", "input_tokens", "output_tokens"],
        cyclopts.Parameter(help="Sort cases by: passed|name|input_tokens|output_tokens"),
    ] = "passed",
) -> None:
    """Print parsed report metrics using `rich`."""
    console = Console()

    with report_path.open("r", encoding="utf-8") as f:
        header_line = f.readline()
        if not header_line.strip():
            msg = f"Empty report file: {report_path}"
            raise ValueError(msg)
        header = ExperimentHeader.model_validate_json(header_line)

        cases: list[CaseRow] = []
        for line_ in f:
            line = line_.strip()
            if not line:
                continue
            cases.append(CaseRow.model_validate_json(line))

    summary = Table(title="Experiment Summary", box=box.SIMPLE)
    summary.add_column("Experiment")
    summary.add_column("Trace ID")
    summary.add_column("Cost")
    summary.add_column("Input Tokens")
    summary.add_column("Output Tokens")
    summary.add_column("Cache Read Tokens")
    summary.add_column("Requests")
    summary.add_column("Total Tasks")
    summary.add_column("Passed Tasks")
    summary.add_row(
        header.experiment_name,
        header.trace_id,
        f"{header.cost:.6g}",
        f"{header.input_tokens:.0f}",
        f"{header.output_tokens:.0f}",
        f"{header.cache_read_tokens:.0f}",
        f"{header.requests:.2f}",
        str(header.total_tasks),
        str(header.passed_tasks),
    )
    console.print(summary)

    if sort_cases == "passed":
        cases_sorted = sorted(cases, key=lambda c: (not c.passed, c.case_name))
    elif sort_cases == "name":
        cases_sorted = sorted(cases, key=lambda c: c.case_name)
    elif sort_cases == "input_tokens":
        cases_sorted = sorted(cases, key=lambda c: c.metrics.input_tokens, reverse=True)
    elif sort_cases == "output_tokens":
        cases_sorted = sorted(cases, key=lambda c: c.metrics.output_tokens, reverse=True)
    else:
        raise ValueError("sort_cases must be one of: passed|name|input_tokens|output_tokens")

    per_case = Table(title="Per-case Results", box=box.SIMPLE_HEAVY)
    per_case.add_column("Case")
    per_case.add_column("Passed")
    per_case.add_column("Input Tokens")
    per_case.add_column("Output Tokens")

    for c in cases_sorted:
        per_case.add_row(
            c.case_name,
            "true" if c.passed else "false",
            f"{c.metrics.input_tokens:.0f}",
            f"{c.metrics.output_tokens:.0f}",
        )
    console.print(per_case)


if __name__ == "__main__":
    app()
