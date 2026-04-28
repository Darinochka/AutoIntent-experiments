"""Download and visualize experiment results.

This module contains two CLI commands:

1. Default command: download experiment spans from Logfire and write a JSONL report.
2. `table` command: read a JSONL report and print a Rich summary.

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
"""

import os
from pathlib import Path
from typing import Annotated, Literal

import cyclopts
from dotenv import load_dotenv
from logfire.query_client import AsyncLogfireQueryClient
from loguru import logger
from rich import box
from rich.console import Console
from rich.table import Table

from src.report import CaseRow, ExperimentHeader, query, write_experiment_jsonl

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
