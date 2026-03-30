"""Download experiment cases.

Load total cost, tokens and per-case results.
"""

import os
from pathlib import Path
from typing import Annotated, Any, Literal

import cyclopts
from dotenv import load_dotenv
from logfire.query_client import AsyncLogfireQueryClient
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from rich import box
from rich.console import Console
from rich.table import Table

load_dotenv()

app = cyclopts.App(
    # name="pipeline",
    # help="ETL pipeline for offers reranker: load, parse, and split data",
)


PASSED_EPS = 1e-9


class ExperimentHeader(BaseModel):
    """Outpu JSONL file header."""

    experiment_name: str = "unknown"
    trace_id: str
    cost: float
    input_tokens: float
    output_tokens: float
    cache_read_tokens: float
    requests: float
    total_tasks: int = 0
    passed_tasks: int = 0

    model_config = ConfigDict(extra="forbid")


class TraceMetrics(BaseModel):
    """Aggregated metrics."""

    cost: float = 0.0
    input_tokens: float = 0.0
    output_tokens: float = 0.0
    cache_read_tokens: float = 0.0
    requests: float = 0.0

    model_config = ConfigDict(extra="allow")


class CaseMetrics(BaseModel):
    """Single case metrics."""

    cost: float = 0.0
    input_tokens: float = 0.0
    output_tokens: float = 0.0
    cache_read_tokens: float = 0.0
    requests: float = 0.0

    model_config = ConfigDict(extra="allow")


class EvaluatorResult(BaseModel):
    """One evaluator result object inside Logfire `scores`."""

    name: str | None = None
    value: float | None = None
    reason: str | None = None
    source: Any | None = None


class CaseRow(BaseModel):
    """Output from logfire API."""

    case_name: str
    passed: bool
    scores: dict[str, EvaluatorResult] = Field(default_factory=dict)
    metrics: CaseMetrics


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
    """Load raw data from Logfire and save to JSONL file."""
    output_path = ((output_dir or Path.cwd()).resolve() / experiment).with_suffix(".jsonl")
    logger.info(f"Result will be saved to {output_path}")

    query = f"""
    SELECT
        parent.trace_id,
        parent.attributes AS parent_attributes,
        child.attributes AS child_attributes
    FROM records parent
    JOIN records child
        ON child.trace_id = parent.trace_id
    WHERE
        parent.message = 'evaluate {experiment}'
        AND parent.otel_scope_name = 'pydantic-evals'
        AND child.otel_scope_name = 'pydantic-evals'
        AND child.span_name ILIKE 'case: %'
    """  # noqa: S608

    # Ensure output directory exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    async with AsyncLogfireQueryClient(
        read_token=os.getenv("LOGFIRE_API_KEY") or "fake",
        timeout=timeout,  # type: ignore[arg-type]
    ) as client:
        json_rows = await client.query_json_rows(sql=query)

    with output_path.open("w", encoding="utf-8") as output_file:
        rows: list[dict[str, Any]] = json_rows.get("rows", [])
        if not rows:
            logger.warning("Logfire query returned no rows.")
            return

        # JSONL format:
        # 1) single header line (trace_id + aggregated cost/tokens over unique traces)
        # 2) per-case lines (verbatim per-evaluator scores + aggregate `passed`)

        trace_totals: dict[str, TraceMetrics] = {}
        trace_order: list[str] = []
        case_by_name: dict[str, CaseRow] = {}
        case_requests_sum: float = 0.0
        case_requests_count: int = 0

        for row in rows:
            trace_id = row.get("trace_id")
            if trace_id is None:
                continue
            trace_id_str = str(trace_id)

            if trace_id_str not in trace_totals:
                trace_totals[trace_id_str] = _extract_parent_metrics(row)
                trace_order.append(trace_id_str)

            case_row = _parse_case_row(row)
            case_name = case_row.case_name
            case_requests_sum += case_row.metrics.requests
            case_requests_count += 1

            # Merge by case_name: prefer a passing version if we see duplicates.
            # TODO(voorhs): is it ok?
            if case_name not in case_by_name:
                case_by_name[case_name] = case_row
            elif (not case_by_name[case_name].passed) and case_row.passed:
                logger.warning(f"Duplicate case info found: {case_name}")
                case_by_name[case_name] = case_row

        merged_cost = sum(t.cost for t in trace_totals.values())
        merged_input_tokens = sum(t.input_tokens for t in trace_totals.values())
        merged_output_tokens = sum(t.output_tokens for t in trace_totals.values())
        merged_cache_read_tokens = sum(t.cache_read_tokens for t in trace_totals.values())
        merged_requests = case_requests_sum / case_requests_count if case_requests_count else 0.0

        total_tasks = len(case_by_name)
        passed_tasks = sum(1 for c in case_by_name.values() if c.passed)

        header_trace_id = trace_order[0] if trace_order else "unknown_trace"
        header = ExperimentHeader(
            experiment_name=experiment,
            trace_id=header_trace_id,
            cost=merged_cost,
            input_tokens=merged_input_tokens,
            output_tokens=merged_output_tokens,
            cache_read_tokens=merged_cache_read_tokens,
            requests=merged_requests,
            total_tasks=total_tasks,
            passed_tasks=passed_tasks,
        )
        output_file.write(header.model_dump_json() + "\n")

        for case_name in sorted(case_by_name.keys()):
            output_file.write(case_by_name[case_name].model_dump_json() + "\n")

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


def _safe_float(v: object) -> float:
    """Best-effort float conversion for metrics/scores coming from Logfire."""
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v))
    except ValueError:
        return 0.0


def _extract_parent_metrics(row: dict[str, Any]) -> TraceMetrics:
    """Extract cost/tokens from `parent_attributes` for a single trace.

    The query joins experiment parent span (which contains aggregated metrics) with each case span,
    so these totals must be deduped per `trace_id` and not accumulated per case.
    """
    parent_attributes = row.get("parent_attributes") or {}
    logfire_meta = parent_attributes.get("logfire.experiment.metadata") or {}
    averages = logfire_meta.get("averages") or {}
    metrics = averages.get("metrics") or {}
    return TraceMetrics(
        cost=_safe_float(metrics.get("cost")),
        input_tokens=_safe_float(metrics.get("input_tokens")),
        output_tokens=_safe_float(metrics.get("output_tokens")),
        cache_read_tokens=_safe_float(metrics.get("cache_read_tokens")),
        requests=_safe_float(metrics.get("requests")),
    )


def _parse_case_row(row: dict[str, Any]) -> CaseRow:
    """Extract a `CaseRow` from a case span.

    Scores are copied (as `EvaluatorResult` models); we compute only the aggregate `passed` boolean.
    """
    child_attributes = row.get("child_attributes") or {}
    case_name = child_attributes.get("case_name") or "unknown_case"
    scores_raw_obj = child_attributes.get("scores") or {}
    metrics_raw_obj = child_attributes.get("metrics") or {}

    scores_raw: dict[str, Any] = scores_raw_obj if isinstance(scores_raw_obj, dict) else {}
    metrics_raw: dict[str, Any] = metrics_raw_obj if isinstance(metrics_raw_obj, dict) else {}

    scores = {
        str(eval_name): (
            EvaluatorResult.model_validate(eval_result)
            if isinstance(eval_result, dict)
            else EvaluatorResult(value=_safe_float(eval_result))
        )
        for eval_name, eval_result in scores_raw.items()
    }
    case_metrics = CaseMetrics.model_validate(metrics_raw)

    passed = bool(scores) and all(
        (score.value is not None) and abs(score.value - 1.0) < PASSED_EPS for score in scores.values()
    )
    return CaseRow(case_name=str(case_name), passed=passed, scores=scores, metrics=case_metrics)


if __name__ == "__main__":
    app()
