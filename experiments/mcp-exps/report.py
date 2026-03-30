"""Download experiment cases.

Load total cost, tokens and per-case results.
"""

import json
import os
from pathlib import Path
from typing import Annotated, Any

import cyclopts
from dotenv import load_dotenv
from logfire.query_client import AsyncLogfireQueryClient
from loguru import logger

load_dotenv()

app = cyclopts.App(
    # name="pipeline",
    # help="ETL pipeline for offers reranker: load, parse, and split data",
)


PASSED_EPS = 1e-9


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

        trace_totals: dict[str, dict[str, float]] = {}
        trace_order: list[str] = []
        case_by_name: dict[str, tuple[bool, dict[str, Any]]] = {}

        for row in rows:
            trace_id = row.get("trace_id")
            if trace_id is None:
                continue
            trace_id_str = str(trace_id)

            if trace_id_str not in trace_totals:
                trace_totals[trace_id_str] = _extract_parent_metrics(row)
                trace_order.append(trace_id_str)

            case_name, passed, scores = _parse_case_row(row)

            # Merge by case_name: prefer a passing version if we see duplicates.
            # TODO(voorhs): is it ok?
            if case_name not in case_by_name:
                case_by_name[case_name] = (passed, scores)
            else:
                logger.warning(f"Duplicate case info found: {case_name}")
                existing_passed, _existing_scores = case_by_name[case_name]
                if (not existing_passed) and passed:
                    case_by_name[case_name] = (passed, scores)

        merged_totals = {
            "cost": 0.0,
            "input_tokens": 0.0,
            "output_tokens": 0.0,
            "cache_read_tokens": 0.0,
        }
        for totals in trace_totals.values():
            merged_totals["cost"] += totals.get("cost", 0.0)
            merged_totals["input_tokens"] += totals.get("input_tokens", 0.0)
            merged_totals["output_tokens"] += totals.get("output_tokens", 0.0)
            merged_totals["cache_read_tokens"] += totals.get("cache_read_tokens", 0.0)

        header_trace_id = trace_order[0] if trace_order else "unknown_trace"
        output_file.write(json.dumps({"trace_id": header_trace_id, **merged_totals}, sort_keys=True) + "\n")

        for case_name in sorted(case_by_name.keys()):
            passed, scores = case_by_name[case_name]
            output_file.write(
                json.dumps({"case_name": case_name, "passed": passed, "scores": scores}, sort_keys=True) + "\n"
            )

    logger.success("Done!")


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


def _extract_parent_metrics(row: dict[str, Any]) -> dict[str, float]:
    """Extract cost/tokens from `parent_attributes` for a single trace.

    The query joins experiment parent span (which contains aggregated metrics) with each case span,
    so these totals must be deduped per `trace_id` and not accumulated per case.
    """
    parent_attributes = row.get("parent_attributes") or {}
    logfire_meta = parent_attributes.get("logfire.experiment.metadata") or {}
    averages = logfire_meta.get("averages") or {}
    metrics = averages.get("metrics") or {}
    return {
        "cost": _safe_float(metrics.get("cost")),
        "input_tokens": _safe_float(metrics.get("input_tokens")),
        "output_tokens": _safe_float(metrics.get("output_tokens")),
        "cache_read_tokens": _safe_float(metrics.get("cache_read_tokens")),
        "requests": _safe_float(metrics.get("requests")),
    }


def _parse_case_row(row: dict[str, Any]) -> tuple[str, bool, dict[str, Any]]:
    """Extract (case_name, passed, per-evaluator scores) from a case span.

    Scores are copied verbatim; we only compute the aggregate `passed` boolean.
    """
    child_attributes = row.get("child_attributes") or {}
    case_name = child_attributes.get("case_name") or "unknown_case"
    scores: dict[str, Any] = child_attributes.get("scores") or {}

    def _is_green(v: object) -> bool:
        if isinstance(v, dict) and "value" in v:
            return abs(_safe_float(v.get("value")) - 1.0) < PASSED_EPS
        return abs(_safe_float(v) - 1.0) < PASSED_EPS

    passed = bool(scores) and all(_is_green(v) for v in scores.values())
    return str(case_name), passed, scores


if __name__ == "__main__":
    app()
