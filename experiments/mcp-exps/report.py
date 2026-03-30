"""Download experiment cases.

Load total cost, tokens and per-case results.
"""

import json
import os
from collections import defaultdict
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

        by_trace_id = _group_rows_by_trace_id(rows)

        # JSONL format:
        # 1) header line (trace_id + aggregated cost/tokens)
        # 2) per-case lines (case_name + per-evaluator scores + passed flag)
        for trace_id in sorted(by_trace_id.keys()):
            trace_rows = by_trace_id[trace_id]
            totals = _compute_totals(trace_rows)

            output_file.write(
                json.dumps(
                    {
                        "trace_id": trace_id,
                        **totals,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

            # Sort for stable output: by case_name, then evaluator keys.
            def _case_sort_key(r: dict[str, Any]) -> str:
                case_name = (r.get("child_attributes") or {}).get("case_name") or ""
                return str(case_name)

            for row in sorted(trace_rows, key=_case_sort_key):
                case_name, passed, per_eval_scores = _parse_case_row(row)

                output_file.write(
                    json.dumps(
                        {
                            "case_name": str(case_name),
                            "passed": passed,
                            "scores": per_eval_scores,
                        },
                        sort_keys=True,
                    )
                    + "\n"
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


def _group_rows_by_trace_id(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group Logfire rows by `trace_id` (one experiment run)."""
    by_trace_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        trace_id = row.get("trace_id")
        if trace_id is None:
            continue
        by_trace_id[str(trace_id)].append(row)
    return by_trace_id


def _compute_totals(trace_rows: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate cost/tokens across all case spans for a single trace."""
    totals = {
        "cost": 0.0,
        "input_tokens": 0.0,
        "output_tokens": 0.0,
        "cache_read_tokens": 0.0,
        "requests": 0.0,
    }
    for row in trace_rows:
        child_attributes = row.get("child_attributes") or {}
        metrics = child_attributes.get("metrics") or {}
        totals["cost"] += _safe_float(metrics.get("cost"))
        totals["input_tokens"] += _safe_float(metrics.get("input_tokens"))
        totals["output_tokens"] += _safe_float(metrics.get("output_tokens"))
        totals["cache_read_tokens"] += _safe_float(metrics.get("cache_read_tokens"))
        totals["requests"] += _safe_float(metrics.get("requests"))
    return totals


def _parse_case_row(row: dict[str, Any]) -> tuple[str, bool, dict[str, float]]:
    """Extract (case_name, passed, per-evaluator scores) from a case span."""
    child_attributes = row.get("child_attributes") or {}
    case_name = child_attributes.get("case_name") or "unknown_case"
    scores = child_attributes.get("scores") or {}

    per_eval_scores: dict[str, float] = {}
    for eval_name, eval_result in scores.items():
        # eval_result is typically {"name": ..., "value": float, "reason": ..., ...}
        if isinstance(eval_result, dict):
            per_eval_scores[str(eval_name)] = _safe_float(eval_result.get("value"))
        else:
            per_eval_scores[str(eval_name)] = _safe_float(eval_result)

    passed = bool(per_eval_scores) and all(abs(v - 1.0) < PASSED_EPS for v in per_eval_scores.values())
    return str(case_name), passed, per_eval_scores


if __name__ == "__main__":
    app()
