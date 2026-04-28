"""Utility `write_experiment_jsonl`."""

from pathlib import Path
from typing import Any

from loguru import logger

from .constants import PASSED_EPS
from .models import CaseMetrics, CaseRow, EvaluatorResult, ExperimentHeader, LogfireEvalFetchResult, TraceMetrics
from .parse import trace_metrics_has_usage
from .safe import safe_float


def write_experiment_jsonl(
    output_path: Path,
    experiment: str,
    fetch: LogfireEvalFetchResult,
) -> None:
    """Write JSONL header (aggregates from leaf ``chat %`` sums) and merged case rows."""
    rows = fetch.case_rows
    trace_order = fetch.trace_order
    trace_leaf_totals = fetch.leaf_totals_by_trace
    case_leaf_totals = fetch.case_leaf_totals
    case_by_name: dict[str, CaseRow] = {}
    case_requests_sum = 0.0
    case_requests_count = 0

    for row in rows:
        if row.get("trace_id") is None:
            continue

        trace_id_str = str(row.get("trace_id"))
        case_row = _parse_case_row(row)
        case_name = case_row.case_name

        leaf_for_case = case_leaf_totals.get((trace_id_str, case_name))
        if leaf_for_case is not None and trace_metrics_has_usage(leaf_for_case):
            case_row = CaseRow(
                case_name=case_row.case_name,
                passed=case_row.passed,
                scores=case_row.scores,
                metrics=_case_metrics_from_leaf_chat_totals(case_row.metrics, leaf_for_case),
            )

        case_requests_sum += case_row.metrics.requests
        case_requests_count += 1

        # Merge by case_name: prefer a passing version if we see duplicates.
        # TODO(voorhs): is it ok?
        if case_name not in case_by_name:
            case_by_name[case_name] = case_row
        elif (not case_by_name[case_name].passed) and case_row.passed:
            logger.warning(f"Duplicate case info found: {case_name}")
            case_by_name[case_name] = case_row

    merged_cost = sum(trace_leaf_totals.get(tid, TraceMetrics()).cost for tid in trace_order)
    merged_input_tokens = sum(trace_leaf_totals.get(tid, TraceMetrics()).input_tokens for tid in trace_order)
    merged_output_tokens = sum(trace_leaf_totals.get(tid, TraceMetrics()).output_tokens for tid in trace_order)
    merged_cache_read_tokens = sum(trace_leaf_totals.get(tid, TraceMetrics()).cache_read_tokens for tid in trace_order)
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

    with output_path.open("w", encoding="utf-8") as output_file:
        output_file.write(header.model_dump_json() + "\n")
        for case_name in sorted(case_by_name.keys()):
            output_file.write(case_by_name[case_name].model_dump_json() + "\n")


def _case_metrics_from_leaf_chat_totals(pydantic_metrics: CaseMetrics, leaf: TraceMetrics) -> CaseMetrics:
    """Prefer summed ``chat %`` usage under this case span (covers usage-limit aborts with empty pydantic metrics)."""
    if not trace_metrics_has_usage(leaf):
        return pydantic_metrics
    return CaseMetrics(
        cost=leaf.cost,
        input_tokens=leaf.input_tokens,
        output_tokens=leaf.output_tokens,
        cache_read_tokens=leaf.cache_read_tokens,
        requests=pydantic_metrics.requests,
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
            else EvaluatorResult(value=safe_float(eval_result))
        )
        for eval_name, eval_result in scores_raw.items()
    }
    case_metrics = CaseMetrics.model_validate(metrics_raw)

    passed = bool(scores) and all(
        (score.value is not None) and abs(score.value - 1.0) < PASSED_EPS for score in scores.values()
    )
    return CaseRow(case_name=str(case_name), passed=passed, scores=scores, metrics=case_metrics)
