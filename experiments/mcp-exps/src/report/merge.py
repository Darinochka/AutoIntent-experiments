"""Merge ``LogfireEvalFetchResult`` bundles from several traces (e.g. CV folds)."""

from typing import Any

from .models import LogfireEvalFetchResult, TraceMetrics


def merge_logfire_eval_fetch_results(parts: list[LogfireEvalFetchResult]) -> LogfireEvalFetchResult:
    """Concatenate case rows and metric maps from multiple fetches.

    ``trace_order`` preserves first-seen order of ``trace_id`` across parts. Maps are merged with
    ``update`` (later parts win on duplicate ``trace_id`` keys; normally each part owns disjoint traces).
    """
    if not parts:
        msg = "merge_logfire_eval_fetch_results requires at least one bundle"
        raise ValueError(msg)

    case_rows: list[dict[str, Any]] = []
    trace_order: list[str] = []
    seen: set[str] = set()
    leaf_totals_by_trace: dict[str, TraceMetrics] = {}
    case_leaf_totals: dict[tuple[str, str], TraceMetrics] = {}

    for part in parts:
        case_rows.extend(part.case_rows)
        for tid in part.trace_order:
            if tid not in seen:
                seen.add(tid)
                trace_order.append(tid)
        leaf_totals_by_trace.update(part.leaf_totals_by_trace)
        case_leaf_totals.update(part.case_leaf_totals)

    return LogfireEvalFetchResult(
        case_rows=case_rows,
        trace_order=trace_order,
        leaf_totals_by_trace=leaf_totals_by_trace,
        case_leaf_totals=case_leaf_totals,
    )
