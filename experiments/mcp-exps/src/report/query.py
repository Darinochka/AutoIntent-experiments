"""SQL query to retrieve Logfire span with metrics."""

from collections import defaultdict
from typing import Any

from logfire.query_client import AsyncLogfireQueryClient
from loguru import logger

from .constants import PASSED_EPS
from .models import LogfireEvalFetchResult, TraceMetrics
from .parse import metrics_from_chat_span_attributes


async def query(
    client: AsyncLogfireQueryClient,
    experiment: str,
) -> LogfireEvalFetchResult | None:
    """Run the evaluate/case query plus per-trace sums over leaf ``chat %`` spans."""
    evaluate_cases_sql = f"""
    SELECT
        parent.trace_id,
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

    json_rows = await client.query_json_rows(sql=evaluate_cases_sql)
    rows_raw = json_rows.get("rows", [])
    if not isinstance(rows_raw, list) or not rows_raw:
        return None
    rows: list[dict[str, Any]] = [r for r in rows_raw if isinstance(r, dict)]

    trace_order: list[str] = []
    seen: set[str] = set()
    for row in rows:
        trace_id = row.get("trace_id")
        if trace_id is None:
            continue
        trace_id_str = str(trace_id)
        if trace_id_str not in seen:
            seen.add(trace_id_str)
            trace_order.append(trace_id_str)

    leaf_by_trace, chat_counts = await _query_leaf_chat_metrics(client, trace_order)
    for tid in trace_order:
        _warn_leaf_rollup_if_empty(tid, leaf_by_trace.get(tid, TraceMetrics()), chat_counts.get(tid, 0))

    return LogfireEvalFetchResult(
        case_rows=rows,
        trace_order=trace_order,
        leaf_totals_by_trace=leaf_by_trace,
    )


async def _query_leaf_chat_metrics(
    client: AsyncLogfireQueryClient,
    trace_ids: list[str],
) -> tuple[dict[str, TraceMetrics], dict[str, int]]:
    """Sum token/cost metrics from leaf LLM spans named like ``chat <model>`` within each trace."""
    per_trace: dict[str, TraceMetrics] = {}
    counts: dict[str, int] = defaultdict(int)

    if not trace_ids:
        return {}, {}

    in_list = _sql_in_trace_ids(trace_ids)
    sql = f"""
    SELECT trace_id, span_name, attributes
    FROM records
    WHERE trace_id IN ({in_list})
      AND span_name ILIKE 'chat %'
    """  # noqa: S608

    json_rows = await client.query_json_rows(sql=sql)
    for row in json_rows.get("rows", []):
        tid_raw = row.get("trace_id")
        if tid_raw is None:
            continue
        tid = str(tid_raw)
        counts[tid] += 1
        partial = metrics_from_chat_span_attributes(row.get("attributes"))
        cur = per_trace.get(tid, TraceMetrics())
        per_trace[tid] = _add_trace_metrics(cur, partial)

    return per_trace, dict(counts)


def _sql_in_trace_ids(trace_ids: list[str]) -> str:
    parts: list[str] = []
    for tid in trace_ids:
        if any(ch in tid for ch in ("'", ";")):
            msg = f"invalid trace_id for SQL literal: {tid!r}"
            raise ValueError(msg)
        parts.append(f"'{tid}'")
    return ", ".join(parts)


def _add_trace_metrics(a: TraceMetrics, b: TraceMetrics) -> TraceMetrics:
    return TraceMetrics(
        cost=a.cost + b.cost,
        input_tokens=a.input_tokens + b.input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        cache_read_tokens=a.cache_read_tokens + b.cache_read_tokens,
        requests=a.requests + b.requests,
    )


def _warn_leaf_rollup_if_empty(trace_id: str, leaf: TraceMetrics, n_chat: int) -> None:
    """Warn when header token/cost for this trace stay zero after leaf aggregation."""
    has_usage = (
        leaf.cost > PASSED_EPS
        or leaf.input_tokens > PASSED_EPS
        or leaf.output_tokens > PASSED_EPS
        or leaf.cache_read_tokens > PASSED_EPS
    )
    if n_chat == 0:
        logger.warning(
            f"{trace_id}: no spans matching ILIKE 'chat %'; header token/cost for this trace are 0.",
        )
        return
    if not has_usage:
        logger.warning(
            f"{trace_id}: found {n_chat} `chat %` span(s) but no token/cost in `logfire.metrics`; "
            "header totals for this trace are 0.",
        )
