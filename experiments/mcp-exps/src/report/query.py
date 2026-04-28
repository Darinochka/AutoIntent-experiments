"""SQL query to retrieve Logfire span with metrics."""

from collections import defaultdict
from typing import Any

from logfire.query_client import AsyncLogfireQueryClient
from loguru import logger

from .constants import PASSED_EPS
from .models import LogfireEvalFetchResult, TraceMetrics
from .parse import case_name_from_case_span, metrics_from_chat_span_attributes


async def query(
    client: AsyncLogfireQueryClient,
    experiment: str,
) -> LogfireEvalFetchResult | None:
    """Run the evaluate/case query plus chat-span metrics (trace totals + per-case attribution)."""
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

    leaf_by_trace, case_leaf_totals, chat_counts = await _partition_chat_metrics(client, trace_order)
    for tid in trace_order:
        _warn_leaf_rollup_if_empty(tid, leaf_by_trace.get(tid, TraceMetrics()), chat_counts.get(tid, 0))

    return LogfireEvalFetchResult(
        case_rows=rows,
        trace_order=trace_order,
        leaf_totals_by_trace=leaf_by_trace,
        case_leaf_totals=case_leaf_totals,
    )


async def _partition_chat_metrics(
    client: AsyncLogfireQueryClient,
    trace_ids: list[str],
) -> tuple[dict[str, TraceMetrics], dict[tuple[str, str], TraceMetrics], dict[str, int]]:
    """Load all spans for traces, sum ``chat %`` metrics per trace and per ``case: …`` subtree."""
    if not trace_ids:
        return {}, {}, {}

    in_list = _sql_in_trace_ids(trace_ids)
    sql = f"""
    SELECT trace_id, span_id, parent_span_id, span_name, attributes
    FROM records
    WHERE trace_id IN ({in_list})
    """  # noqa: S608

    json_rows = await client.query_json_rows(sql=sql)
    span_rows = json_rows.get("rows", [])
    if not isinstance(span_rows, list):
        return {}, {}, {}

    return _partition_chat_metrics_from_span_rows(span_rows)


def _partition_chat_metrics_from_span_rows(
    span_rows: list[Any],
) -> tuple[dict[str, TraceMetrics], dict[tuple[str, str], TraceMetrics], dict[str, int]]:
    per_trace: dict[str, TraceMetrics] = {}
    case_totals: dict[tuple[str, str], TraceMetrics] = {}
    counts: dict[str, int] = defaultdict(int)

    by_id: dict[str, dict[str, Any]] = {}
    for row in span_rows:
        if not isinstance(row, dict):
            continue
        sid_raw = row.get("span_id")
        if sid_raw is None:
            continue
        by_id[str(sid_raw)] = row

    for row in span_rows:
        if isinstance(row, dict):
            _accumulate_chat_span_into_totals(row, by_id, per_trace, case_totals, counts)

    return per_trace, dict(case_totals), dict(counts)


def _accumulate_chat_span_into_totals(
    row: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
    per_trace: dict[str, TraceMetrics],
    case_totals: dict[tuple[str, str], TraceMetrics],
    counts: dict[str, int],
) -> None:
    span_name = str(row.get("span_name") or "")
    if not span_name.lower().startswith("chat "):
        return
    tid_raw = row.get("trace_id")
    if tid_raw is None:
        return
    tid = str(tid_raw)
    sid_raw = row.get("span_id")
    if sid_raw is None:
        return
    chat_sid = str(sid_raw)

    partial = metrics_from_chat_span_attributes(row.get("attributes"))
    cur_t = per_trace.get(tid, TraceMetrics())
    per_trace[tid] = _add_trace_metrics(cur_t, partial)
    counts[tid] += 1

    case_sid = _walk_up_to_case_span_id(chat_sid, by_id)
    if case_sid is None:
        return
    case_row = by_id.get(case_sid)
    if not case_row:
        return
    cn = case_name_from_case_span(str(case_row.get("span_name") or ""), case_row.get("attributes"))
    key = (tid, cn)
    cur_c = case_totals.get(key, TraceMetrics())
    case_totals[key] = _add_trace_metrics(cur_c, partial)


def _walk_up_to_case_span_id(start_span_id: str, by_id: dict[str, dict[str, Any]]) -> str | None:
    """Follow ``parent_span_id`` until we hit a pydantic-evals ``case: …`` span."""
    sid: str | None = start_span_id
    visited: set[str] = set()
    while sid and sid not in visited:
        visited.add(sid)
        row = by_id.get(sid)
        if not row:
            return None
        name = str(row.get("span_name") or "")
        if name.lower().startswith("case:"):
            return sid
        pid = row.get("parent_span_id")
        sid = str(pid) if pid is not None else None
    return None


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
            f"{trace_id}: found {n_chat} `chat %` span(s) but could not parse token/cost from span attributes; "
            "header totals for this trace are 0.",
        )
