"""Resolve experiment + trace from a Logfire public trace URL."""

import re
from urllib.parse import parse_qs, urlparse

from logfire.query_client import AsyncLogfireQueryClient

_EVALUATE_PREFIX = "evaluate "


def parse_span_id_from_public_trace_url(url: str) -> str:
    """Extract ``spanId`` from ``…?spanId=…`` and validate OTEL 64-bit hex form."""
    parsed = urlparse(url.strip())
    query = parse_qs(parsed.query)
    vals = query.get("spanId") or query.get("span_id")
    raw = vals[0] if vals else None
    if raw is None or not str(raw).strip():
        msg = "public trace URL must include a non-empty spanId query parameter"
        raise ValueError(msg)
    return normalize_hex_span_id(str(raw))


def normalize_hex_span_id(span_id: str) -> str:
    """Return lowercase 16-char hex ``span_id`` or raise."""
    s = span_id.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{16}", s):
        msg = f"span_id must be 16 hexadecimal characters, got {span_id!r}"
        raise ValueError(msg)
    return s


def experiment_name_from_evaluate_message(message: str) -> str:
    """Strip pydantic-evals ``evaluate <name>`` prefix."""
    m = message.strip()
    if not m.lower().startswith(_EVALUATE_PREFIX):
        msg = f"expected message starting with {_EVALUATE_PREFIX!r}, got {message!r}"
        raise ValueError(msg)
    name = m[len(_EVALUATE_PREFIX) :].strip()
    if not name:
        msg = "empty experiment name after evaluate prefix"
        raise ValueError(msg)
    return name


async def resolve_experiment_for_span(
    client: AsyncLogfireQueryClient,
    span_id: str,
) -> tuple[str, str]:
    """Return ``(trace_id, experiment_name)`` for an anchor span id (e.g. root span from public URL).

    Looks up the pydantic-evals parent span whose ``message`` is ``evaluate <experiment>`` in the same trace.
    """
    sid = normalize_hex_span_id(span_id)

    sql = f"""
    SELECT
        anchor.trace_id AS trace_id,
        ev.message AS evaluate_message
    FROM records AS anchor
    JOIN records AS ev
        ON ev.trace_id = anchor.trace_id
    WHERE anchor.span_id = '{sid}'
      AND ev.otel_scope_name = 'pydantic-evals'
      AND ev.message LIKE 'evaluate %'
    ORDER BY ev.start_timestamp ASC
    LIMIT 1
    """  # noqa: S608

    json_rows = await client.query_json_rows(sql=sql)
    rows_raw = json_rows.get("rows", [])
    if not isinstance(rows_raw, list) or not rows_raw:
        msg = (
            f"No pydantic-evals evaluate span found for span_id={sid!r}. "
            "Check spanId (16 hex chars) and Logfire read token scope."
        )
        raise ValueError(msg)

    row = rows_raw[0]
    if not isinstance(row, dict):
        msg = "unexpected Logfire row shape"
        raise TypeError(msg)

    tid_raw = row.get("trace_id")
    msg_raw = row.get("evaluate_message")
    if tid_raw is None:
        msg = "trace_id missing from Logfire resolution row"
        raise ValueError(msg)
    if not isinstance(msg_raw, str):
        msg = "evaluate_message missing or not a string"
        raise TypeError(msg)

    trace_id = str(tid_raw)
    experiment = experiment_name_from_evaluate_message(msg_raw)
    return trace_id, experiment


def trace_prefix(trace_id: str, *, n: int = 8) -> str:
    """Short prefix for filenames; safe ASCII hex slice."""
    t = trace_id.strip().lower()
    return t[:n] if len(t) >= n else t
