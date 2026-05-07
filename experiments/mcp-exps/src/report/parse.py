"""Extract metrics from an experiment trace."""

import json
from typing import Any

from .constants import PASSED_EPS
from .models import TraceMetrics
from .safe import safe_float


def metrics_from_chat_span_attributes(attributes_obj: object) -> TraceMetrics:
    """Parse token/cost from a ``chat <model>`` span.

    Pydantic AI records usage as **flat** OpenTelemetry attributes on the span
    (``gen_ai.usage.input_tokens``, ``gen_ai.usage.output_tokens``, ``operation.cost``, etc.).
    Logfire may also expose OTel meter data under ``logfire.metrics`` with a nested
    ``gen_ai.client.token.usage`` shape — we use that only when flat attributes are absent.
    """
    attrs = _coerce_mapping(attributes_obj)
    flat = _trace_metrics_from_flat_otel_attributes(attrs)
    if _has_usage_signal(flat):
        return flat
    return _trace_metrics_from_logfire_metrics_blob(attrs)


def _has_usage_signal(m: TraceMetrics) -> bool:
    return (
        m.cost > PASSED_EPS
        or m.input_tokens > PASSED_EPS
        or m.output_tokens > PASSED_EPS
        or m.cache_read_tokens > PASSED_EPS
    )


def trace_metrics_has_usage(m: TraceMetrics) -> bool:
    """True if any token/cost field is non-zero (for merging / display)."""
    return _has_usage_signal(m)


def case_name_from_case_span(span_name: str, attributes_obj: object) -> str:
    """Resolve pydantic-evals ``case_name`` from a ``case: …`` span."""
    attrs = _coerce_mapping(attributes_obj)
    raw = attrs.get("case_name")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    sn = span_name.strip()
    if sn.lower().startswith("case:"):
        return sn.split(":", 1)[1].strip()
    return "unknown_case"


def _trace_metrics_from_flat_otel_attributes(attrs: dict[str, Any]) -> TraceMetrics:
    """Read pydantic-ai / OpenTelemetry GenAI semconv attributes stored directly on the span."""
    input_tokens = safe_float(attrs.get("gen_ai.usage.input_tokens"))
    output_tokens = safe_float(attrs.get("gen_ai.usage.output_tokens"))
    cache_read_tokens = safe_float(attrs.get("gen_ai.usage.details.cache_read_tokens"))
    cost = safe_float(attrs.get("operation.cost"))
    return TraceMetrics(
        cost=cost,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        requests=0.0,
    )


def _trace_metrics_from_logfire_metrics_blob(attrs: dict[str, Any]) -> TraceMetrics:
    """Fallback: nested ``logfire.metrics`` structure (e.g. meter rollup in Logfire UI)."""
    blob = _normalize_nested_metrics_field(attrs.get("logfire.metrics"))

    usage_dict = _normalize_nested_metrics_field(blob.get("gen_ai.client.token.usage"))
    raw_details = usage_dict.get("details")
    details_list = raw_details if isinstance(raw_details, list) else []
    input_tokens, output_tokens, cache_read_tokens = _accumulate_gen_ai_usage_details(details_list)

    raw_cost = blob.get("operation.cost")
    cost_obj = _coerce_mapping(raw_cost) if isinstance(raw_cost, str) else raw_cost
    cost = _sum_operation_cost_details(cost_obj)

    return TraceMetrics(
        cost=cost,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        requests=0.0,
    )


def _normalize_nested_metrics_field(raw: object) -> dict[str, Any]:
    if isinstance(raw, str):
        return _coerce_mapping(raw)
    if isinstance(raw, dict):
        return dict(raw)
    return {}


def _accumulate_gen_ai_usage_details(details: list[object]) -> tuple[float, float, float]:
    input_tokens = 0.0
    output_tokens = 0.0
    cache_read_tokens = 0.0
    for detail in details:
        if not isinstance(detail, dict):
            continue
        detail_dict = dict(detail)
        total = safe_float(detail_dict.get("total"))
        d_attrs_obj = detail_dict.get("attributes")
        d_attrs = _coerce_mapping(d_attrs_obj)
        otel_kind = str(d_attrs.get("gen_ai.token.type") or "").lower()
        if otel_kind == "input":
            input_tokens += total
        elif otel_kind == "output":
            output_tokens += total
        elif "cache" in otel_kind:
            cache_read_tokens += total
    return input_tokens, output_tokens, cache_read_tokens


def _sum_operation_cost_details(cost_obj: object) -> float:
    block = cost_obj if isinstance(cost_obj, dict) else {}
    details = block.get("details")
    if not isinstance(details, list):
        return safe_float(block.get("total"))
    total = 0.0
    for detail in details:
        if isinstance(detail, dict):
            total += safe_float(detail.get("total"))
    return total


def _coerce_mapping(obj: object) -> dict[str, Any]:
    """Logfire may return JSON columns as dict or serialized JSON string."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if isinstance(obj, str):
        try:
            parsed = json.loads(obj)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}
