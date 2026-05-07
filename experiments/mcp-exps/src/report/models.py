"""Models/DTOs for report creation."""

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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


@dataclass(frozen=True, slots=True)
class LogfireEvalFetchResult:
    """Bundle returned by ``query``.

    ``case_rows``: joined SQL rows (``trace_id``, ``child_attributes``). ``trace_order``: distinct
    trace ids in first-seen order. ``leaf_totals_by_trace``: sum of all ``chat %`` spans in the trace.

    ``case_leaf_totals``: token/cost summed only over chat spans whose ancestor chain reaches that
    case's ``case: …`` span (fixes per-task metrics when pydantic-evals task metrics are empty, e.g.
    usage limit before evaluators ran).
    """

    case_rows: list[dict[str, Any]]
    trace_order: list[str]
    leaf_totals_by_trace: dict[str, TraceMetrics]
    case_leaf_totals: dict[tuple[str, str], TraceMetrics]
