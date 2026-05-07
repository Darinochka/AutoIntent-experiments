"""Offline retrieval metrics (top-1, top-k, MRR) for tool-suggest on JSONL sample repos."""

from __future__ import annotations

from .loading import group_samples_by_task, load_and_normalize
from .metrics import (
    AggregatedRetrievalMetrics,
    SampleRetrievalMetrics,
    aggregate_task_and_global,
    compute_sample_metrics,
)
from .ranking import assert_suggester_supported, full_ranked_tool_ids
from .runner import FoldResult, OfflineEvalConfig, evaluate_fold
from .splits import TaskSplit, build_cv_splits, build_holdout_split, samples_for_tasks

__all__ = [
    "AggregatedRetrievalMetrics",
    "FoldResult",
    "OfflineEvalConfig",
    "SampleRetrievalMetrics",
    "TaskSplit",
    "aggregate_task_and_global",
    "assert_suggester_supported",
    "build_cv_splits",
    "build_holdout_split",
    "compute_sample_metrics",
    "evaluate_fold",
    "full_ranked_tool_ids",
    "group_samples_by_task",
    "load_and_normalize",
    "samples_for_tasks",
]
