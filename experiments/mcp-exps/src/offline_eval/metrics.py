"""Top-1, top-k, and MRR from a ranked tool list and a set of ground-truth tool names."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class SampleRetrievalMetrics:
    """Retrieval metrics for one (context, label set) pair."""

    top1: float
    topk: float
    mrr: float


def _first_relevant_rank(ranked_tool_ids: Sequence[str], truth: set[str]) -> int | None:
    """1-based rank of the first relevant tool, or ``None`` if none appear."""
    for i, tid in enumerate(ranked_tool_ids):
        if tid in truth:
            return i + 1
    return None


def compute_sample_metrics(
    ranked_tool_ids: Sequence[str],
    truth_tools: set[str],
    *,
    topk_value: int,
) -> SampleRetrievalMetrics:
    """Compute top-1 / top-k / MRR for one sample.

    * **top-1**: ``1.0`` if the first ranked tool is in ``truth_tools``, else ``0.0``. If the model
      returns an empty ranking, ``0.0``.
    * **top-k**: ``1.0`` if any of the first ``topk_value`` tools is in ``truth_tools``, else ``0.0``.
    * **MRR**: ``1/r`` where ``r`` is the 1-based rank of the first relevant tool; ``0.0`` if none.
    """
    if not truth_tools:
        return SampleRetrievalMetrics(top1=0.0, topk=0.0, mrr=0.0)
    if not ranked_tool_ids:
        return SampleRetrievalMetrics(top1=0.0, topk=0.0, mrr=0.0)

    top1 = 1.0 if ranked_tool_ids[0] in truth_tools else 0.0
    k = min(topk_value, len(ranked_tool_ids))
    topk = 1.0 if any(ranked_tool_ids[i] in truth_tools for i in range(k)) else 0.0
    r = _first_relevant_rank(ranked_tool_ids, truth_tools)
    mrr = (1.0 / r) if r is not None else 0.0
    return SampleRetrievalMetrics(top1=top1, topk=topk, mrr=mrr)


@dataclass(frozen=True)
class AggregatedRetrievalMetrics:
    """Micro (all samples) and macro (mean over tasks) aggregates."""

    n_samples: int
    n_tasks: int
    micro_top1: float
    micro_topk: float
    micro_mrr: float
    macro_top1: float
    macro_topk: float
    macro_mrr: float
    macro_top1_std: float
    macro_topk_std: float
    macro_mrr_std: float


def aggregate_task_and_global(
    per_task: dict[str, list[SampleRetrievalMetrics]],
) -> AggregatedRetrievalMetrics:
    """Micro averages over all samples; macro = mean of per-task means."""
    all_m: list[SampleRetrievalMetrics] = [m for rows in per_task.values() for m in rows]
    n = len(all_m)
    if n == 0:
        return AggregatedRetrievalMetrics(
            n_samples=0,
            n_tasks=0,
            micro_top1=0.0,
            micro_topk=0.0,
            micro_mrr=0.0,
            macro_top1=0.0,
            macro_topk=0.0,
            macro_mrr=0.0,
            macro_top1_std=0.0,
            macro_topk_std=0.0,
            macro_mrr_std=0.0,
        )

    micro_top1 = mean(m.top1 for m in all_m)
    micro_topk = mean(m.topk for m in all_m)
    micro_mrr = mean(m.mrr for m in all_m)

    task_means_top1: list[float] = []
    task_means_topk: list[float] = []
    task_means_mrr: list[float] = []
    for _tid, rows in sorted(per_task.items()):
        if not rows:
            continue
        task_means_top1.append(mean(r.top1 for r in rows))
        task_means_topk.append(mean(r.topk for r in rows))
        task_means_mrr.append(mean(r.mrr for r in rows))

    nt = len(task_means_top1)
    macro_top1 = mean(task_means_top1) if task_means_top1 else 0.0
    macro_topk = mean(task_means_topk) if task_means_topk else 0.0
    macro_mrr = mean(task_means_mrr) if task_means_mrr else 0.0
    macro_top1_std = pstdev(task_means_top1) if len(task_means_top1) > 1 else 0.0
    macro_topk_std = pstdev(task_means_topk) if len(task_means_topk) > 1 else 0.0
    macro_mrr_std = pstdev(task_means_mrr) if len(task_means_mrr) > 1 else 0.0

    return AggregatedRetrievalMetrics(
        n_samples=n,
        n_tasks=nt,
        micro_top1=micro_top1,
        micro_topk=micro_topk,
        micro_mrr=micro_mrr,
        macro_top1=macro_top1,
        macro_topk=macro_topk,
        macro_mrr=macro_mrr,
        macro_top1_std=macro_top1_std,
        macro_topk_std=macro_topk_std,
        macro_mrr_std=macro_mrr_std,
    )
