"""Train a suggester on one train split and score retrieval metrics on the test split."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from loguru import logger
from tool_suggest.orchestration import train_suggester
from tool_suggest.services.formatter import SampleFormatter
from tool_suggest.services.repository.inmemory import InMemoryRepository
from tool_suggest.services.selector import GreedySelector
from tool_suggest.services.suggester import AutoIntentSuggester, KNNSuggester

from src.agents._tool_suggest.embedding import build_embedding_resources, get_ai_config
from src.offline_eval.metrics import (
    AggregatedRetrievalMetrics,
    SampleRetrievalMetrics,
    aggregate_task_and_global,
    compute_sample_metrics,
)
from src.offline_eval.ranking import assert_suggester_supported, full_ranked_tool_ids

if TYPE_CHECKING:
    from collections.abc import Callable

    from autointent import OptimizationConfig
    from tool_suggest.models import Sample
    from tool_suggest.services.embedder.base import BaseEmbedder
    from tool_suggest.services.suggester.base import BaseSuggester

    from src.agents._tool_suggest.types import EmbBackend


@dataclass(frozen=True)
class OfflineEvalConfig:
    """Suggester and training options aligned with :mod:`run_exp` tool-suggest mode."""

    suggester: Literal["autointent", "knn"]
    emb_backend: EmbBackend
    emb_model: str
    experiment_name: str
    formatter_max_len: int
    multilabel: bool
    max_oos_fraction: float
    selection_target_size: int | None
    min_samples_per_tool: int
    knn_neighbors: int
    knn_aggregation: Literal["weighted", "uniform"]
    task_key: str


@dataclass(frozen=True)
class FoldResult:
    """Metrics for one train/test split."""

    fold_index: int
    n_train_samples: int
    n_test_samples_scored: int
    n_test_oos_skipped: int
    metrics: AggregatedRetrievalMetrics
    error: str | None = None


def _truth_tools(sample: Sample, *, multilabel: bool) -> set[str] | None:
    if sample.is_out_of_scope:
        return None
    if not sample.tools:
        return None
    if multilabel:
        return set(sample.tools)
    return {sample.tools[0]}


async def _build_and_train(
    train_samples: list[Sample],
    cfg: OfflineEvalConfig,
    *,
    embedder: BaseEmbedder,
    token_counter: Callable[[str], int] | None,
    ai_config: OptimizationConfig,
) -> BaseSuggester:
    repo = InMemoryRepository(collection_name="offline_eval")
    await repo.add_bulk(train_samples)

    formatter = SampleFormatter(max_len=cfg.formatter_max_len, token_counter=token_counter)

    if cfg.suggester == "knn":
        suggester: BaseSuggester = KNNSuggester(
            formatter,
            embedder,
            k=cfg.knn_neighbors,
            aggregation=cfg.knn_aggregation,
        )
        await train_suggester(repo, suggester, selector=None)
        return suggester

    suggester = AutoIntentSuggester(
        formatter=formatter,
        config=ai_config,
        multilabel=cfg.multilabel,
        emergency_toolset="full",
        under_represented_behavior="emergency_only",
        max_oos_fraction=cfg.max_oos_fraction,
    )
    selector = GreedySelector(
        embedder=embedder,
        formatter=formatter,
        min_samples_per_tool=cfg.min_samples_per_tool,
        min_target_size=cfg.selection_target_size,
    )
    await train_suggester(repo, suggester, selector=selector)
    return suggester


async def evaluate_fold(
    train_samples: list[Sample],
    test_samples: list[Sample],
    cfg: OfflineEvalConfig,
    *,
    topk_value: int,
    fold_index: int = 0,
) -> FoldResult:
    """Train on ``train_samples``, then score retrieval on in-domain ``test_samples`` only."""
    embedder, token_counter, emb_config = build_embedding_resources(
        emb_backend=cfg.emb_backend,
        emb_model=cfg.emb_model,
    )
    ai_config = get_ai_config(cfg.experiment_name, ai_embedder_config=emb_config)

    if not train_samples:
        return FoldResult(
            fold_index=fold_index,
            n_train_samples=0,
            n_test_samples_scored=0,
            n_test_oos_skipped=0,
            metrics=aggregate_task_and_global({}),
            error="empty train set",
        )

    try:
        suggester = await _build_and_train(
            train_samples,
            cfg,
            embedder=embedder,
            token_counter=token_counter,
            ai_config=ai_config,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("train failed for fold {}", fold_index)
        return FoldResult(
            fold_index=fold_index,
            n_train_samples=len(train_samples),
            n_test_samples_scored=0,
            n_test_oos_skipped=0,
            metrics=aggregate_task_and_global({}),
            error=str(e),
        )

    assert_suggester_supported(suggester)

    per_task: dict[str, list[SampleRetrievalMetrics]] = {}
    n_oos = 0
    n_scored = 0

    for s in test_samples:
        tkey = str(s.data.get(cfg.task_key, "")) if isinstance(s.data, dict) else "__missing_task__"
        if not tkey or tkey == "None":
            tkey = "__missing_task__"
        truth = _truth_tools(s, multilabel=cfg.multilabel)
        if truth is None:
            n_oos += 1
            continue
        try:
            ranked = await full_ranked_tool_ids(suggester, s.context)
        except Exception as e:  # noqa: BLE001
            logger.warning("suggest failed sample {}: {}", s.id, e)
            ranked = []
        m = compute_sample_metrics(ranked, truth, topk_value=topk_value)
        per_task.setdefault(tkey, []).append(m)
        n_scored += 1

    metrics = aggregate_task_and_global(per_task)
    return FoldResult(
        fold_index=fold_index,
        n_train_samples=len(train_samples),
        n_test_samples_scored=n_scored,
        n_test_oos_skipped=n_oos,
        metrics=metrics,
        error=None,
    )
