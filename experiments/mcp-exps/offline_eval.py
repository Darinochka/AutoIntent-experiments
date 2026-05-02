"""CLI: offline top-1 / top-k / MRR for tool-suggest on a JSONL sample repository."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path  # noqa: TC003
from statistics import mean
from typing import Annotated, Any, Literal

import cyclopts
from cyclopts import Parameter
from loguru import logger

from src.agents import EmbBackend  # noqa: TC001
from src.offline_eval.loading import group_samples_by_task, load_and_normalize
from src.offline_eval.runner import FoldResult, OfflineEvalConfig, evaluate_fold
from src.offline_eval.splits import TaskSplit, build_cv_splits, build_holdout_split, samples_for_tasks

app = cyclopts.App(
    name="offline-eval",
    help="Offline top-1 / top-k / MRR for tool-suggest (AutoIntent or KNN) on a JSONL sample repository.",
)


@dataclass(frozen=True, kw_only=True)
class OfflineEvalCliArgs:
    """Command-line arguments for offline retrieval evaluation."""

    repo: Annotated[Path, Parameter(help="Path to tool-suggest JSONL repository.")]
    split: Annotated[
        Literal["cv", "ho"],
        Parameter(help="Task-level split: k-fold CV or single holdout."),
    ] = "cv"
    cv_folds: Annotated[int, Parameter(help="Number of CV folds (split mode=cv).")] = 5
    test_size: Annotated[
        float,
        Parameter(help="Fraction of tasks in the test set (split mode=ho)."),
    ] = 0.2
    random_state: Annotated[int, Parameter(help="Random seed for splits.")] = 42
    suggester: Annotated[
        Literal["autointent", "knn"],
        Parameter(help="autointent: same stack as run_exp ts; knn: fast debug baseline."),
    ] = "autointent"
    emb_backend: Annotated[EmbBackend, Parameter(help="Embedder backend (matches run_exp).")] = "openai"
    emb_model: Annotated[str, Parameter(help="Embedding model name.")] = "text-embedding-3-small"
    experiment_name: Annotated[
        str,
        Parameter(help="AutoIntent run name (logging / checkpoints)."),
    ] = "offline-eval"
    formatter_max_len: Annotated[
        int,
        Parameter(help="SampleFormatter max length (run_exp: formatter-max-len)."),
    ] = 1000
    multilabel: Annotated[bool, Parameter(help="AutoIntent multilabel vs multiclass.")] = False
    max_oos: Annotated[
        float,
        Parameter(help="Max OOS fraction for AutoIntent training (run_exp: --max-oos)."),
    ] = 0.2
    selection_target_size: Annotated[
        int | None,
        Parameter(help="GreedySelector min target size (run_exp: --selection-target-size)."),
    ] = 100
    min_samples_per_tool: Annotated[
        int,
        Parameter(help="GreedySelector min samples per tool (run_exp: --tool-samples)."),
    ] = 4
    knn_neighbors: Annotated[int, Parameter(help="KNN k (neighbor count) when suggester=knn.")] = 5
    knn_aggregation: Annotated[
        Literal["weighted", "uniform"],
        Parameter(help="KNN vote aggregation when suggester=knn."),
    ] = "weighted"
    topk_metric: Annotated[int, Parameter(help="k for top-k hit rate (e.g. 5).")] = 5
    task_key: Annotated[
        str,
        Parameter(help="``sample.data`` key for task / case id (e.g. case_name)."),
    ] = "case_name"
    json_out: Annotated[Path | None, Parameter(help="Optional path to write full result JSON.")] = None

    def to_offline_eval_config(self, *, experiment_name: str) -> OfflineEvalConfig:
        """Build :class:`OfflineEvalConfig` for :func:`evaluate_fold` (per-fold name may differ)."""
        return OfflineEvalConfig(
            suggester=self.suggester,
            emb_backend=self.emb_backend,
            emb_model=self.emb_model,
            experiment_name=experiment_name,
            formatter_max_len=self.formatter_max_len,
            multilabel=self.multilabel,
            max_oos_fraction=self.max_oos,
            selection_target_size=self.selection_target_size,
            min_samples_per_tool=self.min_samples_per_tool,
            knn_neighbors=self.knn_neighbors,
            knn_aggregation=self.knn_aggregation,
            task_key=self.task_key,
        )


def _fold_to_jsonable(fr: FoldResult, *, topk: int) -> dict[str, Any]:
    m = fr.metrics
    return {
        "fold_index": fr.fold_index,
        "n_train_samples": fr.n_train_samples,
        "n_test_samples_scored": fr.n_test_samples_scored,
        "n_test_skipped_no_eval": fr.n_test_oos_skipped,
        "error": fr.error,
        "micro": {
            "top1": m.micro_top1,
            f"topk@{topk}": m.micro_topk,
            "mrr": m.micro_mrr,
        },
        "macro_over_tasks": {
            "top1": m.macro_top1,
            f"topk@{topk}": m.macro_topk,
            "mrr": m.macro_mrr,
            "std_top1": m.macro_top1_std,
            f"std_topk@{topk}": m.macro_topk_std,
            "std_mrr": m.macro_mrr_std,
        },
        "n_samples": m.n_samples,
        "n_tasks": m.n_tasks,
    }


def run_offline_eval(a: OfflineEvalCliArgs) -> None:
    """Evaluate retrieval metrics with task-level CV or holdout; retrain on each train split."""
    samples = load_and_normalize(a.repo)
    if not samples:
        logger.error("No samples loaded from {}", a.repo)
        return

    task_to_samples = group_samples_by_task(samples, task_key=a.task_key)
    n_tasks = len(task_to_samples)
    logger.info("Loaded {} samples in {} task groups (key={})", len(samples), n_tasks, a.task_key)

    base_cfg = a.to_offline_eval_config(experiment_name=a.experiment_name)

    fold_list: list[TaskSplit]
    if a.split == "ho":
        fold_list = [build_holdout_split(task_to_samples, test_size=a.test_size, random_state=a.random_state)]
    else:
        fold_list = build_cv_splits(
            task_to_samples,
            n_splits=a.cv_folds,
            random_state=a.random_state,
        )

    results: list[FoldResult] = []

    async def _run_all() -> None:
        for i, fsp in enumerate(fold_list):
            train_s = samples_for_tasks(task_to_samples, fsp.train_task_ids)
            test_s = samples_for_tasks(task_to_samples, fsp.test_task_ids)
            name = f"{a.experiment_name}-fold{i}" if len(fold_list) > 1 else a.experiment_name
            cfg = a.to_offline_eval_config(experiment_name=name)
            r = await evaluate_fold(
                train_s,
                test_s,
                cfg,
                topk_value=a.topk_metric,
                fold_index=i,
            )
            results.append(r)

    asyncio.run(_run_all())

    valid = [r for r in results if r.error is None and r.n_test_samples_scored > 0]
    if not valid:
        logger.error("No successful folds; fold errors: {}", [r.error for r in results])
    else:
        for mname, field in [
            ("micro_top1", "micro_top1"),
            ("micro_topk", "micro_topk"),
            ("micro_mrr", "micro_mrr"),
            ("macro_top1", "macro_top1"),
            ("macro_topk", "macro_topk"),
            ("macro_mrr", "macro_mrr"),
        ]:
            vals = [getattr(r.metrics, field) for r in valid]
            v: float = mean(vals)
            logger.info(
                "mean over folds: {} = {:.4f} (topk_metric k={}, split={})",
                mname,
                v,
                a.topk_metric,
                a.split,
            )

    out_obj: dict[str, Any] = {
        "repo": str(a.repo.resolve()),
        "split": a.split,
        "cv_folds": a.cv_folds if a.split == "cv" else None,
        "test_size": a.test_size if a.split == "ho" else None,
        "random_state": a.random_state,
        "topk_metric": a.topk_metric,
        "config": asdict(base_cfg),
        "folds": [_fold_to_jsonable(fr, topk=a.topk_metric) for fr in results],
    }
    if a.json_out is not None:
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(json.dumps(out_obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        logger.info("Wrote {}", a.json_out)


@app.default
def main(args: Annotated[OfflineEvalCliArgs, Parameter(name="*")]) -> None:
    """CLI entry: parse into :class:`OfflineEvalCliArgs` and run."""
    run_offline_eval(args)


if __name__ == "__main__":
    app()
