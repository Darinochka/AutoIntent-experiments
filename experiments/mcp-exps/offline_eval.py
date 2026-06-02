"""CLI: offline top-1 / top-k / MRR for tool-suggest on a JSONL sample repository."""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict, dataclass, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Annotated, Any, Literal, TextIO

if TYPE_CHECKING:
    from collections.abc import Callable

import cyclopts
from cyclopts import Parameter
from loguru import logger

from src.agents import EmbBackend  # noqa: TC001
from src.offline_eval.loading import group_samples_by_task, load_and_normalize
from src.offline_eval.metrics import AggregatedRetrievalMetrics
from src.offline_eval.runner import FoldResult, OfflineEvalConfig, evaluate_fold
from src.offline_eval.splits import TaskSplit, build_cv_splits, build_holdout_split, samples_for_tasks

_EMPTY_AGG_METRICS = AggregatedRetrievalMetrics(
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
    balanced_accuracy=0.0,
    class_entropy_normalized=0.0,
    mean_average_precision=0.0,
)


@dataclass(frozen=True, kw_only=True)
class OfflineEvalCommonArgs:
    """Shared hyperparameters for offline retrieval (single repo or batch)."""

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
    emb_backend: Annotated[EmbBackend, Parameter(help="Embedder backend (matches run_exp).")] = "openai"
    emb_model: Annotated[str, Parameter(help="Embedding model name.")] = "text-embedding-3-small"
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
    train_on_passed_only: Annotated[
        bool,
        Parameter(
            help=(
                "Train only on samples whose case passed (``data['passed']`` is True). Test set is unfiltered. "
                "Folds whose train split has no passing samples are skipped (recorded with an error). "
                "Requires every loaded sample to have ``data['passed']`` — raises otherwise."
            ),
        ),
    ] = False


_OFFLINE_COMMON_STAR_DEFAULT = OfflineEvalCommonArgs()

app = cyclopts.App(
    name="offline-eval",
    help="Offline top-1 / top-k / MRR for tool-suggest (AutoIntent or KNN) on a JSONL sample repository.",
)


def _common_fields_dict(c: OfflineEvalCommonArgs) -> dict[str, Any]:
    return {f.name: getattr(c, f.name) for f in fields(c)}


@dataclass(frozen=True, kw_only=True)
class OfflineEvalCliArgs(OfflineEvalCommonArgs):
    """Command-line arguments for offline retrieval evaluation."""

    repo: Annotated[Path, Parameter(help="Path to tool-suggest JSONL repository.")]
    suggester: Annotated[
        Literal["autointent", "knn"],
        Parameter(help="autointent: same stack as run_exp ts; knn: fast debug baseline."),
    ] = "autointent"
    experiment_name: Annotated[
        str,
        Parameter(help="AutoIntent run name (logging / checkpoints)."),
    ] = "offline-eval"
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
        "balanced_accuracy": m.balanced_accuracy,
        "class_entropy_normalized": m.class_entropy_normalized,
        "mean_average_precision": m.mean_average_precision,
        "n_samples": m.n_samples,
        "n_tasks": m.n_tasks,
    }


def _append_jsonl(f: TextIO, obj: dict[str, Any]) -> None:
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    f.flush()


def _make_fold_row_appender(
    out_f: TextIO,
    *,
    repo_resolved: str,
    suggester: str,
    experiment_name: str,
    split: Literal["cv", "ho"],
    cv_folds: int | None,
    test_size: float | None,
    random_state: int,
    topk_metric: int,
    task_key: str,
) -> Callable[[FoldResult], None]:
    """Append one JSONL record per fold; closure captures scalars only (ruff B023)."""

    def on_fold(fr: FoldResult) -> None:
        _append_jsonl(
            out_f,
            {
                "record_type": "fold",
                "timestamp": datetime.now(UTC).isoformat(),
                "repo": repo_resolved,
                "suggester": suggester,
                "experiment_name": experiment_name,
                "split": split,
                "cv_folds": cv_folds,
                "test_size": test_size,
                "random_state": random_state,
                "topk_metric": topk_metric,
                "task_key": task_key,
                "fold": _fold_to_jsonable(fr, topk=topk_metric),
            },
        )

    return on_fold


def _prepare_folds(a: OfflineEvalCliArgs) -> tuple[dict[str, list[Any]], list[TaskSplit], int] | None:
    samples = load_and_normalize(a.repo)
    if not samples:
        return None
    if a.train_on_passed_only:
        missing = sum(1 for s in samples if not (isinstance(s.data, dict) and "passed" in s.data))
        if missing:
            msg = (
                f"--train-on-passed-only requires every sample to carry data['passed']; "
                f"{missing}/{len(samples)} are missing it in {a.repo}. Re-export the repo with the updated samples.py."
            )
            raise ValueError(msg)
    task_to_samples = group_samples_by_task(samples, task_key=a.task_key)
    if a.split == "ho":
        fold_list = [build_holdout_split(task_to_samples, test_size=a.test_size, random_state=a.random_state)]
    else:
        fold_list = build_cv_splits(
            task_to_samples,
            n_splits=a.cv_folds,
            random_state=a.random_state,
        )
    return task_to_samples, fold_list, len(samples)


async def _run_folds_async(
    a: OfflineEvalCliArgs,
    task_to_samples: dict[str, list[Any]],
    fold_list: list[TaskSplit],
    *,
    on_fold: Callable[[FoldResult], None] | None = None,
) -> list[FoldResult]:
    results: list[FoldResult] = []
    for i, fsp in enumerate(fold_list):
        train_s = samples_for_tasks(task_to_samples, fsp.train_task_ids)
        test_s = samples_for_tasks(task_to_samples, fsp.test_task_ids)
        if a.train_on_passed_only:
            train_s = [s for s in train_s if isinstance(s.data, dict) and s.data.get("passed") is True]
            if not train_s:
                logger.warning(
                    "Fold {}: no passing samples in train split ({} tasks); skipping fold.",
                    i,
                    len(fsp.train_task_ids),
                )
                r = FoldResult(
                    fold_index=i,
                    n_train_samples=0,
                    n_test_samples_scored=0,
                    n_test_oos_skipped=0,
                    metrics=_EMPTY_AGG_METRICS,
                    error="skipped: no passing train samples (train_on_passed_only)",
                )
                results.append(r)
                if on_fold is not None:
                    on_fold(r)
                continue
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
        if on_fold is not None:
            on_fold(r)
    return results


_FOLD_MEAN_FIELDS: tuple[tuple[str, str], ...] = (
    ("micro_top1", "micro_top1"),
    ("micro_topk", "micro_topk"),
    ("micro_mrr", "micro_mrr"),
    ("macro_top1", "macro_top1"),
    ("macro_topk", "macro_topk"),
    ("macro_mrr", "macro_mrr"),
    ("balanced_accuracy", "balanced_accuracy"),
    ("class_entropy_normalized", "class_entropy_normalized"),
    ("mean_average_precision", "mean_average_precision"),
)


def _mean_over_folds(valid: list[FoldResult]) -> dict[str, float]:
    out: dict[str, float] = {}
    for mname, field in _FOLD_MEAN_FIELDS:
        vals = [getattr(r.metrics, field) for r in valid]
        out[mname] = float(mean(vals))
    return out


def _log_mean_over_folds(valid: list[FoldResult], *, topk: int, split: str) -> None:
    for mname, field in _FOLD_MEAN_FIELDS:
        vals = [getattr(r.metrics, field) for r in valid]
        v: float = mean(vals)
        logger.info(
            "mean over folds: {} = {:.4f} (topk_metric k={}, split={})",
            mname,
            v,
            topk,
            split,
        )


def run_offline_eval(a: OfflineEvalCliArgs) -> None:
    """Evaluate retrieval metrics with task-level CV or holdout; retrain on each train split."""
    prepared = _prepare_folds(a)
    if prepared is None:
        logger.error("No samples loaded from {}", a.repo)
        return
    task_to_samples, fold_list, n_samples = prepared
    n_tasks = len(task_to_samples)
    logger.info(
        "Loaded {} samples in {} task groups (key={}); {} folds",
        n_samples,
        n_tasks,
        a.task_key,
        len(fold_list),
    )

    base_cfg = a.to_offline_eval_config(experiment_name=a.experiment_name)
    results = asyncio.run(_run_folds_async(a, task_to_samples, fold_list, on_fold=None))

    valid = [r for r in results if r.error is None and r.n_test_samples_scored > 0]
    if not valid:
        logger.error("No successful folds; fold errors: {}", [r.error for r in results])
    else:
        _log_mean_over_folds(valid, topk=a.topk_metric, split=a.split)

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


@app.command(
    name="batch-redo-repos",
    help=(
        "Run KNN and AutoIntent for each JSONL repo matching a glob; append each fold and each "
        "repo x suggester summary line to JSONL immediately (durable partial results)."
    ),
)
def batch_redo_repos(
    jsonl_out: Annotated[
        Path,
        Parameter(help="JSONL path to append: one `fold` line per completed fold, then one `repo_suggester_summary`."),
    ],
    shared: Annotated[
        OfflineEvalCommonArgs,
        Parameter(name="*", help="Same flags as default command except repo / suggester / experiment-name / json-out."),
    ] = _OFFLINE_COMMON_STAR_DEFAULT,
    repo_glob: Annotated[
        str,
        Parameter(help="Glob for repo paths (cwd-relative), e.g. exported_repos/basic-fs-redo-*.jsonl."),
    ] = "exported_repos/basic-fs-redo-*.jsonl",
) -> None:
    """Evaluate knn + autointent on every matching repo; stream JSONL after each fold."""
    shared = replace(shared)
    repos = sorted(Path().glob(repo_glob))
    if not repos:
        logger.error("No paths matched repo_glob={!r}", repo_glob)
        return

    jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).isoformat()
    suggesters: tuple[Literal["knn", "autointent"], ...] = ("knn", "autointent")

    async def _batch() -> None:
        with jsonl_out.open("a", encoding="utf-8") as out_f:
            _append_jsonl(
                out_f,
                {
                    "record_type": "batch_start",
                    "timestamp": ts,
                    "repo_glob": repo_glob,
                    "repos": [str(p.resolve()) for p in repos],
                    "suggesters": list(suggesters),
                    "common": _common_fields_dict(shared),
                },
            )
            for repo in repos:
                for suggester in suggesters:
                    exp = f"batch-redo-{repo.stem}-{suggester}"
                    a = OfflineEvalCliArgs(
                        **_common_fields_dict(shared),
                        repo=repo,
                        suggester=suggester,
                        experiment_name=exp,
                        json_out=None,
                    )
                    prepared = _prepare_folds(a)
                    if prepared is None:
                        logger.error("No samples in {}; skipping", repo)
                        _append_jsonl(
                            out_f,
                            {
                                "record_type": "repo_suggester_error",
                                "timestamp": datetime.now(UTC).isoformat(),
                                "repo": str(repo.resolve()),
                                "suggester": suggester,
                                "error": "no_samples_loaded",
                            },
                        )
                        continue
                    task_to_samples, fold_list, n_samples = prepared
                    n_tasks = len(task_to_samples)
                    logger.info(
                        "Batch {} {}: {} samples, {} tasks, {} folds",
                        repo.name,
                        suggester,
                        n_samples,
                        n_tasks,
                        len(fold_list),
                    )
                    base_cfg = a.to_offline_eval_config(experiment_name=a.experiment_name)
                    repo_resolved = str(repo.resolve())
                    on_fold = _make_fold_row_appender(
                        out_f,
                        repo_resolved=repo_resolved,
                        suggester=suggester,
                        experiment_name=exp,
                        split=a.split,
                        cv_folds=a.cv_folds if a.split == "cv" else None,
                        test_size=a.test_size if a.split == "ho" else None,
                        random_state=a.random_state,
                        topk_metric=a.topk_metric,
                        task_key=a.task_key,
                    )

                    results = await _run_folds_async(a, task_to_samples, fold_list, on_fold=on_fold)
                    valid = [r for r in results if r.error is None and r.n_test_samples_scored > 0]
                    if valid:
                        _log_mean_over_folds(valid, topk=a.topk_metric, split=a.split)
                    else:
                        logger.error(
                            "No successful folds for {} {}; errors={}",
                            repo,
                            suggester,
                            [r.error for r in results],
                        )
                    _append_jsonl(
                        out_f,
                        {
                            "record_type": "repo_suggester_summary",
                            "timestamp": datetime.now(UTC).isoformat(),
                            "repo": str(repo.resolve()),
                            "suggester": suggester,
                            "experiment_name": exp,
                            "split": a.split,
                            "cv_folds": a.cv_folds if a.split == "cv" else None,
                            "test_size": a.test_size if a.split == "ho" else None,
                            "random_state": a.random_state,
                            "topk_metric": a.topk_metric,
                            "task_key": a.task_key,
                            "config": asdict(base_cfg),
                            "n_folds": len(results),
                            "n_successful_folds": len(valid),
                            "fold_errors": [r.error for r in results],
                            "mean_over_folds": _mean_over_folds(valid) if valid else None,
                        },
                    )

            _append_jsonl(
                out_f,
                {
                    "record_type": "batch_end",
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )

    asyncio.run(_batch())
    logger.info("Appended batch results to {}", jsonl_out.resolve())


def _redo_model_slug(repo: str) -> str:
    """``basic-fs-redo-haiku45_test_0`` -> ``haiku45``; otherwise ``Path(repo).stem``."""
    stem = Path(repo).stem
    prefix = "basic-fs-redo-"
    if stem.startswith(prefix) and "_test_" in stem:
        return stem[len(prefix) :].split("_test_", maxsplit=1)[0]
    return stem


def _fmt_scalar(v: float, *, decimals: int) -> str:
    return f"{v:.{decimals}f}"


def _load_batch_pivot(
    jsonl: Path,
) -> tuple[int | None, list[str], dict[str, dict[str, dict[str, float]]]]:
    """Return ``(topk_metric, preferred_row_order, model -> suggester -> mean_over_folds)``."""
    topk: int | None = None
    preferred_order: list[str] = []
    pivot: dict[str, dict[str, dict[str, float]]] = {}

    with jsonl.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            rec: dict[str, Any] = json.loads(line)
            rt = rec.get("record_type")
            if rt == "batch_start":
                for rp in rec.get("repos", []):
                    slug = _redo_model_slug(str(rp))
                    if slug not in preferred_order:
                        preferred_order.append(slug)
                continue
            if rt != "repo_suggester_summary":
                continue
            if topk is None and isinstance(rec.get("topk_metric"), int):
                topk = rec["topk_metric"]
            mof = rec.get("mean_over_folds")
            if not isinstance(mof, dict):
                continue
            suggester = str(rec.get("suggester", ""))
            if suggester not in ("knn", "autointent"):
                continue
            repo = str(rec.get("repo", ""))
            slug = _redo_model_slug(repo)
            cast_mof: dict[str, float] = {
                str(k): float(v) for k, v in mof.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            pivot.setdefault(slug, {})[suggester] = cast_mof

    return topk, preferred_order, pivot


def _pivot_row_order(preferred: list[str], pivot: dict[str, dict[str, dict[str, float]]]) -> list[str]:
    seen = set()
    out: list[str] = []
    for s in preferred:
        if s in pivot and s not in seen:
            out.append(s)
            seen.add(s)
    out.extend(s for s in sorted(pivot.keys()) if s not in seen)
    return out


@app.command(
    name="batch-jsonl-table",
    help="Pivot ``repo_suggester_summary`` into a model x macro-metrics table (AI vs KNN columns).",
)
def batch_jsonl_table(
    jsonl: Annotated[Path, Parameter(help="JSONL written by ``batch-redo-repos``.")],
    fmt: Annotated[
        Literal["markdown", "tsv"],
        Parameter(name="--format", help="markdown: GitHub-style pipe table; tsv: tabs."),
    ] = "markdown",
    decimals: Annotated[int, Parameter(help="Decimal places per metric in each cell.")] = 3,
    out: Annotated[Path | None, Parameter(help="Write table to this path instead of stdout.")] = None,
) -> None:
    """One column per macro metric; values are mean-over-folds from each suggester's summary."""
    topk, preferred, pivot = _load_batch_pivot(jsonl)
    if not pivot:
        logger.error("No repo_suggester_summary rows with mean_over_folds in {}", jsonl)
        return
    rows = _pivot_row_order(preferred, pivot)
    k = topk if topk is not None else "?"

    col_specs: tuple[tuple[str, Literal["autointent", "knn"], str], ...] = (
        ("AI top1", "autointent", "macro_top1"),
        ("KNN top1", "knn", "macro_top1"),
        ("AI topk", "autointent", "macro_topk"),
        ("KNN topk", "knn", "macro_topk"),
        ("AI MRR", "autointent", "macro_mrr"),
        ("KNN MRR", "knn", "macro_mrr"),
    )

    def cell(slug: str, sug: Literal["autointent", "knn"], metric_key: str) -> str:
        m = pivot.get(slug, {}).get(sug)
        if m is None or metric_key not in m:
            return "—"
        return _fmt_scalar(m[metric_key], decimals=decimals)

    lines: list[str] = []
    if fmt == "markdown":
        header = (
            f"Offline batch pivot (macro mean-over-folds; topk matches eval k={k}). Source: `{jsonl.name}`.\n\n"
            "| model | "
            + " | ".join(h for h, _, _ in col_specs)
            + " |\n| :--- | "
            + " | ".join("---:" for _ in col_specs)
            + " |"
        )
        lines.append(header)
        for slug in rows:
            cells = [cell(slug, sug, mk) for _, sug, mk in col_specs]
            lines.append("| " + " | ".join([slug, *cells]) + " |")
    else:
        lines.append("\t".join(["model", *[h for h, _, _ in col_specs]]))
        for slug in rows:
            cells = [cell(slug, sug, mk) for _, sug, mk in col_specs]
            lines.append("\t".join([slug, *cells]))

    text = "\n".join(lines) + "\n"
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        logger.info("Wrote {}", out.resolve())
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    app()
