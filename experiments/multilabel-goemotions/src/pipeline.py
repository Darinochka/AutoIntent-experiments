"""Build, fit, and report an AutoIntent pipeline for the GoEmotions multilabel task."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from autointent import Context, Dataset, Pipeline
from autointent.configs import LoggingConfig
from autointent.metrics import DICISION_METRICS_MULTILABEL, SCORING_METRICS_MULTILABEL
from autointent.utils import load_preset
from loguru import logger

from src.naming import metrics_path

if TYPE_CHECKING:
    from autointent.custom_types import SearchSpacePreset


def _patch_embedder_offline_cache_key() -> None:
    """Resolve the embedder's cache-key commit hash locally instead of via the Hugging Face API.

    AutoIntent 0.3.x calls ``huggingface_hub.model_info`` on every ``embed`` to key its embeddings
    cache by the model's remote commit hash. Under a fast sweep this trips HF's 1000-req/5-min rate
    limit (HTTP 429) and is fatal under ``HF_HUB_OFFLINE``. The hash only needs to be a stable
    per-model identifier, so we read the cached ``refs/main`` commit (identical to the remote sha)
    and fall back to the model name — no network, and the same cache key as the online path.

    0.3.1 changed the call site to ``_get_latest_commit_hash(model_name, revision)`` (two positional
    args), so the replacement must accept the optional ``revision`` or the call raises ``TypeError``.
    """
    from autointent._wrappers.embedder import sentence_transformers as st

    def _local_commit_hash(model_name: str, revision: str | None = None) -> str:
        if Path(model_name).exists():
            return model_name
        from huggingface_hub.constants import HF_HUB_CACHE

        ref = Path(HF_HUB_CACHE) / f"models--{model_name.replace('/', '--')}" / "refs" / "main"
        try:
            return ref.read_text(encoding="utf-8").strip()
        except OSError:
            return model_name

    st._get_latest_commit_hash = _local_commit_hash


_patch_embedder_offline_cache_key()


def validate_metrics(scoring_metric: str | None, decision_metric: str | None) -> None:
    """Reject unknown target-metric names with the valid multilabel options."""
    if scoring_metric and scoring_metric not in SCORING_METRICS_MULTILABEL:
        msg = f"Invalid --scoring-metric '{scoring_metric}'. Choose from: {sorted(SCORING_METRICS_MULTILABEL)}"
        raise ValueError(msg)
    if decision_metric and decision_metric not in DICISION_METRICS_MULTILABEL:
        msg = f"Invalid --decision-metric '{decision_metric}'. Choose from: {sorted(DICISION_METRICS_MULTILABEL)}"
        raise ValueError(msg)


def build_pipeline(preset: str, seed: int, scoring_metric: str | None, decision_metric: str | None) -> Pipeline:
    """Load a preset and optionally override the scoring/decision target metrics before building."""
    validate_metrics(scoring_metric, decision_metric)
    cfg = load_preset(cast("SearchSpacePreset", preset))
    for node in cfg["search_space"]:
        if node["node_type"] == "scoring" and scoring_metric:
            node["target_metric"] = scoring_metric
        elif node["node_type"] == "decision" and decision_metric:
            node["target_metric"] = decision_metric
    return Pipeline.from_optimization_config({**cfg, "seed": seed})


def load_multilabel_dataset(data_path: str | Path) -> Dataset:
    """Load the dataset JSON and assert it is multilabel."""
    dataset = Dataset.from_json(data_path)
    if not dataset.multilabel:
        raise ValueError("Loaded dataset is not multilabel; check prepare_data.py output.")
    return dataset


def selected_modules(context: Context) -> list[dict[str, Any]]:
    """Best-effort extraction of the chosen module configs (never fatal)."""
    try:
        return [cfg.asdict() for cfg in context.optimization_info.get_inference_nodes_config()]
    except Exception as exc:  # noqa: BLE001
        return [{"error": f"could not serialize node configs: {exc}"}]


def run_experiment(
    *,
    data_path: str | Path,
    preset: str,
    exp_name: str,
    logs_dir: str | Path,
    embedder_model: str | None,
    device: str | None,
    scoring_metric: str | None,
    decision_metric: str | None,
    seed: int,
    eval_split: str = "validation",
    dump_modules: bool = True,
) -> dict[str, Any]:
    """Optimize the pipeline, evaluate on test, write the report, and return it.

    ``eval_split`` is the GoEmotions split fed as AutoIntent's ``test`` (``validation`` for Phase-1
    selection, ``test`` for Phase-2 reporting); it is recorded for provenance, not used for fitting.
    dump_modules persists fitted modules to disk for reuse; final test metrics are computed regardless
    (clear_ram is left False). Sweeps pass dump_modules=False to avoid writing a module dump per run.
    """
    dataset = load_multilabel_dataset(data_path)
    logger.info("Loaded {}: {} classes, multilabel={}", Path(data_path).name, dataset.n_classes, dataset.multilabel)
    for split in dataset:
        logger.info("split {}: {} samples", split, len(dataset[split]))

    pipeline = build_pipeline(preset, seed, scoring_metric, decision_metric)
    pipeline.set_config(LoggingConfig(project_dir=Path(logs_dir), run_name=exp_name, dump_modules=dump_modules))

    emb_updates: dict[str, str] = {}
    if embedder_model:
        emb_updates["model_name"] = embedder_model
    if device:
        emb_updates["device"] = device
    if emb_updates:
        pipeline.set_config(pipeline.embedder_config.model_copy(update=emb_updates))
        logger.info("Embedder overrides: {}", emb_updates)

    # Snapshot the splits we feed before fit() (AutoIntent carves an HPO-validation out of train in place).
    fed_split_sizes = {split: len(dataset[split]) for split in dataset}

    logger.info("Optimizing preset '{}' (experiment '{}') ...", preset, exp_name)
    context = pipeline.fit(dataset)

    metrics = dict(context.optimization_info.pipeline_metrics)
    emb_cfg = pipeline.embedder_config.model_dump()
    from importlib.metadata import version

    report: dict[str, Any] = {
        "preset": preset,
        "autointent_version": version("autointent"),
        "exp_name": exp_name,
        "n_classes": dataset.n_classes,
        "fed_split_sizes": fed_split_sizes,
        "eval_split": eval_split,
        "eval_on": f"ai-test (= GoEmotions {eval_split}); HPO-validation carved from train by AutoIntent",
        "embedder": emb_cfg.get("model_name"),
        "embedder_model_override": embedder_model,
        "device": emb_cfg.get("device"),
        "target_metrics": {
            "scoring": scoring_metric or "preset-default",
            "decision": decision_metric or "preset-default",
        },
        "test_metrics": metrics,
        "selected_modules": selected_modules(context),
    }

    out_path = metrics_path(logs_dir, exp_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("Test metrics (multilabel):")
    for name, value in metrics.items():
        logger.info("  {}: {}", name, value)
    logger.info("Wrote {}", out_path)
    return report
