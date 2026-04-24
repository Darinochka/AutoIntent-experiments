"""Embedders and AutoIntent optimization config for local tool-suggest backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, assert_never

import tiktoken
from autointent import OptimizationConfig
from autointent.configs import (
    EmbedderConfig,
    LoggingConfig,
    OpenaiEmbeddingConfig,
    SentenceTransformerEmbeddingConfig,
)
from tool_suggest.services.embedder import OpenAIEmbedder, SentenceTransformerEmbedder

from .types import EmbBackend  # noqa: TC001  # used in runtime-evaluated annotations for public API

if TYPE_CHECKING:
    from collections.abc import Callable

    from tool_suggest.services.embedder import BaseEmbedder

# Decision modules with supports_oos=True in autointent (excludes argmax, adaptive).
_OOS_DECISION_MODULE_NAMES: frozenset[str] = frozenset({"threshold"})


def _filter_decision_modules_to_oos_only(search_space: list[dict[str, Any]]) -> None:
    """Drop decision modules that cannot emit out-of-scope (None) predictions."""
    for node in search_space:
        if node.get("node_type") != "decision":
            continue
        inner = node.get("search_space")
        if not isinstance(inner, list):
            continue
        node["search_space"] = [
            entry
            for entry in inner
            if isinstance(entry, dict) and entry.get("module_name") in _OOS_DECISION_MODULE_NAMES
        ]


def get_ai_config(experiment_name: str, *, ai_embedder_config: EmbedderConfig) -> OptimizationConfig:
    ai_config = OptimizationConfig.from_preset("classic-light")
    _filter_decision_modules_to_oos_only(ai_config.search_space)
    ai_config.logging_config = LoggingConfig(
        dump_modules=True, clear_ram=True, project_dir="./.autointent_runs", run_name=experiment_name
    )
    ai_config.embedder_config = ai_embedder_config
    return ai_config


def build_tiktoken_counter(model: str) -> Callable[[str], int]:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    def counter(text: str) -> int:
        return len(encoding.encode(text))

    return counter


def build_embedding_resources(
    *,
    emb_backend: EmbBackend,
    emb_model: str,
    st_classification_prompt: str | None = None,
    st_query_prompt: str | None = None,
) -> tuple[BaseEmbedder, Callable[[str], int] | None, EmbedderConfig]:
    """Instantiate embedder(s) and token counter based on CLI settings.

    When ``emb_backend`` is ``"st"``, optional prompts are forwarded to AutoIntent's
    ``SentenceTransformerEmbeddingConfig`` (``classification_prompt`` for linear-style
    ``TaskTypeEnum.classification`` embeddings, ``query_prompt`` for
    ``TaskTypeEnum.query`` / KNN query side). They are ignored for OpenAI embeddings.
    """
    tool_suggest_embedder: BaseEmbedder
    token_counter: Callable[[str], int] | None
    ai_embedder_config: EmbedderConfig
    if emb_backend == "openai":
        tool_suggest_embedder = OpenAIEmbedder(model=emb_model)
        token_counter = build_tiktoken_counter(model=emb_model)
        ai_embedder_config = OpenaiEmbeddingConfig(model_name=emb_model)
    elif emb_backend == "st":
        tool_suggest_embedder = SentenceTransformerEmbedder(model_name=emb_model)
        token_counter = None
        ai_embedder_config = SentenceTransformerEmbeddingConfig(
            model_name=emb_model,
            classification_prompt=st_classification_prompt,
            query_prompt=st_query_prompt,
        )
    else:
        assert_never(emb_backend)

    return (tool_suggest_embedder, token_counter, ai_embedder_config)
