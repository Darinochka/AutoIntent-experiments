"""Embedders and AutoIntent optimization config for local tool-suggest backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

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


def get_ai_config(experiment_name: str, *, ai_embedder_config: EmbedderConfig) -> OptimizationConfig:
    ai_config = OptimizationConfig.from_preset("classic-light")
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
    *, emb_backend: EmbBackend, emb_model: str
) -> tuple[BaseEmbedder, Callable[[str], int] | None, EmbedderConfig]:
    """Instantiate embedder(s) and token counter based on CLI settings."""
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
        ai_embedder_config = SentenceTransformerEmbeddingConfig(model_name=emb_model)
    else:
        assert_never(emb_backend)

    return (tool_suggest_embedder, token_counter, ai_embedder_config)
