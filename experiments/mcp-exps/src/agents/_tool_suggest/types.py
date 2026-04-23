"""Shared types for tool-suggest-backed agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from tool_suggest import ToolSuggestClient

type EmbBackend = Literal["openai", "st"]


@dataclass
class TSAgentState:
    tool_suggest_client: ToolSuggestClient
    speculations: list[str] = field(default_factory=list)
    tool_return_limit: int = 10_000
    top_k: int | None = None
