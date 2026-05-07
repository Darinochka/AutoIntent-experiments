"""Shared pydantic-ai capabilities for MCP experiment agents (basic vs tool-suggest)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_ai import RunContext  # noqa: TC002 — nested instruction callable uses deps typing
from pydantic_ai.capabilities import AbstractCapability, HistoryProcessor, PrepareTools, Toolset
from pydantic_ai.tools import SystemPromptFunc  # noqa: TC002 — return type of get_instructions
from pydantic_ai.toolsets import FunctionToolset

from src.history_processors import truncate_tool_returns
from src.tools import change_output_limit, get_thoughts, record_intermediate_speculations

WORKER_SYSTEM_PROMPT = (
    "You are an autonomous worker that can use tools to complete tasks. "
    "You must solve the task before sumbitting the final answer (or report that it is not possible). "
    "Don't ask for confirmations; make your own decisions. "
    "After you submit `final_result`, you will be strictly judged and evaluated."
)

_worker_toolset = FunctionToolset(
    tools=[record_intermediate_speculations, get_thoughts, change_output_limit],
)


@dataclass
class ToolReturnLimitInstructions(AbstractCapability[Any]):
    """Adds dynamic instructions for the current tool-return truncation limit."""

    def get_instructions(self) -> SystemPromptFunc[Any]:
        """Append the current ``tool_return_limit`` from deps to the system prompt."""

        def current_tool_return_limit(ctx: RunContext[Any]) -> str:
            return f"Current tool return limit is {ctx.deps.tool_return_limit} (in chars)."

        return current_tool_return_limit


@dataclass
class ToolSuggestHighlighterInstructions(AbstractCapability[Any]):
    """Explains non-binding tool-suggestion notes for highlighter mode."""

    def get_instructions(self) -> SystemPromptFunc[Any]:
        """Append guidance on how to interpret injected tool-suggestion notes."""

        def highlighter_instructions(_ctx: RunContext[Any]) -> str:
            return (
                "The conversation may include short appended notes listing suggested tools for the current step. "
                "Treat them as strong recommendations when they fit the task; they are not mandatory, and you "
                "may use any available tool, including tools not listed there."
            )

        return highlighter_instructions


def shared_worker_capabilities() -> list[AbstractCapability[Any]]:
    """Capabilities shared by baseline and tool-suggest agents."""
    return [
        Toolset(_worker_toolset),
        HistoryProcessor(truncate_tool_returns),
        ToolReturnLimitInstructions(),
    ]


def tool_suggest_capabilities() -> list[AbstractCapability[Any]]:
    """Shared worker stack plus tool filtering via trained suggester."""
    # Lazy import: importing `_tool_suggest.suggest_filter` loads `_tool_suggest/__init__.py`,
    # which imports `agent`, which imports this module — avoid circular import at startup.
    from ._tool_suggest.suggest_filter import suggest_tools  # noqa: PLC0415

    return [*shared_worker_capabilities(), PrepareTools(suggest_tools)]


def tool_suggest_highlighter_capabilities() -> list[AbstractCapability[Any]]:
    """Worker stack with truncation then highlight injection; full tool list (no ``PrepareTools``)."""
    from ._tool_suggest.highlight_processor import highlight_tool_suggestions  # noqa: PLC0415

    return [
        Toolset(_worker_toolset),
        HistoryProcessor(truncate_tool_returns),
        HistoryProcessor(highlight_tool_suggestions),
        ToolReturnLimitInstructions(),
        ToolSuggestHighlighterInstructions(),
    ]
