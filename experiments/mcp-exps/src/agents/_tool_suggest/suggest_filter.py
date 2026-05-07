"""prepare_tools filter for tool-suggest-backed agents."""

from __future__ import annotations

from typing import TYPE_CHECKING

import logfire
from loguru import logger

if TYPE_CHECKING:
    from pydantic_ai import RunContext, ToolDefinition

    from .types import TSAgentState


async def suggest_tools(
    ctx: RunContext[TSAgentState],
    tool_defs: list[ToolDefinition],
) -> list[ToolDefinition] | None:
    """Filter tools using trained suggester; uses ``check_is_trained`` for local and remote clients."""
    trained = await ctx.deps.tool_suggest_client.check_is_trained()
    if not trained:
        logger.debug("Suggester is not trained yet")
        return tool_defs
    with logfire.span("Getting suggestions from suggester") as span:
        messages = ctx.messages
        suggest_result = await ctx.deps.tool_suggest_client.suggest_detailed(
            context=messages,
            top_k=ctx.deps.top_k,
            session_id=ctx.deps.suggest_session_id,
        )
        names = [s.id for s in suggest_result.suggestions]
        selected = sorted((t for t in tool_defs if t.name in names), key=lambda t: names.index(t.name))
        span.set_attribute("tools_suggested", [s.name for s in selected])
        span.set_attribute("reason", suggest_result.reason.value)
        span.set_attribute("explanation_detail", suggest_result.detail)
    return selected
