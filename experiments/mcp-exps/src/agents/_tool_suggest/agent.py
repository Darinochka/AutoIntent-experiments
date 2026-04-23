"""Pydantic AI agent wired to tool-suggest for prepare_tools."""

from __future__ import annotations

import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent, RunContext, ToolDefinition

from src.history_processors import truncate_tool_returns
from src.tools import change_output_limit, get_thoughts, record_intermediate_speculations

from .types import TSAgentState


def create_tool_suggest_agent(model: str) -> Agent[TSAgentState, str]:
    """Create agent that uses tool suggestion service for reducing context size.

    Details:
    - Training data is collected via run result processor (record from result.all_messages()
      with parent_context compression), not event_stream_handler.
    - suggest in Agent.__init__(prepare_tools=...)

    Note:
        Training of tool suggester is to be launched on behalf of an admin,
        that has access to control tool suggester service.

    """
    load_dotenv()
    agent = Agent(
        model,
        system_prompt=(
            "You are an autonomous worker that can use tools to complete tasks. "
            "You must solve the task before sumbitting the final answer (or report that it is not possible). "
            "Don't ask for confirmations; make your own decisions. "
            "After you submit `final_result`, you will be strictly judged and evaluated."
        ),
        tools=[record_intermediate_speculations, get_thoughts, change_output_limit],
        prepare_tools=suggest_tools,
        history_processors=[truncate_tool_returns],
        deps_type=TSAgentState,
        retries=5,
    )

    @agent.instructions
    def current_tool_return_limit(ctx: RunContext[TSAgentState]) -> str:
        return f"Current tool return limit is {ctx.deps.tool_return_limit} (in chars)."

    return agent


async def suggest_tools(ctx: RunContext[TSAgentState], tool_defs: list[ToolDefinition]) -> list[ToolDefinition] | None:
    """Filter tools using trained suggester; uses ``check_is_trained`` for local and remote clients."""
    trained = await ctx.deps.tool_suggest_client.check_is_trained()
    if not trained:
        logger.debug("Suggester is not trained yet")
        return tool_defs
    with logfire.span("Gettings suggestions from suggester") as span:
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
