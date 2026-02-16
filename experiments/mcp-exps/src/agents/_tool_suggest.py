from collections.abc import AsyncIterable
from dataclasses import dataclass

import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import (
    Agent,
    AgentStreamEvent,
    FunctionToolCallEvent,
    RunContext,
    ToolDefinition,
)
from tool_suggest import ToolSuggestClient

from src.history_processors import truncate_tool_returns
from src.tools import intermediate_speculations


@dataclass
class AgentState:
    tool_suggest_client: ToolSuggestClient


def create_tool_suggest_agent(model: str) -> Agent[AgentState, str]:
    """Create agent that uses tool suggestion service for reducing context size.

    Details:
    - collect train data using FunctionToolCallEvent stream handler
    - suggest in Agent.__init__(prepare_tools=...)

    Note:
        Training of tool suggester is to be launched on behalf of an admin,
        that has access to control tool suggester service.

    """
    load_dotenv()
    return Agent(
        model,
        system_prompt=(
            "You are a helpful assistant that can use tools to complete tasks. "
            "You can provide text messages beside the final answer as a means of "
            "intermediate speculations and reasoning."
        ),
        tools=[intermediate_speculations],
        prepare_tools=suggest_tools,
        history_processors=[truncate_tool_returns],
        deps_type=AgentState,
        event_stream_handler=_record_tool_calls,
        retries=5,
    )


async def suggest_tools(ctx: RunContext[AgentState], tool_defs: list[ToolDefinition]) -> list[ToolDefinition] | None:
    if not ctx.deps.tool_suggest_client.is_trained:
        logger.debug("Suggester is not trained yet")
        return tool_defs
    with logfire.span("Gettings suggestions from suggester") as span:
        messages = ctx.messages
        suggestions = await ctx.deps.tool_suggest_client.suggest(context=messages)
        names = [s.id for s in suggestions]
        selected = sorted((t for t in tool_defs if t.name in names), key=lambda t: names.index(t.name))
        span.set_attribute("tools_suggested", [s.name for s in selected])
    return selected


async def _handle_event(ctx: RunContext[AgentState], event: AgentStreamEvent) -> None:
    if not isinstance(event, FunctionToolCallEvent):
        return
    context = ctx.messages
    # NOTE: handle multiple tool calls in a single run gracefully
    selected_tools = [event.part.tool_name]
    await ctx.deps.tool_suggest_client.record(context=context, selected_tools=selected_tools)


async def _record_tool_calls(
    ctx: RunContext[AgentState],
    event_stream: AsyncIterable[AgentStreamEvent],
) -> None:
    async for event in event_stream:
        await _handle_event(ctx, event)
