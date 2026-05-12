"""Pydantic AI agent wired to tool-suggest for prepare_tools."""

from __future__ import annotations

from dotenv import load_dotenv
from pydantic_ai import Agent

from src.agents.worker_capabilities import (
    WORKER_SYSTEM_PROMPT,
    tool_suggest_capabilities,
    tool_suggest_highlighter_capabilities,
)

from .types import TSAgentState


def create_tool_suggest_agent(model: str) -> Agent[TSAgentState, str]:
    """Create agent that uses tool suggestion service for reducing context size.

    Details:
    - Training data is collected via run result processor (record from result.all_messages()
      with parent_context compression), not event_stream_handler.
    - Tool filtering comes from :class:`pydantic_ai.capabilities.PrepareTools`, registered by
      ``tool_suggest_capabilities()`` in :mod:`src.agents.worker_capabilities`.

    Note:
        Training of tool suggester is to be launched on behalf of an admin,
        that has access to control tool suggester service.

    """
    load_dotenv()
    return Agent(
        model,
        system_prompt=WORKER_SYSTEM_PROMPT,
        capabilities=tool_suggest_capabilities(),
        deps_type=TSAgentState,
        retries=5,
    )


def create_tool_suggest_highlighter_agent(model: str) -> Agent[TSAgentState, str]:
    """Like :func:`create_tool_suggest_agent` but keeps all tools in schema and injects suggestions in history."""
    load_dotenv()
    return Agent(
        model,
        system_prompt=WORKER_SYSTEM_PROMPT,
        capabilities=tool_suggest_highlighter_capabilities(),
        deps_type=TSAgentState,
        retries=5,
    )
