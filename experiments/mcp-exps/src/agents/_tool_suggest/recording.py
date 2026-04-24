"""Record tool-suggest samples from pydantic-ai run results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai.messages import ModelResponse, ToolCallPart

from .constants import IGNORED_TOOL_LABELS
from .types import TSAgentState

if TYPE_CHECKING:
    from uuid import UUID

    from mcp_evals.task import Task
    from pydantic_ai.run import AgentRunResult


def _tool_names_from_response(msg: ModelResponse) -> list[str]:
    """Collect tool names from a ModelResponse's tool-call parts."""
    return [part.tool_name for part in msg.parts if isinstance(part, ToolCallPart) and part.tool_name]


async def tool_suggest_run_result_processor(_task: Task[Any, Any], result: AgentRunResult[Any], deps: object) -> None:
    """Record tool-suggest samples from result.all_messages() with parent_context."""
    if not isinstance(deps, TSAgentState):
        return
    client = deps.tool_suggest_client
    messages = result.all_messages()
    steps: list[tuple[int, list[str]]] = []
    for i, msg in enumerate(messages):
        if isinstance(msg, ModelResponse):
            raw_tool_names = _tool_names_from_response(msg)
            if raw_tool_names:
                filtered_tool_names = [n for n in raw_tool_names if n not in IGNORED_TOOL_LABELS]
                steps.append((i + 1, filtered_tool_names))
    if not steps:
        return
    last_sample_id: UUID | None = None
    prev_end = 0
    for end_idx, selected_tools in steps:
        context_slice = messages[prev_end:end_idx]
        sample_id = await client.record(
            context=context_slice,
            selected_tools=selected_tools,
            is_out_of_scope=not selected_tools,
            parent_context=last_sample_id,
        )
        last_sample_id = sample_id
        prev_end = end_idx
