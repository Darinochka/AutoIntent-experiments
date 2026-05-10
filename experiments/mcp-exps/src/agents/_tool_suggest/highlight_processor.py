"""Inject tool-suggest output into the model-visible history (highlighter mode)."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import logfire
from loguru import logger
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelRequestPart, UserPromptPart

if TYPE_CHECKING:
    from pydantic_ai import RunContext

    from .types import TSAgentState

# Distinct sentinel so we can strip injected blocks from suggester input (avoids classifier bias).
TOOL_SUGGEST_HIGHLIGHT_PREFIX = "[mcp-exps:tool-suggest-highlight]\n"


async def highlight_tool_suggestions(ctx: RunContext[TSAgentState], messages: list[ModelMessage]) -> list[ModelMessage]:
    """Append ranked tools as a synthetic user prompt on the trailing :class:`ModelRequest`.

    Context passed to the suggester has prior highlight blocks stripped. The last
    ``ModelRequest`` may already end with a highlight from a retried request; that
    tail is removed before appending a fresh block.
    """
    trained = await ctx.deps.tool_suggest_client.check_is_trained()
    if not trained:
        logger.debug("Suggester is not trained yet; skipping tool highlight")
        return messages

    if not messages or not isinstance(messages[-1], ModelRequest):
        return messages

    suggest_messages = strip_tool_suggest_highlights(messages)
    with logfire.span("Tool-suggest highlighter") as span:
        suggest_result = await ctx.deps.tool_suggest_client.suggest_detailed(
            context=suggest_messages,
            top_k=ctx.deps.top_k,
            session_id=ctx.deps.suggest_session_id,
        )
        labels = [s.id for s in suggest_result.suggestions]
        span.set_attribute("tools_suggested", labels)
        span.set_attribute("reason", suggest_result.reason.value)
        span.set_attribute("explanation_detail", suggest_result.detail)

    out = deepcopy(messages)
    last = out[-1]
    if not isinstance(last, ModelRequest):
        return messages
    parts = list(last.parts)
    while parts and _is_highlight_part(parts[-1]):
        parts.pop()

    body = _format_highlight_body(labels)
    if body:
        parts.append(UserPromptPart(content=TOOL_SUGGEST_HIGHLIGHT_PREFIX + body))
        last.parts = parts
    return out


def _is_highlight_part(part: ModelRequestPart) -> bool:
    if not isinstance(part, UserPromptPart):
        return False
    content = part.content
    return isinstance(content, str) and content.startswith(TOOL_SUGGEST_HIGHLIGHT_PREFIX)


def strip_tool_suggest_highlights(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Return a deep copy of history with highlight user-prompt parts removed."""
    out: list[ModelMessage] = []
    for msg in messages:
        if not isinstance(msg, ModelRequest):
            out.append(deepcopy(msg))
            continue
        kept = [deepcopy(p) for p in msg.parts if not _is_highlight_part(p)]
        if not kept:
            continue
        clone = deepcopy(msg)
        clone.parts = kept
        out.append(clone)
    return out


def _format_highlight_body(tool_labels: list[str], detail: str | None = None) -> str | None:
    if not tool_labels:
        return None
    intro = "Suggested tools for this step: "
    names = ", ".join(tool_labels)
    body = intro + names
    if detail and detail.strip():
        body += f"\nDetail: {detail.strip()}"
    return body
