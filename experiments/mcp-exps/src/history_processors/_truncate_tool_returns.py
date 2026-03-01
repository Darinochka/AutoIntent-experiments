"""Truncator for excessive tool returns."""

from copy import deepcopy
from typing import Protocol

from loguru import logger
from pydantic_ai import RunContext
from pydantic_ai.messages import MULTI_MODAL_CONTENT_TYPES, ModelMessage, ModelRequest, ModelRequestPart, ToolReturnPart

TRUNCATION_MESSAGE = "\n[too long... truncated...]"


class DepsWithToolReturnLimit(Protocol):
    tool_return_limit: int


def truncate_tool_returns(ctx: RunContext[DepsWithToolReturnLimit], messages: list[ModelMessage]) -> list[ModelMessage]:
    """Truncate overly long tool returns to prevent model fail."""
    res: list[ModelMessage] = []
    limit = ctx.deps.tool_return_limit
    for m in messages:
        if not isinstance(m, ModelRequest):
            res.append(m)
            continue
        parts: list[ModelRequestPart] = []
        for p in m.parts:
            if not isinstance(p, ToolReturnPart):
                parts.append(p)
                continue
            if isinstance(p.content, MULTI_MODAL_CONTENT_TYPES):
                parts.append(p)
                continue
            content_stringifyed = str(p.content)
            if len(content_stringifyed) > limit:
                logger.warning(f"Met too long tool return: {len(content_stringifyed)}. Truncating to {limit}...")
                edited_part = deepcopy(p)
                edited_part.content = content_stringifyed[: limit - len(TRUNCATION_MESSAGE)] + TRUNCATION_MESSAGE
                parts.append(edited_part)
            else:
                parts.append(p)
        edited_message = deepcopy(m)
        edited_message.parts = parts
        res.append(edited_message)

    return res
