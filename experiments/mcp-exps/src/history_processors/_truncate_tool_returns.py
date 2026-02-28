"""Truncator for excessive tool returns."""

from copy import deepcopy

from loguru import logger
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelRequestPart, ToolReturnPart

TOOL_RETURN_LIMIT = 10_000


def truncate_tool_returns(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Truncate overly long tool returns to prevent model fail."""
    res: list[ModelMessage] = []
    for m in messages:
        if not isinstance(m, ModelRequest):
            res.append(m)
            continue
        parts: list[ModelRequestPart] = []
        for p in m.parts:
            if not isinstance(p, ToolReturnPart):
                parts.append(p)
                continue
            if not isinstance(p.content, str):
                parts.append(p)
                continue
            if len(p.content) > TOOL_RETURN_LIMIT:
                logger.warning(f"Met too long tool return: {len(p.content)}. Truncating to {TOOL_RETURN_LIMIT}...")
                edited_part = deepcopy(p)
                edited_part.content = p.content[:TOOL_RETURN_LIMIT] + "\n[too long... truncated...]"
                parts.append(edited_part)
            else:
                parts.append(p)
        edited_message = deepcopy(m)
        edited_message.parts = parts
        res.append(edited_message)

    return res
