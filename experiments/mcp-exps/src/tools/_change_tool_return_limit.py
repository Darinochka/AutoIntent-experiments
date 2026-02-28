from typing import Protocol

from pydantic import Field
from pydantic_ai import RunContext


class DepsWithToolReturnLimit(Protocol):
    tool_return_limit: int


def change_output_limit(
    ctx: RunContext[DepsWithToolReturnLimit],
    limit: int = Field(description="maximum tool output size (in chars)", gt=0, lt=3_500_000),
) -> str:
    """Change limit of tool returns in chars.

    By default tool outputs are limited to avoid accidental bloat. This tool allows
    for reading long files by changing the truncation length.

    Returns:
        operation status message
    """
    ctx.deps.tool_return_limit = limit
    return f"Tool output limit changed to {limit}"
