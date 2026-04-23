"""Constants shared by recording and JSONL replay."""

IGNORED_TOOL_LABELS: frozenset[str] = frozenset(
    {
        # pydantic-ai ToolOutput names (final structured output)
        "final_result",
        "finish",
    }
)
