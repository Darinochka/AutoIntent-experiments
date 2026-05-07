"""Convert Logfire GenAI-format messages to pydantic-ai ModelMessage objects."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)


def convert_genai_messages(raw_messages: list[dict[str, Any]]) -> list[ModelMessage]:  # noqa: C901, PLR0912
    """Convert Logfire ``gen_ai.*.messages`` rows to pydantic-ai :class:`ModelMessage` list."""
    result: list[ModelMessage] = []
    for msg in raw_messages:
        role = msg.get("role")
        parts_raw = msg.get("parts", [])

        if role in ("system", "user"):
            request_parts: list[ModelRequestPart] = []
            for p in parts_raw:
                ptype = p.get("type")
                if ptype == "text":
                    if role == "system":
                        request_parts.append(SystemPromptPart(content=p["content"]))
                    else:
                        request_parts.append(UserPromptPart(content=p["content"]))
                elif ptype == "tool_call_response":
                    request_parts.append(
                        ToolReturnPart(
                            tool_name=p["name"],
                            content=p["result"],
                            tool_call_id=p.get("id"),
                        )
                    )
            if request_parts:
                result.append(ModelRequest(parts=request_parts))

        elif role == "assistant":
            response_parts: list[ModelResponsePart] = []
            for p in parts_raw:
                ptype = p.get("type")
                if ptype == "tool_call":
                    response_parts.append(
                        ToolCallPart(
                            tool_name=p["name"],
                            args=p["arguments"],
                            tool_call_id=p.get("id"),
                        )
                    )
                elif ptype == "text":
                    response_parts.append(TextPart(content=p["content"]))
            if response_parts:
                result.append(ModelResponse(parts=response_parts))

    return result


def transcript_from_chat_span(
    input_messages: list[dict[str, Any]],
    output_messages: list[dict[str, Any]],
) -> list[ModelMessage]:
    """One Logfire ``chat %`` span: request messages then response messages."""
    return [
        *convert_genai_messages(input_messages),
        *convert_genai_messages(output_messages),
    ]


def serialize_model_messages_json(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    """Serialize pydantic-ai messages to JSON-compatible dicts (for export scripts)."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, BaseModel):
            dumped = msg.model_dump(mode="json")
            out.append(dumped if isinstance(dumped, dict) else {"value": dumped})
        else:
            out.append({"repr": repr(msg)})
    return out
