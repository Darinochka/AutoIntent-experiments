"""Export tool-suggest samples from Logfire spans.

This script loads experiment spans from Logfire, finds the last `chat %` span
inside each `case: %` span, reconstructs the transcript from that chat's input
and output messages, then writes tool-suggest `Sample` objects to JSONL.

Usage example:
```bash
export LOGFIRE_API_KEY="..."
uv run samples.py --experiment basic-fs --output-dir ./tool_suggest_repos
```
"""

import os
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any

import cyclopts
from dotenv import load_dotenv
from logfire.query_client import AsyncLogfireQueryClient
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from tool_suggest.models import Sample

load_dotenv()

app = cyclopts.App()


class SpanRow(BaseModel):
    """Subset of Logfire span data needed for sample extraction."""

    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    span_name: str
    otel_scope_name: str
    created_at: float | None = None
    start_timestamp: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


@app.default
async def load(
    experiment: Annotated[str, cyclopts.Parameter(help="Experiment name (like basic-fs)")],
    output_dir: Annotated[
        Path | None,
        cyclopts.Parameter(help="Where to save jsonl file. CWD by default"),
    ] = None,
    timeout: Annotated[  # noqa: ASYNC109
        int,
        cyclopts.Parameter(help="Query timeout in seconds"),
    ] = 10,
) -> None:
    """Load Logfire spans and save tool-suggest samples to JSONL."""
    output_path = ((output_dir or Path.cwd()).resolve() / experiment).with_suffix(".jsonl")
    logger.info(f"Samples will be saved to {output_path}")

    query = f"""
    SELECT
        trace_id,
        span_id,
        parent_span_id,
        span_name,
        otel_scope_name,
        created_at,
        start_timestamp,
        attributes
    FROM records
    WHERE
        trace_id IN (
            SELECT DISTINCT trace_id
            FROM records
            WHERE
                message = 'evaluate {experiment}'
                AND otel_scope_name = 'pydantic-evals'
        )
        AND (
            (otel_scope_name = 'pydantic-evals' AND span_name ILIKE 'case: %')
            OR (otel_scope_name = 'pydantic-ai' AND span_name ILIKE 'chat %')
        )
    ORDER BY trace_id, created_at
    """  # noqa: S608

    output_path.parent.mkdir(parents=True, exist_ok=True)

    async with AsyncLogfireQueryClient(
        read_token=os.getenv("LOGFIRE_API_KEY") or "fake",
        timeout=timeout,  # type: ignore[arg-type]
    ) as client:
        json_rows = await client.query_json_rows(sql=query)

    rows: list[dict[str, Any]] = json_rows.get("rows", [])
    if not rows:
        logger.warning("Logfire query returned no rows.")
        return

    spans = [SpanRow.model_validate(row) for row in rows]
    samples = _extract_samples_from_spans(spans=spans, experiment_name=experiment)

    with output_path.open("w", encoding="utf-8") as output_file:
        for sample in samples:
            output_file.write(sample.model_dump_json() + "\n")

    logger.success(f"Done! Saved {len(samples)} samples.")


def _extract_samples_from_spans(spans: list[SpanRow], experiment_name: str) -> list[Sample]:
    """Extract tool-suggest samples from the last chat span under each case span."""
    children_by_parent: dict[str, list[SpanRow]] = defaultdict(list)
    case_spans: list[SpanRow] = []

    for span in spans:
        if span.parent_span_id is not None:
            children_by_parent[span.parent_span_id].append(span)
        if span.otel_scope_name == "pydantic-evals" and span.span_name.startswith("case: "):
            case_spans.append(span)

    all_samples: list[Sample] = []
    for case_span in case_spans:
        case_name = str(case_span.attributes.get("case_name") or "unknown_case")
        chat_spans = _collect_descendant_chat_spans(case_span.span_id, children_by_parent)
        if not chat_spans:
            logger.warning(f"No chat spans found for case {case_name}")
            continue

        last_chat = max(chat_spans, key=_span_sort_key)
        transcript = _messages_from_last_chat(last_chat)
        case_samples = _samples_from_messages(
            messages=transcript,
            base_data={
                "experiment_name": experiment_name,
                "case_name": case_name,
                "task_name": case_span.attributes.get("task_name"),
                "trace_id": case_span.trace_id,
                "case_span_id": case_span.span_id,
                "source_chat_span_id": last_chat.span_id,
                "request_model": last_chat.attributes.get("gen_ai.request.model")
                or last_chat.attributes.get("gen_ai_request_model"),
                "response_model": last_chat.attributes.get("gen_ai.response.model")
                or last_chat.attributes.get("gen_ai_response_model"),
            },
        )
        all_samples.extend(case_samples)

    return all_samples


def _collect_descendant_chat_spans(
    root_span_id: str,
    children_by_parent: dict[str, list[SpanRow]],
) -> list[SpanRow]:
    """Collect descendant `chat %` spans under the given case span."""
    stack = [root_span_id]
    result: list[SpanRow] = []

    while stack:
        current_id = stack.pop()
        for child in children_by_parent.get(current_id, []):
            stack.append(child.span_id)
            if child.otel_scope_name == "pydantic-ai" and child.span_name.startswith("chat "):
                result.append(child)

    return result


def _span_sort_key(span: SpanRow) -> tuple[float, str]:
    """Sort spans by creation time with a stable fallback."""
    created_at = span.created_at if isinstance(span.created_at, int | float) else 0.0
    return (float(created_at), span.start_timestamp or "")


def _messages_from_last_chat(chat_span: SpanRow) -> list[ModelMessage]:
    """Build transcript from the last chat span's input and output messages."""
    attributes = chat_span.attributes or {}
    system_instructions = _extract_system_instructions(attributes.get("gen_ai.system_instructions"))
    input_messages_raw = attributes.get("gen_ai.input.messages") or []
    output_messages_raw = attributes.get("gen_ai.output.messages") or []
    return [
        *_convert_logfire_messages(input_messages_raw, system_instructions=system_instructions),
        *_convert_logfire_messages(output_messages_raw, system_instructions=None),
    ]


def _extract_system_instructions(system_instructions_raw: object) -> str | None:
    """Extract plain-text instructions from Logfire system instructions payload."""
    if not isinstance(system_instructions_raw, list):
        return None

    chunks: list[str] = []
    for part in system_instructions_raw:
        if not isinstance(part, dict):
            continue
        content = part.get("content")
        if isinstance(content, str) and content:
            chunks.append(content)

    return "\n".join(chunks) if chunks else None


def _convert_logfire_messages(messages_raw: object, system_instructions: str | None) -> list[ModelMessage]:
    """Convert Logfire message payload to Pydantic AI messages."""
    if not isinstance(messages_raw, list):
        return []

    converted: list[ModelMessage] = []
    for msg_raw in messages_raw:
        if not isinstance(msg_raw, dict):
            continue
        converted_msg = _convert_logfire_message(msg_raw, system_instructions=system_instructions)
        if converted_msg is not None:
            converted.append(converted_msg)
    return converted


def _convert_logfire_message(message_raw: dict[str, Any], system_instructions: str | None) -> ModelMessage | None:
    """Convert one Logfire message dict into a `ModelMessage`."""
    role = str(message_raw.get("role") or "")
    parts_raw_obj = message_raw.get("parts") or []
    parts_raw = [part for part in parts_raw_obj if isinstance(part, dict)] if isinstance(parts_raw_obj, list) else []

    if role == "assistant":
        response_parts = _convert_assistant_parts(parts_raw)
        if not response_parts:
            return None
        return ModelResponse(parts=response_parts)

    request_parts = _convert_request_parts(role=role, message_raw=message_raw, parts_raw=parts_raw)
    if not request_parts:
        return None
    return ModelRequest(parts=request_parts, instructions=system_instructions)


def _convert_request_parts(
    role: str,
    message_raw: dict[str, Any],
    parts_raw: list[dict[str, Any]],
) -> list[SystemPromptPart | UserPromptPart | ToolReturnPart]:
    """Convert request-side Logfire parts."""
    result: list[SystemPromptPart | UserPromptPart | ToolReturnPart] = []

    if role == "system":
        for part_raw in parts_raw:
            content = _text_content(part_raw.get("content"))
            if content:
                result.append(SystemPromptPart(content=content))
        return result

    if role == "user":
        for part_raw in parts_raw:
            content = _text_content(part_raw.get("content"))
            if content:
                result.append(UserPromptPart(content=content))
        return result

    if role == "tool":
        tool_name = _first_str(message_raw.get("name"), message_raw.get("tool_name")) or "unknown_tool"
        tool_call_id = _first_str(message_raw.get("tool_call_id"), message_raw.get("id")) or f"tool_{tool_name}"
        content = _tool_return_content(message_raw.get("content"))
        result.append(ToolReturnPart(tool_name=tool_name, content=content, tool_call_id=tool_call_id))
        return result

    for part_raw in parts_raw:
        part_type = str(part_raw.get("type") or "")
        if part_type in {"tool_return", "tool-return"}:
            tool_name = _first_str(part_raw.get("name"), part_raw.get("tool_name")) or "unknown_tool"
            tool_call_id = _first_str(part_raw.get("tool_call_id"), part_raw.get("id")) or f"tool_{tool_name}"
            content = _tool_return_content(part_raw.get("content"))
            result.append(ToolReturnPart(tool_name=tool_name, content=content, tool_call_id=tool_call_id))

    return result


def _convert_assistant_parts(parts_raw: list[dict[str, Any]]) -> list[TextPart | ToolCallPart]:
    """Convert assistant-side Logfire parts."""
    result: list[TextPart | ToolCallPart] = []

    for part_raw in parts_raw:
        part_type = str(part_raw.get("type") or "")
        if part_type == "text":
            content = _text_content(part_raw.get("content"))
            if content:
                result.append(TextPart(content=content))
        elif part_type in {"tool_call", "tool-call"}:
            tool_name = _first_str(part_raw.get("name"), part_raw.get("tool_name"))
            if not tool_name:
                continue
            tool_call_id = _first_str(part_raw.get("id"), part_raw.get("tool_call_id")) or f"call_{tool_name}"
            result.append(
                ToolCallPart(
                    tool_name=tool_name,
                    args=part_raw.get("arguments"),
                    tool_call_id=tool_call_id,
                )
            )

    return result


def _samples_from_messages(messages: list[ModelMessage], base_data: dict[str, Any]) -> list[Sample]:
    """Create parent-linked samples from a reconstructed transcript."""
    steps: list[tuple[int, list[str]]] = []
    for i, msg in enumerate(messages):
        if isinstance(msg, ModelResponse):
            tool_names = _tool_names_from_response(msg)
            if tool_names:
                steps.append((i + 1, tool_names))

    samples: list[Sample] = []
    last_sample_id = None
    prev_end = 0

    for end_idx, selected_tools in steps:
        context_slice = messages[prev_end:end_idx]
        sample = Sample(
            context=context_slice,
            tools=selected_tools,
            data=dict(base_data),
            parent_context=last_sample_id,
        )
        samples.append(sample)
        last_sample_id = sample.id
        prev_end = end_idx

    return samples


def _tool_names_from_response(msg: ModelResponse) -> list[str]:
    """Collect tool names from a ModelResponse's tool-call parts."""
    return [part.tool_name for part in msg.parts if isinstance(part, ToolCallPart)]


def _first_str(*values: object) -> str | None:
    """Return the first non-empty string representation."""
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None


def _text_content(value: object) -> str | None:
    """Extract plain text content from a Logfire message part."""
    if isinstance(value, str):
        return value
    return None


def _tool_return_content(value: object) -> str | list[str]:
    """Normalize tool return payload into ToolReturnPart-compatible content."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return ""
    return str(value)


if __name__ == "__main__":
    app()
