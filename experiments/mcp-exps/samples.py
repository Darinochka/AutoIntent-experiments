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
from pathlib import Path
from typing import Annotated, Any

import cyclopts
from dotenv import load_dotenv
from logfire.query_client import AsyncLogfireQueryClient
from loguru import logger
from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ToolCallPart,
)
from tool_suggest.models import Sample
from tool_suggest.services.repository import JSONFileRepository

load_dotenv()

app = cyclopts.App()


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
    WITH root_span AS (
        SELECT trace_id, span_id
        FROM records
        WHERE message = 'evaluate {experiment}'
          AND otel_scope_name = 'pydantic-evals'
        LIMIT 1
    ),
    case_spans AS (
        SELECT
            s.trace_id,
            s.span_id,
            s.attributes->>'case_name' as case_name,
            s.attributes->>'task_name' as task_name,
            s.attributes as case_attributes
        FROM records s
        JOIN root_span r ON s.trace_id = r.trace_id AND s.parent_span_id = r.span_id
        WHERE s.message LIKE 'case: %'
          AND s.otel_scope_name = 'pydantic-evals'
    ),
    chat_spans_ranked AS (
        SELECT
            c.case_name,
            c.task_name,
            c.trace_id,
            c.span_id as case_span_id,
            c.case_attributes,
            s.span_id as chat_span_id,
            s.attributes->'gen_ai.input.messages' as input_messages,
            s.attributes->'gen_ai.output.messages' as output_messages,
            s.attributes->'gen_ai.request.model' as request_model,
            s.attributes->'gen_ai.response.model' as response_model,
            ROW_NUMBER() OVER (PARTITION BY c.span_id ORDER BY s.start_timestamp DESC) as rank
        FROM records s
        JOIN case_spans c ON s.trace_id = c.trace_id AND s.parent_span_id = c.span_id
        WHERE s.message LIKE 'chat %'
          AND s.otel_scope_name = 'pydantic-ai'
    )
    SELECT
        case_name,
        task_name,
        trace_id,
        case_span_id,
        case_attributes,
        chat_span_id,
        input_messages,
        output_messages,
        request_model,
        response_model
    FROM chat_spans_ranked
    WHERE rank = 1
    """  # noqa: S608

    if output_path.exists():
        raise FileExistsError

    output_path.parent.mkdir(parents=True, exist_ok=True)

    async with AsyncLogfireQueryClient(
        read_token=os.getenv("LOGFIRE_API_KEY") or "fake",
        timeout=timeout,  # type: ignore[arg-type]
    ) as client:
        json_rows = await client.query_json_rows(sql=query)

    rows: list[dict[str, Any]] = json_rows.get("rows", [])
    if not rows:
        logger.warning("Logfire query returned no rows")
        return

    samples = _extract_samples_from_rows(rows=rows, experiment_name=experiment)

    repo = JSONFileRepository(file_path=output_path, collection_name=experiment)
    await repo.add_bulk(samples, wait=True)

    logger.success(f"Done! Saved {len(samples)} samples.")


def _extract_samples_from_rows(rows: list[dict[str, Any]], experiment_name: str) -> list[Sample]:
    """Extract tool-suggest samples from the last chat span under each case span."""
    all_samples: list[Sample] = []
    for row in rows:
        case_name = str(row.get("case_name") or "unknown_case")
        input_messages_raw = row.get("input_messages") or []
        output_messages_raw = row.get("output_messages") or []

        transcript = [
            *ModelMessagesTypeAdapter.validate_python(input_messages_raw),
            *ModelMessagesTypeAdapter.validate_python(output_messages_raw),
        ]

        case_samples = _samples_from_messages(
            messages=transcript,
            base_data={
                "experiment_name": experiment_name,
                "case_name": case_name,
                "task_name": row.get("task_name"),
                "trace_id": row.get("trace_id"),
                "case_span_id": row.get("case_span_id"),
                "source_chat_span_id": row.get("chat_span_id"),
                "request_model": row.get("request_model"),
                "response_model": row.get("response_model"),
            },
        )
        all_samples.extend(case_samples)

    return all_samples


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


if __name__ == "__main__":
    app()
