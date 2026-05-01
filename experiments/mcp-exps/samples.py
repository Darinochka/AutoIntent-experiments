"""Export tool-suggest samples from Logfire spans.

This script loads experiment spans from Logfire, finds the last `chat %` span
inside each `case: %` span, reconstructs the transcript from that chat's input
and output messages, then writes tool-suggest `Sample` objects to JSONL.

All traces whose root message is ``evaluate <experiment>`` are considered (e.g. retries).
For each ``case_name``, the row from the **latest** evaluate root (by root start time) wins.

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
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from tool_suggest.models import Sample
from tool_suggest.services.repository import JSONFileRepository

from src.logfire_genai import transcript_from_chat_span

load_dotenv()

app = cyclopts.App()

_IGNORED_TOOL_LABELS: frozenset[str] = frozenset(
    {
        # pydantic-ai ToolOutput names (final structured output)
        "final_result",
        "finish",
    }
)


def _sql_single_quoted_literal(value: str) -> str:
    """Escape a string for safe use inside a single-quoted SQL literal."""
    return value.replace("'", "''")


def _evaluate_message_literal(experiment: str) -> str:
    return _sql_single_quoted_literal(f"evaluate {experiment}")


def _build_samples_query(*, evaluate_message_sql: str) -> str:
    """SQL: all evaluate roots for this experiment; dedupe ``case_name`` to latest root."""
    return f"""
    WITH evaluate_roots AS (
        SELECT trace_id, span_id, start_timestamp AS root_start_ts
        FROM records
        WHERE message = '{evaluate_message_sql}'
          AND otel_scope_name = 'pydantic-evals'
    ),
    case_spans AS (
        SELECT
            r.root_start_ts,
            s.trace_id,
            s.span_id,
            s.start_timestamp AS case_start_ts,
            s.end_timestamp,
            s.attributes->>'case_name' AS case_name,
            s.attributes->>'task_name' AS task_name,
            s.attributes AS case_attributes
        FROM records s
        INNER JOIN evaluate_roots r
          ON s.trace_id = r.trace_id AND s.parent_span_id = r.span_id
        WHERE s.span_name ILIKE 'case: %'
          AND s.otel_scope_name = 'pydantic-evals'
    ),
    chat_spans_ranked AS (
        SELECT
            c.root_start_ts,
            c.case_start_ts,
            c.case_name,
            c.task_name,
            c.trace_id,
            c.span_id AS case_span_id,
            c.case_attributes,
            s.span_id AS chat_span_id,
            s.attributes->'gen_ai.input.messages' AS input_messages,
            s.attributes->'gen_ai.output.messages' AS output_messages,
            s.attributes->'gen_ai.request.model' AS request_model,
            s.attributes->'gen_ai.response.model' AS response_model,
            ROW_NUMBER() OVER (
                PARTITION BY c.span_id ORDER BY s.start_timestamp DESC
            ) AS rank
        FROM records s
        INNER JOIN case_spans c
          ON s.trace_id = c.trace_id
          AND s.start_timestamp >= c.case_start_ts
          AND s.start_timestamp <= c.end_timestamp
        WHERE s.span_name ILIKE 'chat %'
          AND s.otel_scope_name = 'pydantic-ai'
    ),
    best_chat_per_case_span AS (
        SELECT *
        FROM chat_spans_ranked
        WHERE rank = 1
    ),
    winner_per_case_name AS (
        SELECT *
        FROM (
            SELECT
                bc.root_start_ts,
                bc.case_start_ts,
                bc.case_name,
                bc.task_name,
                bc.trace_id,
                bc.case_span_id,
                bc.case_attributes,
                bc.chat_span_id,
                bc.input_messages,
                bc.output_messages,
                bc.request_model,
                bc.response_model,
                ROW_NUMBER() OVER (
                    PARTITION BY bc.case_name
                    ORDER BY bc.root_start_ts DESC, bc.case_start_ts DESC
                ) AS case_name_rank
            FROM best_chat_per_case_span bc
        ) ranked
        WHERE case_name_rank = 1
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
    FROM winner_per_case_name
    ORDER BY case_name
    """  # noqa: S608


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

    evaluate_msg = _evaluate_message_literal(experiment)
    query = _build_samples_query(evaluate_message_sql=evaluate_msg)

    if output_path.exists():
        raise FileExistsError

    output_path.parent.mkdir(parents=True, exist_ok=True)

    async with AsyncLogfireQueryClient(
        read_token=os.getenv("LOGFIRE_API_KEY") or "fake",
        timeout=timeout,  # type: ignore[arg-type]
    ) as client:
        json_rows = await client.query_json_rows(sql=query, limit=10_000)

    rows: list[dict[str, Any]] = json_rows.get("rows", [])
    if not rows:
        logger.warning("Logfire query returned no rows")
        return

    logger.info(
        "After merging duplicate case_name across traces: {} case row(s)",
        len(rows),
    )

    samples = _extract_samples_from_rows(rows=rows, experiment_name=experiment)

    repo = JSONFileRepository(file_path=output_path, collection_name=experiment)
    await repo.add_bulk(samples)

    logger.success(f"Done! Saved {len(samples)} samples.")


def _extract_samples_from_rows(rows: list[dict[str, Any]], experiment_name: str) -> list[Sample]:
    """Extract tool-suggest samples from the last chat span under each case span."""
    all_samples: list[Sample] = []
    for row in rows:
        case_name = str(row.get("case_name") or "unknown_case")
        input_messages_raw = row.get("input_messages") or []
        output_messages_raw = row.get("output_messages") or []

        transcript = transcript_from_chat_span(input_messages_raw, output_messages_raw)

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
            raw_tool_names = _tool_names_from_response(msg)
            if raw_tool_names:
                filtered_tool_names = [n for n in raw_tool_names if n not in _IGNORED_TOOL_LABELS]
                steps.append((i + 1, filtered_tool_names))

    samples: list[Sample] = []
    last_sample_id = None
    prev_end = 0

    for end_idx, selected_tools in steps:
        context_slice = messages[prev_end:end_idx]
        sample = Sample(
            context=context_slice,
            tools=selected_tools,
            is_out_of_scope=not selected_tools,
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
