"""Download case-wise chat histories from Logfire for an MCP-evals experiment.

Unlike :file:`samples.py`, this script pulls **every** ``chat %`` span under each
``case: %`` span (ordered by time), groups them by evaluation case, and writes
structured JSON for qualitative analysis.

Usage:
```bash
export LOGFIRE_API_KEY="..."
uv run case_chats.py --experiment basic-fs --output ./exports/basic-fs_cases.jsonl
```
"""

from __future__ import annotations

import asyncio
import json
import os
from itertools import groupby
from pathlib import Path
from typing import Annotated, Any, cast

import cyclopts
from dotenv import load_dotenv
from logfire.query_client import AsyncLogfireQueryClient
from loguru import logger
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart

from src.logfire_genai import serialize_model_messages_json, transcript_from_chat_span
from src.report.parse import case_name_from_case_span

load_dotenv()

app = cyclopts.App()

# Logfire `/v1/query` defaults to a small row cap (~100) when ``limit`` is omitted;
# the API also caps ``limit`` at 10_000 per request.
_DEFAULT_QUERY_ROW_LIMIT = 10_000


def _escape_sql_literal(value: str) -> str:
    return value.replace("'", "''")


def _norm_span_id(raw: object | None) -> str | None:
    """Normalize span ids so ``parent_span_id`` lookups match ``span_id`` keys (e.g. dashed UUID)."""
    if raw is None:
        return None
    return str(raw).strip().replace("-", "")


def _span_display_name(row: dict[str, Any]) -> str:
    """Prefer ``span_name``; Logfire sometimes mirrors the label on ``message``."""
    return str(row.get("span_name") or row.get("message") or "")


def _walk_up_to_case_span_id(start_span_id: str, by_id: dict[str, dict[str, Any]]) -> str | None:
    """Follow ``parent_span_id`` until we hit a pydantic-evals ``case: …`` span (same idea as ``report/query``)."""
    sid: str | None = _norm_span_id(start_span_id)
    visited: set[str] = set()
    while sid and sid not in visited:
        visited.add(sid)
        row = by_id.get(sid)
        if not row:
            return None
        name = _span_display_name(row)
        if name.lower().startswith("case:"):
            return sid
        pid = _norm_span_id(row.get("parent_span_id"))
        sid = pid
    return None


def _attributes_mapping(attributes_obj: object) -> dict[str, Any]:
    return attributes_obj if isinstance(attributes_obj, dict) else {}


def _pick_attr(attributes_obj: object, dotted_key: str) -> object | None:
    """Resolve OTel-style flat or nested keys on span ``attributes`` JSON."""
    m = _attributes_mapping(attributes_obj)
    if dotted_key in m:
        return cast("object | None", m[dotted_key])
    parts = dotted_key.split(".")
    cur: object = m
    for part in parts:
        if not isinstance(cur, dict):
            return None
        nxt = cur.get(part)
        if nxt is None:
            return None
        cur = nxt
    return cur


def _sql_resolve_trace_id(*, experiment: str, trace_id: str | None) -> str:
    """Return SQL that yields one ``trace_id`` row for an evaluate run."""
    if trace_id:
        tid = _escape_sql_literal(trace_id)
        return f"""
        SELECT trace_id
        FROM records
        WHERE trace_id = '{tid}'
          AND otel_scope_name = 'pydantic-evals'
          AND message LIKE 'evaluate %'
        LIMIT 1
        """  # noqa: S608
    esc_exp = _escape_sql_literal(experiment)
    return f"""
        SELECT trace_id
        FROM records
        WHERE message = 'evaluate {esc_exp}'
          AND otel_scope_name = 'pydantic-evals'
        LIMIT 1
        """  # noqa: S608


def _sql_all_spans_for_trace(trace_id: str) -> str:
    tid = _escape_sql_literal(trace_id)
    return f"""
    SELECT trace_id, span_id, parent_span_id, span_name, start_timestamp, otel_scope_name, attributes
    FROM records
    WHERE trace_id = '{tid}'
    """  # noqa: S608


def _flatten_chat_rows_from_span_rows(span_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach each pydantic-ai ``chat …`` span to its parent ``case: …`` span via ``parent_span_id``."""
    by_id: dict[str, dict[str, Any]] = {}
    for row in span_rows:
        sid_raw = row.get("span_id")
        sid_k = _norm_span_id(sid_raw)
        if sid_k:
            by_id[sid_k] = row

    out: list[dict[str, Any]] = []
    skipped_no_case = 0
    for row in span_rows:
        span_name = _span_display_name(row)
        if not span_name.lower().startswith("chat "):
            continue
        chat_sid_raw = row.get("span_id")
        chat_sid_k = _norm_span_id(chat_sid_raw)
        if not chat_sid_k:
            continue
        case_sid = _walk_up_to_case_span_id(chat_sid_k, by_id)
        if case_sid is None:
            skipped_no_case += 1
            continue
        case_row = by_id.get(case_sid)
        if not case_row:
            skipped_no_case += 1
            continue
        attrs = row.get("attributes")
        case_attrs = case_row.get("attributes")
        cn = case_name_from_case_span(_span_display_name(case_row), case_attrs)
        task_raw = _pick_attr(case_attrs, "task_name")
        task_name: str | None = task_raw if isinstance(task_raw, str) else None

        out.append(
            {
                "case_name": cn,
                "task_name": task_name,
                "trace_id": row.get("trace_id"),
                "case_span_id": case_sid,
                "case_attributes": case_attrs,
                "chat_span_id": chat_sid_k,
                "chat_start_timestamp": row.get("start_timestamp"),
                "chat_span_message": span_name,
                "input_messages": _pick_attr(attrs, "gen_ai.input.messages"),
                "output_messages": _pick_attr(attrs, "gen_ai.output.messages"),
                "request_model": _pick_attr(attrs, "gen_ai.request.model"),
                "response_model": _pick_attr(attrs, "gen_ai.response.model"),
            }
        )
    if skipped_no_case:
        logger.debug(
            "chat spans skipped (no case ancestor): {} of {} chat-like rows",
            skipped_no_case,
            sum(1 for r in span_rows if _span_display_name(r).lower().startswith("chat ")),
        )
    return out


def _tool_calls_in_messages(messages: list[ModelMessage]) -> list[str]:
    names: list[str] = []
    for msg in messages:
        if isinstance(msg, ModelResponse):
            names.extend(part.tool_name for part in msg.parts if isinstance(part, ToolCallPart) and part.tool_name)
    return names


def _sort_key(row: dict[str, Any]) -> tuple[str, str, str]:
    ts = row.get("chat_start_timestamp")
    ts_str = "" if ts is None else str(ts)
    tid = str(row.get("trace_id") or "")
    csid = str(row.get("case_span_id") or "")
    return (tid, csid, ts_str)


def _build_case_record(
    *,
    experiment: str,
    rows: list[dict[str, Any]],
    include_raw: bool,
) -> dict[str, Any]:
    first = rows[0]
    chats_out: list[dict[str, Any]] = []
    all_tool_calls: list[str] = []

    for row in rows:
        input_raw = row.get("input_messages") or []
        output_raw = row.get("output_messages") or []
        if not isinstance(input_raw, list):
            input_raw = []
        if not isinstance(output_raw, list):
            output_raw = []

        transcript = transcript_from_chat_span(input_raw, output_raw)
        tool_names = _tool_calls_in_messages(transcript)
        all_tool_calls.extend(tool_names)

        chat_obj: dict[str, Any] = {
            "chat_span_id": row.get("chat_span_id"),
            "chat_start_timestamp": row.get("chat_start_timestamp"),
            "chat_span_message": row.get("chat_span_message"),
            "request_model": row.get("request_model"),
            "response_model": row.get("response_model"),
            "tool_calls": tool_names,
            "transcript_messages": serialize_model_messages_json(transcript),
        }
        if include_raw:
            chat_obj["input_messages"] = input_raw
            chat_obj["output_messages"] = output_raw

        chats_out.append(chat_obj)

    case_attrs_json: Any = first.get("case_attributes")

    return {
        "experiment_name": experiment,
        "case_name": first.get("case_name"),
        "task_name": first.get("task_name"),
        "trace_id": first.get("trace_id"),
        "case_span_id": first.get("case_span_id"),
        "case_attributes": case_attrs_json,
        "chat_count": len(chats_out),
        "total_tool_calls": len(all_tool_calls),
        "chats": chats_out,
    }


def _group_rows_to_cases(
    rows: list[dict[str, Any]],
    *,
    experiment: str,
    include_raw: bool,
) -> list[dict[str, Any]]:
    rows_list = list(rows)
    rows_list.sort(key=_sort_key)
    cases: list[dict[str, Any]] = []
    for _key, group in groupby(rows_list, key=lambda r: (r.get("trace_id"), r.get("case_span_id"))):
        batch = list(group)
        if not batch:
            continue
        cases.append(_build_case_record(experiment=experiment, rows=batch, include_raw=include_raw))
    return cases


def _safe_filename_fragment(name: str, *, max_len: int = 100) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return (cleaned[:max_len] if cleaned else "case").strip("_") or "case"


@app.default
async def download(  # noqa: C901, PLR0912, PLR0915
    experiment: Annotated[str, cyclopts.Parameter(help="Experiment name passed to evaluate (e.g. basic-fs).")],
    output: Annotated[
        Path | None,
        cyclopts.Parameter(help="Output file (.json or .jsonl). Default: ./{experiment}_cases.jsonl"),
    ] = None,
    timeout: Annotated[  # noqa: ASYNC109
        int,
        cyclopts.Parameter(help="Logfire query timeout in seconds."),
    ] = 60,
    output_format: Annotated[
        str,
        cyclopts.Parameter(name="--format", help="Output format: jsonl (one case per line) or json (single array)."),
    ] = "jsonl",
    include_raw: Annotated[
        bool,
        cyclopts.Parameter(help="Include raw gen_ai input/output message blobs (larger files).", negative_bool=()),
    ] = True,
    one_file_per_case: Annotated[
        bool,
        cyclopts.Parameter(
            help="Write one JSON file per case under output-dir (implies directory output).",
            negative_bool=(),
        ),
    ] = False,
    output_dir: Annotated[
        Path | None,
        cyclopts.Parameter(help="Directory for per-case files when --one-file-per-case is set."),
    ] = None,
    trace_id: Annotated[
        str | None,
        cyclopts.Parameter(
            help=(
                "Restrict to this evaluate trace_id (matches reports/*.jsonl header). "
                "Required when the Logfire evaluate message differs from --experiment "
                "(e.g. merged label cv-readme-haiku-4-5 vs evaluate ts-fs-repro-oos-cv-…)."
            ),
        ),
    ] = None,
    query_row_limit: Annotated[
        int,
        cyclopts.Parameter(
            help=(
                "Max rows for the trace span query (API max 10_000). Required for parent-span walks."
            ),
        ),
    ] = _DEFAULT_QUERY_ROW_LIMIT,
) -> None:
    """Fetch all chat spans grouped by evaluation case and write to disk."""
    resolve_sql = _sql_resolve_trace_id(experiment=experiment, trace_id=trace_id)

    async with AsyncLogfireQueryClient(
        read_token=os.getenv("LOGFIRE_API_KEY") or "fake",
        timeout=timeout,  # type: ignore[arg-type]
    ) as client:
        tid_json = await client.query_json_rows(sql=resolve_sql, limit=min(query_row_limit, 10_000))
        tid_rows = tid_json.get("rows", [])
        if not isinstance(tid_rows, list) or not tid_rows:
            logger.warning(
                "No evaluate root span for experiment={!r} trace_id={!r}",
                experiment,
                trace_id,
            )
            return
        first_tid = tid_rows[0]
        if not isinstance(first_tid, dict):
            logger.warning("Unexpected trace_id query shape")
            return
        resolved_trace = first_tid.get("trace_id")
        if resolved_trace is None:
            logger.warning("evaluate query returned no trace_id column")
            return
        await asyncio.sleep(12)
        span_json = await client.query_json_rows(
            sql=_sql_all_spans_for_trace(str(resolved_trace)),
            limit=min(query_row_limit, 10_000),
        )
        span_raw = span_json.get("rows", [])
        span_rows = [r for r in span_raw if isinstance(r, dict)]

    rows = _flatten_chat_rows_from_span_rows(span_rows)
    if not rows:
        logger.warning(
            "No chat spans attributed to cases after span-tree walk (experiment={!r} trace_id={!r})",
            experiment,
            trace_id,
        )
        return

    cases = _group_rows_to_cases(rows, experiment=experiment, include_raw=include_raw)
    logger.info("Grouped {} chat rows into {} cases", len(rows), len(cases))

    if one_file_per_case:
        base_dir = output_dir or Path.cwd() / f"{experiment}_case_chats"
        base_dir.mkdir(parents=True, exist_ok=True)
        for case in cases:
            cn = str(case.get("case_name") or "unknown")
            stem = _safe_filename_fragment(cn)
            path = base_dir / f"{stem}.json"
            if path.exists():
                tid = str(case.get("trace_id") or "")[:8]
                path = base_dir / f"{stem}_{tid}.json"
            path.write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.success("Wrote {} case files under {}", len(cases), base_dir.resolve())
        return

    if output is not None:
        out_path = output
    elif trace_id:
        out_path = Path.cwd() / f"{experiment}_cases_{trace_id[:8]}.jsonl"
    else:
        out_path = Path.cwd() / f"{experiment}_cases.jsonl"
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        msg = f"Refusing to overwrite existing file: {out_path}"
        raise FileExistsError(msg)

    fmt = output_format.lower().strip()
    if fmt == "jsonl":
        with out_path.open("w", encoding="utf-8") as f:
            for case in cases:
                f.write(json.dumps(case, ensure_ascii=False))
                f.write("\n")
    elif fmt == "json":
        out_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        msg = f"Unknown format: {output_format!r} (use jsonl or json)"
        raise ValueError(msg)

    logger.success("Wrote {} cases to {}", len(cases), out_path)


if __name__ == "__main__":
    app()
