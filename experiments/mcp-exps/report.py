"""Download experiment cases.

Load total cost, tokens and per-case results.
"""

import os
from pathlib import Path
from typing import Annotated

import cyclopts
from dotenv import load_dotenv
from logfire.query_client import AsyncLogfireQueryClient
from loguru import logger

load_dotenv()

app = cyclopts.App(
    # name="pipeline",
    # help="ETL pipeline for offers reranker: load, parse, and split data",
)


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
    """Load raw data from Logfire and save to JSONL file."""
    output_path = ((output_dir or Path.cwd()).resolve() / experiment).with_suffix(".jsonl")
    logger.info(f"Result will be saved to {output_path}")

    query = f"""
    SELECT
        parent.trace_id,
        parent.attributes AS parent_attributes,
        child.attributes AS child_attributes
    FROM records parent
    JOIN records child
        ON child.trace_id = parent.trace_id
    WHERE
        parent.message = 'evaluate {experiment}'
        AND parent.otel_scope_name = 'pydantic-evals'
        AND child.otel_scope_name = 'pydantic-evals'
        AND child.span_name ILIKE 'case: %'
    """  # noqa: S608

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        async with AsyncLogfireQueryClient(
            read_token=os.getenv("LOGFIRE_API_KEY") or "fake",
            timeout=timeout,  # type: ignore[arg-type]
        ) as client:
            json_rows = await client.query_json_rows(sql=query)

    logger.success("Done!")


if __name__ == "__main__":
    app()
