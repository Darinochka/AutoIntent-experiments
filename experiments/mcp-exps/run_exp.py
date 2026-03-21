r"""Run MCP-evals domain tasks using an OpenAI-compatible API.

This script runs tasks from an MCP-evals "domain" (e.g. filesystem or postgres) by constructing a
`DomainRunner` with either:
- a basic agent (`--agent basic`), or
- a tool-suggest / TS agent (`--agent ts`).

Prerequisites:
    Install filesystem domain dependencies:
        uv sync --extra domain-filesystem
    Or with pip:
        pip install 'mcp-evals[domain-filesystem]'
    (Optional) Extra for postgres tasks: `domain-postgres`.

Usage:
    # Environment variables (typical for OpenAI-compatible endpoints)
    export OPENAI_API_KEY="your-api-key"
    export OPENAI_BASE_URL="https://your-custom-endpoint.com/v1"

    # Run (file system domain)
    uv run run_exp.py --domain fs --experiment-name <name> --agent basic

    # Specify a different model (model strings are passed through to pydantic-ai)
    uv run run_exp.py --domain fs --experiment-name <name> --agent basic --model "openai:gpt-4o-mini"

Environment Variables:
    OPENAI_API_KEY: API key for the OpenAI-compatible endpoint.
    OPENAI_BASE_URL: Base URL for the OpenAI-compatible API (must include `/v1`).
    DOWNLOAD_PROXY: URL for proxy used for loading setup data.
"""

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import logfire
from cyclopts import App, Parameter
from loguru import logger
from mcp_evals import CVGrouper, Domain, DomainRunner, Grouper, HoldOutGrouper, PlainGrouper
from pydantic_ai import UsageLimits

from src.agents import (
    create_basic_agent,
    create_basic_deps_maker,
    create_phase_scoped_tool_suggest_deps,
    create_tool_suggest_agent,
    tool_suggest_run_result_processor,
)

if TYPE_CHECKING:
    from mcp_evals.types import TrainingTestingCallback
    from pydantic_ai import Agent


app = App(
    help="Run MCP-evals domain tasks using an OpenAI-compatible API.",
    help_epilogue=__doc__,
)


@app.default
def main(  # noqa: C901, PLR0913
    *,
    model: Annotated[
        str,
        Parameter(
            help="Model name to use (overrides OPENAI_MODEL env var or 'openai:gpt-4.1' default).",
        ),
    ] = "openai:gpt-4.1",
    domain_key: Annotated[
        Literal["pg", "fs"],
        Parameter(
            name="--domain",
            help="Domain to run tasks from: 'pg' (postgres) or 'fs' (file system).",
        ),
    ],
    experiment_name: Annotated[
        str,
        Parameter(help="Experiment name. Use it to differentiate runs."),
    ],
    agent_variant: Annotated[
        Literal["basic", "ts"],
        Parameter(
            name="--agent",
            help="Agent variant: 'basic' or 'ts' (tool-suggest).",
        ),
    ],
    repos_dir: Annotated[
        Path,
        Parameter(
            help="Directory for persistent tool-suggest JSON repositories (default: tool_suggest_repos).",
        ),
    ] = Path("tool_suggest_repos"),
    grouper_kind: Annotated[
        Literal["plain", "ho", "cv"],
        Parameter(
            name="--grouper",
            help=(
                "Grouper: 'ho' (hold-out) or 'cv' (cross-validation). "
                "ho/cv are for ts only. Ignored when --agent basic."
            ),
        ),
    ] = "plain",
    ho_ratio: Annotated[
        float,
        Parameter(help="Hold-out test ratio (default: 0.2). Used when --agent ts --grouper ho."),
    ] = 0.2,
    cv_splits: Annotated[
        int,
        Parameter(help="Number of CV folds (default: 5). Used when --agent ts --grouper cv."),
    ] = 5,
    random_state: Annotated[
        int | None,
        Parameter(help="Random state for train/test splits. Used when --agent ts."),
    ] = None,
    max_tasks: Annotated[
        int | None,
        Parameter(help="Limit number of tasks to run (useful for smoke tests)."),
    ] = None,
    tool_retries: Annotated[
        int,
        Parameter(help="Tool call retries within each task (default: 3)."),
    ] = 3,
    max_self_correction_retries: Annotated[
        int,
        Parameter(help="Max self-correction retries when --self-correction is enabled (default: 3)."),
    ] = 3,
    self_correction: Annotated[
        bool,
        Parameter(
            help="Enable self-correction.",
            negative_bool=(),
        ),
    ] = False,
) -> None:
    """Run all filesystem tasks with OpenAI."""
    logfire.configure(send_to_logfire="if-token-present", scrubbing=False)
    logfire.instrument_pydantic_ai()

    # Create agent with custom base URL

    agent_obj: Agent[Any, Any]
    if agent_variant == "basic":
        agent_obj = create_basic_agent(model=model)
    elif agent_variant == "ts":
        logger.debug("Creating ts agent")
        agent_obj = create_tool_suggest_agent(model=model)

    # Create domain and runner
    domain_obj: Domain[Any]
    if domain_key == "pg":
        from mcp_evals.contrib.postgres import PostgresDomain  # noqa: PLC0415

        domain_obj = PostgresDomain(tool_retries=tool_retries)
    elif domain_key == "fs":
        from mcp_evals.contrib.filesystem import FilesystemDomain  # noqa: PLC0415

        domain_obj = FilesystemDomain(tool_retries=tool_retries)

    deps_maker = None
    start_training_cb: TrainingTestingCallback | None = None
    start_testing_cb: TrainingTestingCallback | None = None

    grouper_obj: Grouper
    if agent_variant == "basic":
        grouper_obj = PlainGrouper()
        deps_maker = create_basic_deps_maker()
    elif agent_variant == "ts":
        if grouper_kind == "plain":
            raise ValueError("ts agent doesnt support plain grouper")
        grouper_obj = (
            HoldOutGrouper(test_ratio=ho_ratio, random_state=random_state)
            if grouper_kind == "ho"
            else CVGrouper(n_splits=cv_splits, random_state=random_state)
        )
        deps_maker, start_training_cb, start_testing_cb = create_phase_scoped_tool_suggest_deps(
            repos_dir / experiment_name,
            multilabel=True,
        )

    run_result_processor = tool_suggest_run_result_processor if agent_variant == "ts" else None
    runner = DomainRunner(
        agent=agent_obj,
        grouper=grouper_obj,
        deps_maker=deps_maker,
        use_self_correction=self_correction,
        max_self_correction_retries=max_self_correction_retries,
        start_training=start_training_cb,
        start_testing=start_testing_cb,
        run_result_processor=run_result_processor,
        max_tasks=max_tasks,
        usage_limits=UsageLimits(request_limit=10),
        rerun_start_training_on_resume=True,
    )

    logger.info(f"Running {domain_key} tasks with model: {model}")

    # Run benchmark
    async def run() -> None:
        report = await runner.run(domain_obj, experiment_name=experiment_name)
        logger.info(f"\nDomain: {domain_key}")
        logger.info(f"Total tasks: {len(report.cases)}")

        report.print(include_reasons=True, include_output=True)
        for case in report.cases:
            passed = all(eval_res.value == 1.0 for eval_res in case.scores.values())
            if passed:
                logger.success(f"Task {case.name} passed")
            else:
                logger.warning(f"Task {case.name} failed")

    asyncio.run(run())


if __name__ == "__main__":
    app()
