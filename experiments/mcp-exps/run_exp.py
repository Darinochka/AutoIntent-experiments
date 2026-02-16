r"""Script to run domain tasks with OpenAI.

This script runs single domain's tasks from MCP Universe and MCPMark using an OpenAI-compatible
API endpoint. It uses the mcp-evals library to execute tasks and evaluate results.

Prerequisites:
    Install filesystem domain dependencies:
        uv sync --extra domain-filesystem
    Or with pip:
        pip install 'mcp-evals[domain-filesystem]'
    Extra for postgres tasks: 'domain-postgres'

Usage:
    # Set OpenAI API key and base URL via environment variables
    export OPENAI_API_KEY="your-api-key"
    export OPENAI_BASE_URL="https://your-custom-endpoint.com/v1"
    uv run python scripts/run_domain_tasks.py

    # Or use command line arguments
    uv run python scripts/run_domain_tasks.py

    # Specify a different model
    uv run python scripts/run_domain_tasks.py --model "gpt-4o-mini"

Environment Variables:
    OPENAI_API_KEY: Your OpenAI API key (required if not provided via --api-key)
    OPENAI_BASE_URL: Custom base URL for OpenAI-compatible API (required if not provided via --base-url)
    OPENAI_MODEL: Model name to use (defaults to "gpt-4o" if not provided)
    DOWNLOAD_PROXY: URL for proxy used for loading setup data

Examples:
    # Run with default settings (gpt-4o)
    uv run python scripts/run_domain_tasks.py

    # Run with a specific model
    OPENAI_MODEL="gpt-4o-mini" uv run python scripts/run_domain_tasks.py

    # Run with custom endpoint
    OPENAI_BASE_URL="http://localhost:8000/v1" \\
    OPENAI_API_KEY="dummy-key" \\
    uv run python scripts/run_domain_tasks.py
"""

import argparse
import asyncio
import secrets
from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import logfire
from loguru import logger
from mcp_evals import BenchmarkRunner, Domain
from mcp_evals.task import Task
from mcp_evals.types import DepsMaker, Runner, TrainingTestingCallback
from tool_suggest import LocalBackendConfig, ToolSuggestClient, ToolSuggestConfig
from tool_suggest.embedder import SentenceTransformerEmbedder
from tool_suggest.repository import JSONFileRepository
from tool_suggest.selector import GreedySelector
from tool_suggest.suggester import KNNSuggester

from src.agents import AgentState, create_basic_agent, create_tool_suggest_agent

if TYPE_CHECKING:
    from pydantic_ai import Agent


def create_phase_scoped_tool_suggest_deps(
    output_dir: Path,
) -> tuple[DepsMaker, TrainingTestingCallback, TrainingTestingCallback]:
    """Build phase-scoped deps: same client/repo for all tasks in a training+testing phase.

    Returns (deps_maker, start_training, start_testing). start_training creates fresh
    AgentState (new collection/repo/client) and stores it in a shared ref; start_testing
    calls await client.train() so the tool-suggest service is ready for test tasks.
    DepsMaker yields the current phase deps for every task (no per-task collection).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedder = SentenceTransformerEmbedder(device="mps")

    # Mutable ref holding current phase's AgentState (or None before first start_training).
    phase_deps_ref: list[AgentState | None] = [None]

    async def start_training() -> None:
        base = "phase"
        suffix = secrets.token_hex(4)
        collection_name = f"{base}_{suffix}"
        file_path = output_dir / f"{collection_name}.json"
        repository = JSONFileRepository(
            collection_name=collection_name,
            file_path=file_path,
        )
        backend_config = LocalBackendConfig(
            repository=repository,
            suggester=KNNSuggester(embedder=embedder),
            selector=GreedySelector(embedder=embedder, target_size=15),  # NOTE: test value
        )
        config = ToolSuggestConfig(
            collection_name=collection_name,
            local_backend=backend_config,
        )
        client = ToolSuggestClient(config=config)
        phase_deps_ref[0] = AgentState(tool_suggest_client=client)

    async def start_testing() -> None:
        state = phase_deps_ref[0]
        if state is not None:
            await state.tool_suggest_client.train()

    def deps_maker(_task: Task[Any, Any]) -> AbstractAsyncContextManager[object]:
        @asynccontextmanager
        async def cm() -> AsyncGenerator[object]:
            state = phase_deps_ref[0]
            if state is None:
                raise RuntimeError("Phase deps not set; start_training must run first.")
            yield state

        return cm()

    return (deps_maker, start_training, start_testing)


def main() -> None:  # noqa: PLR0915
    """Run all filesystem tasks with OpenAI."""
    parser = argparse.ArgumentParser(
        description="Run filesystem tasks with OpenAI-compatible API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to use (overrides OPENAI_MODEL env var or 'gpt-4.1' default)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["pg", "fs"],
        required=True,
        help="Domain to run tasks from. Available: 'pg' (postgres), 'fs' (file system).",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name. Use it to differentiate runs.",
    )
    parser.add_argument(
        "--agent",
        type=str,
        choices=["basic", "ts"],
        help="Agent variant: 'basic' or 'ts' (tool-suggest).",
    )
    parser.add_argument(
        "--repos-dir",
        type=Path,
        default=Path("tool_suggest_repos"),
        help="Directory for persistent tool-suggest JSON repositories (default: tool_suggest_repos).",
    )
    parser.add_argument(
        "--runner",
        type=str,
        choices=["ho", "cv"],
        default="ho",
        help="Runner for ts agent: 'ho' (hold-out) or 'cv' (cross-validation). Ignored when --agent basic.",
    )
    parser.add_argument(
        "--ho-ratio",
        type=float,
        default=0.2,
        help="Hold-out test ratio (default: 0.2). Used when --agent ts --runner ho.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of CV folds (default: 5). Used when --agent ts --runner cv.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Random state for train/test splits. Used when --agent ts.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Limit number of tasks to run (useful for smoke tests).",
    )

    args = parser.parse_args()

    logfire.configure(send_to_logfire="if-token-present")
    logfire.instrument_pydantic_ai()

    # Create agent with custom base URL
    model = f"openai:{args.model}"

    agent: Agent[Any, Any]
    if args.agent == "basic":
        agent = create_basic_agent(model=model)
    elif args.agent == "ts":
        agent = create_tool_suggest_agent(model=model)

    # Create domain and runner
    domain: Domain[Any]
    if args.domain == "pg":
        from mcp_evals.contrib.postgres import PostgresDomain  # noqa: PLC0415

        domain = PostgresDomain()
    elif args.domain == "fs":
        from mcp_evals.contrib.filesystem import FilesystemDomain  # noqa: PLC0415

        domain = FilesystemDomain()

    deps_maker = None
    start_training_cb: TrainingTestingCallback | None = None
    start_testing_cb: TrainingTestingCallback | None = None
    runner_type = Runner.INFERENCE_ONLY

    if args.agent == "basic":
        runner_type = Runner.INFERENCE_ONLY
    elif args.agent == "ts":
        runner_type = Runner.HOLD_OUT if args.runner == "ho" else Runner.CROSS_VALIDATION
        deps_maker, start_training_cb, start_testing_cb = create_phase_scoped_tool_suggest_deps(args.repos_dir)

    runner = BenchmarkRunner(
        agent=agent,
        domains=[domain],
        runner=runner_type,
        deps_maker=deps_maker,
        experiment_name=args.experiment_name,
        hold_out_test_ratio=args.ho_ratio,
        cv_n_splits=args.cv_splits,
        random_state=args.random_state,
        start_training=start_training_cb,
        start_testing=start_testing_cb,
        max_tasks=args.max_tasks,
    )

    logger.info(f"Running {args.domain} tasks with model: {args.model}")

    # Run benchmark
    async def run() -> None:
        reports = await runner.run()
        report = reports[0]
        logger.info(f"\nDomain: {args.domain}")
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
    main()
