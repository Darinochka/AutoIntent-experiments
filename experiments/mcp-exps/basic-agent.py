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
import re
import secrets
from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import Any

import logfire
from loguru import logger
from mcp_evals import BenchmarkRunner, Domain
from mcp_evals.task import Task
from mcp_evals.types import DepsMaker
from tool_suggest import LocalBackendConfig, ToolSuggestClient, ToolSuggestConfig
from tool_suggest.embedder import SentenceTransformerEmbedder
from tool_suggest.repository import JSONFileRepository
from tool_suggest.selector import GreedySelector
from tool_suggest.suggester import KNNSuggester

from src.agents import AgentState, create_basic_agent, create_tool_suggest_agent


def create_tool_suggest_deps_maker(
    output_dir: Path,
) -> DepsMaker:
    """Build a DepsMaker that creates AgentState with ToolSuggestClient per task.

    Collection name and file path are derived from task name + random suffix.
    The embedder is reused for all tasks. JSON repository files are left on disk.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedder = SentenceTransformerEmbedder(device="mps")

    def deps_maker(task: Task[Any, Any]) -> AbstractAsyncContextManager[object]:
        @asynccontextmanager
        async def cm() -> AsyncGenerator[AgentState]:
            base = re.sub(r"[^a-zA-Z0-9_-]", "_", task.name)
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
            yield AgentState(tool_suggest_client=client)

        return cm()

    return deps_maker


def main() -> None:
    """Run all filesystem tasks with OpenAI."""
    parser = argparse.ArgumentParser(
        description="Run filesystem tasks with OpenAI-compatible API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
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

    args = parser.parse_args()

    logfire.configure(send_to_logfire="if-token-present")
    logfire.instrument_pydantic_ai()

    # Create agent with custom base URL
    model = f"openai:{args.model}"

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

    runner = BenchmarkRunner(agent=agent, domains=[domain])

    deps_maker = None
    if args.agent == "ts":
        deps_maker = create_tool_suggest_deps_maker(args.repos_dir)

    logger.info(f"Running {args.domain} tasks with model: {args.model}")

    # Run benchmark
    async def run() -> None:
        reports = await runner.run(
            experiment_name=args.experiment_name,
            deps_maker=deps_maker,
        )
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
