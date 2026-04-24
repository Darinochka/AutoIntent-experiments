r"""Run MCP-evals domain tasks with baseline and tool-suggest agents.

Subcommands:
- `basic`: baseline agent with plain task grouping.
- `ts`: tool-suggest mode with hold-out / cross-validation grouping.
- `ts-repro`: tool-suggest reproduction mode backed by an existing JSONL repo.
- `ts-remote`: tool-suggest against a running HTTP service (see tool-suggest README).

Usage examples:
```bash
# Show full CLI help
uv run run_exp.py --help

# Baseline run on filesystem domain
uv run run_exp.py basic \
    --domain fs \
    --experiment-name basic-fs-smoke \
    --model openai:gpt-4.1

# Tool-suggest with hold-out split
uv run run_exp.py ts \
    --domain pg \
    --experiment-name ts-pg-ho \
    --grouper ho \
    --ho-ratio 0.2

# Tool-suggest with 5-fold CV and bounded task count
uv run run_exp.py ts \
    --domain fs \
    --experiment-name ts-fs-cv \
    --grouper cv \
    --cv-splits 5 \
    --max-tasks 20

# Reproduce from an existing JSONL tool repository
uv run run_exp.py ts-repro \
    --domain fs \
    --experiment-name repro-fs \
    --jsonl-repo path/to/repo.jsonl

# Tool-suggest with a remote server (suggester/repo configured on the server)
uv run run_exp.py ts-remote \
    --domain fs \
    --experiment-name ts-remote-smoke \
    --service-url http://127.0.0.1:8000 \
    --grouper ho \
    --top-k 8
```

Environment variables:
- DOWNLOAD_PROXY: URL for proxy used to load setup data.
- LLM provider-specific credentials, for example:
    - OPENAI_API_KEY and OPENAI_BASE_URL
    - OPENROUTER_API_KEY
    - etc. See pydantic-ai model docs: https://ai.pydantic.dev/models/overview/
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import logfire
from cyclopts import App, Parameter
from loguru import logger
from mcp_evals import CVGrouper, Domain, DomainRunner, Grouper, HoldOutGrouper, PlainGrouper
from pydantic_ai import UsageLimits

from src.agents import (
    EmbBackend,
    create_basic_agent,
    create_basic_deps_maker,
    create_jsonl_repo_tool_suggest_deps,
    create_phase_scoped_tool_suggest_deps,
    create_remote_phase_scoped_tool_suggest_deps,
    create_tool_suggest_agent,
    tool_suggest_run_result_processor,
)

if TYPE_CHECKING:
    from mcp_evals.types import TrainingTestingCallback
    from pydantic_ai import Agent


app = App(
    help="Run MCP-evals domain tasks.",
    help_epilogue=__doc__,
)


@dataclass(frozen=True)
class BasicArgs:
    """Shared CLI arguments for all experiment modes."""

    domain_key: Annotated[
        Literal["pg", "fs"],
        Parameter(name="--domain", help="Domain to run tasks from: 'pg' or 'fs'."),
    ]
    experiment_name: Annotated[
        str,
        Parameter(help="Experiment name. Use it to differentiate runs."),
    ]
    model: Annotated[
        str,
        Parameter(help="Model name to use (default: 'openai:gpt-4.1')."),
    ] = "openai:gpt-4.1"
    max_tasks: Annotated[
        int | None,
        Parameter(help="Limit number of tasks to run."),
    ] = None
    tool_retries: Annotated[
        int,
        Parameter(help="Tool call retries within each task."),
    ] = 3
    max_self_correction_retries: Annotated[
        int,
        Parameter(help="Max self-correction retries when self-correction is enabled."),
    ] = 3
    self_correction: Annotated[
        bool,
        Parameter(help="Enable self-correction.", negative_bool=()),
    ] = False
    max_concurrency: Annotated[
        int,
        Parameter(help="Maximum parallel tasks to run."),
    ] = 1
    request_limit: Annotated[
        int,
        Parameter(help="Maximum number of LLM requests per task."),
    ] = 50


@dataclass(frozen=True, kw_only=True)
class TSArgs(BasicArgs):
    """CLI arguments for tool-suggest mode."""

    grouper_kind: Annotated[
        Literal["ho", "cv"],
        Parameter(name="--grouper", help="Grouper: 'ho' (hold-out) or 'cv' (cross-validation)."),
    ] = "ho"
    ho_ratio: Annotated[float, Parameter(help="Hold-out test ratio. Used when --grouper ho.")] = 0.2
    cv_splits: Annotated[int, Parameter(help="Number of CV folds. Used when --grouper cv.")] = 5
    random_state: Annotated[int | None, Parameter(help="Random state for train/test splits.")] = None
    repos_dir: Annotated[
        Path,
        Parameter(help="Directory for persistent tool-suggest JSON repositories."),
    ] = Path("tool_suggest_repos")
    selection_target_size: Annotated[
        int | None,
        Parameter(help="Lower bound for number of samples used for AutoIntent training."),
    ] = 100
    tool_samples: Annotated[
        int,
        Parameter(help="Lower bound for number of samples per tool in AutoIntent training."),
    ] = 4
    formatter_max_len: Annotated[
        int,
        Parameter(help="Maximum number of tokens for a single training sample."),
    ] = 1000
    multilabel: Annotated[
        bool,
        Parameter(help="Whether to use AutoIntent in multilabel mode"),
    ] = False
    top_k: Annotated[
        int | None,
        Parameter(help="Maximum number of tools to suggest per step (like top-k retrieval)."),
    ]
    emb_backend: Annotated[
        EmbBackend,
        Parameter(help="Which embedding backend to use for core-set selection and for AutoIntent classifier."),
    ] = "openai"
    emb_model: Annotated[
        str,
        Parameter(help="Name of embedding model, e.g. 'Qwen/Qwen3-Embedding-0.6B' or 'text-embedding-3-small'."),
    ] = "text-embedding-3-small"
    max_oos: Annotated[
        float,
        Parameter(help="Maximum fraction of OOS samples to feed to AutoIntentSuggester compared to in-domain data."),
    ] = 0.2
    suggest_session_tracking: Annotated[
        bool,
        Parameter(
            help=(
                "Use suggest session tracking: merge tools from earlier prepare_tools steps "
                "into each suggestion (requires session_id support on client/server)."
            ),
            negative_bool=(),
        ),
    ] = False


@dataclass(frozen=True, kw_only=True)
class TSRemoteArgs(BasicArgs):
    """CLI arguments for remote HTTP tool-suggest (server owns repo + ML)."""

    grouper_kind: Annotated[
        Literal["ho", "cv"],
        Parameter(name="--grouper", help="Grouper: 'ho' (hold-out) or 'cv' (cross-validation)."),
    ] = "ho"
    ho_ratio: Annotated[float, Parameter(help="Hold-out test ratio. Used when --grouper ho.")] = 0.2
    cv_splits: Annotated[int, Parameter(help="Number of CV folds. Used when --grouper cv.")] = 5
    random_state: Annotated[int | None, Parameter(help="Random state for train/test splits.")] = None
    service_url: Annotated[
        str,
        Parameter(help="Base URL of the ToolSuggest HTTP API (e.g. http://127.0.0.1:8000)."),
    ]
    top_k: Annotated[
        int | None,
        Parameter(help="Maximum number of tools to suggest per step (like top-k retrieval)."),
    ]
    suggest_session_tracking: Annotated[
        bool,
        Parameter(
            help=("Use suggest session tracking (remote server merges prior-step tools when session_id is sent)."),
            negative_bool=(),
        ),
    ] = False


@dataclass(frozen=True, kw_only=True)
class TSReproArgs(TSArgs):
    """CLI arguments for JSONL-backed tool-suggest reproduction mode."""

    jsonl_repo: Annotated[Path, Parameter(help="Path to the JSONL repository to reproduce from.")]


@app.command(name="basic", help="Run baseline basic agent.")
def basic(args: Annotated[BasicArgs, Parameter(name="*")]) -> None:
    """Run baseline basic agent."""
    _run(args, mode="basic")


@app.command(name="ts", help="Run tool-suggest mode.")
def ts(args: Annotated[TSArgs, Parameter(name="*")]) -> None:
    """Run tool-suggest mode."""
    _run(args, mode="ts")


@app.command(name="ts-repro", help="Run tool-suggest in JSONL reproduction mode.")
def ts_repro(args: Annotated[TSReproArgs, Parameter(name="*")]) -> None:
    """Run tool-suggest mode using an existing JSONL repository."""
    _run(args, mode="ts-repro")


@app.command(name="ts-remote", help="Run tool-suggest against a remote HTTP service.")
def ts_remote(args: Annotated[TSRemoteArgs, Parameter(name="*")]) -> None:
    """Run tool-suggest with collections and training on a ToolSuggest server."""
    _run(args, mode="ts-remote")


def _run(cfg: BasicArgs, *, mode: Literal["basic", "ts", "ts-repro", "ts-remote"]) -> None:
    _init_logfire()
    agent_obj = _build_agent(mode, cfg.model)
    domain_obj = _build_domain(cfg.domain_key, cfg.tool_retries)
    grouper_obj = _build_grouper(mode, cfg)
    deps_maker, start_training_cb, start_testing_cb = _build_deps(mode, cfg)

    is_repro_mode = mode == "ts-repro"
    run_result_processor = tool_suggest_run_result_processor if mode in ("ts", "ts-repro", "ts-remote") else None
    runner = DomainRunner(
        agent=agent_obj,
        grouper=grouper_obj,
        deps_maker=deps_maker,
        use_self_correction=cfg.self_correction,
        max_self_correction_retries=cfg.max_self_correction_retries,
        start_training=start_training_cb,
        start_testing=start_testing_cb,
        run_result_processor=run_result_processor,
        max_tasks=cfg.max_tasks,
        usage_limits=UsageLimits(request_limit=cfg.request_limit),
        rerun_start_training_on_resume=True,
        rerun_start_testing_on_resume=is_repro_mode,
        max_concurrency=cfg.max_concurrency,
        skip_training_tasks=is_repro_mode,
    )

    logger.info(f"Running {cfg.domain_key} tasks with model: {cfg.model}")

    async def run() -> None:
        report = await runner.run(domain_obj, experiment_name=cfg.experiment_name)
        logger.info(f"\nDomain: {cfg.domain_key}")
        logger.info(f"Total tasks: {len(report.cases)}")

        for case in report.cases:
            passed = all(eval_res.value == 1.0 for eval_res in case.scores.values())
            if passed:
                logger.success(f"Task {case.name} passed")
            else:
                logger.warning(f"Task {case.name} failed")

    asyncio.run(run())


def _init_logfire() -> None:
    logfire.configure(send_to_logfire="if-token-present", scrubbing=False)
    logfire.instrument_pydantic_ai()


def _build_domain(domain_key: Literal["pg", "fs"], tool_retries: int) -> Domain[Any]:
    if domain_key == "pg":
        from mcp_evals.contrib.postgres import PostgresDomain  # noqa: PLC0415

        return PostgresDomain(tool_retries=tool_retries)
    from mcp_evals.contrib.filesystem import FilesystemDomain  # noqa: PLC0415

    return FilesystemDomain(tool_retries=tool_retries)


def _build_agent(mode: Literal["basic", "ts", "ts-repro", "ts-remote"], model: str) -> "Agent[Any, Any]":
    if mode == "basic":
        return create_basic_agent(model=model)
    logger.debug("Creating ts agent")
    return create_tool_suggest_agent(model=model)


def _build_grouper(mode: Literal["basic", "ts", "ts-repro", "ts-remote"], cfg: BasicArgs) -> Grouper:
    if mode == "basic":
        return PlainGrouper()
    if mode == "ts-remote":
        if not isinstance(cfg, TSRemoteArgs):
            msg = "ts-remote mode requires TSRemoteArgs"
            raise TypeError(msg)
        return _tool_suggest_grouper(cfg)
    if not isinstance(cfg, TSArgs):
        msg = "TS mode requires TSArgs"
        raise TypeError(msg)
    return _tool_suggest_grouper(cfg)


def _tool_suggest_grouper(cfg: TSArgs | TSRemoteArgs) -> Grouper:
    if cfg.grouper_kind == "ho":
        return HoldOutGrouper(test_ratio=cfg.ho_ratio, random_state=cfg.random_state)
    return CVGrouper(n_splits=cfg.cv_splits, random_state=cfg.random_state)


def _build_deps(
    mode: Literal["basic", "ts", "ts-repro", "ts-remote"], cfg: BasicArgs
) -> tuple[Any, "TrainingTestingCallback | None", "TrainingTestingCallback | None"]:
    if mode == "basic":
        return create_basic_deps_maker(), None, None
    if mode == "ts":
        if not isinstance(cfg, TSArgs):
            raise TypeError("ts mode requires TSArgs")
        return create_phase_scoped_tool_suggest_deps(
            cfg.experiment_name,
            cfg.repos_dir / cfg.experiment_name,
            multilabel=cfg.multilabel,
            max_oos_fraction=cfg.max_oos,
            formatter_max_len=cfg.formatter_max_len,
            selection_target_size=cfg.selection_target_size,
            min_samples_per_tool=cfg.tool_samples,
            top_k=cfg.top_k,
            emb_backend=cfg.emb_backend,
            emb_model=cfg.emb_model,
            suggest_session_tracking=cfg.suggest_session_tracking,
        )
    if mode == "ts-remote":
        if not isinstance(cfg, TSRemoteArgs):
            raise TypeError("ts-remote mode requires TSRemoteArgs")
        return create_remote_phase_scoped_tool_suggest_deps(
            experiment_name=cfg.experiment_name,
            service_url=cfg.service_url,
            top_k=cfg.top_k,
            suggest_session_tracking=cfg.suggest_session_tracking,
        )
    if not isinstance(cfg, TSReproArgs):
        raise TypeError("ts-repro mode requires TSReproArgs")
    return create_jsonl_repo_tool_suggest_deps(
        experiment_name=cfg.experiment_name,
        jsonl_path=cfg.jsonl_repo,
        output_dir=cfg.repos_dir / cfg.experiment_name,
        multilabel=cfg.multilabel,
        max_oos_fraction=cfg.max_oos,
        formatter_max_len=cfg.formatter_max_len,
        selection_target_size=cfg.selection_target_size,
        min_samples_per_tool=cfg.tool_samples,
        top_k=cfg.top_k,
        emb_backend=cfg.emb_backend,
        emb_model=cfg.emb_model,
        suggest_session_tracking=cfg.suggest_session_tracking,
    )


if __name__ == "__main__":
    app()
