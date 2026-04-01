r"""Run MCP-evals domain tasks using an OpenAI-compatible API.

The CLI is split into subcommands:
- `basic`: basic agent with plain task grouping.
- `ts`: tool-suggest mode with hold-out / cross-validation grouping.
- `ts-repro`: tool-suggest reproduction mode backed by an existing JSONL repo.

Environment Variables:
    DOWNLOAD_PROXY: URL for proxy used for loading setup data.
    LLM provider-specific:
        - OPENAI_API_KEY and OPENAI_BASE_URL
        - OPENROUTER_API_KEY
        - etc, see pydantic ai docs: https://ai.pydantic.dev/models/overview/
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
    create_basic_agent,
    create_basic_deps_maker,
    create_jsonl_repo_tool_suggest_deps,
    create_phase_scoped_tool_suggest_deps,
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
class CommonConfig:
    domain_key: Literal["pg", "fs"]
    experiment_name: str
    model: str = "openai:gpt-4.1"
    max_tasks: int | None = None
    tool_retries: int = 3
    max_self_correction_retries: int = 3
    self_correction: bool = False
    max_concurrency: int = 1


@dataclass(frozen=True)
class TSConfig(CommonConfig):
    grouper_kind: Literal["ho", "cv"] = "ho"
    ho_ratio: float = 0.2
    cv_splits: int = 5
    random_state: int | None = None
    repos_dir: Path = Path("tool_suggest_repos")
    selection_target_size: int = 100
    formatter_max_len: int = 1000


@dataclass(frozen=True)
class TSReproConfig(TSConfig):
    jsonl_repo: Path = Path()


@app.command(name="basic", help="Run baseline basic agent.")
def basic(
    *,
    domain_key: Annotated[
        Literal["pg", "fs"],
        Parameter(name="--domain", help="Domain to run tasks from: 'pg' or 'fs'."),
    ],
    experiment_name: Annotated[str, Parameter(help="Experiment name. Use it to differentiate runs.")],
    model: Annotated[str, Parameter(help="Model name to use (default: 'openai:gpt-4.1').")] = "openai:gpt-4.1",
    max_tasks: Annotated[int | None, Parameter(help="Limit number of tasks to run.")] = None,
    tool_retries: Annotated[int, Parameter(help="Tool call retries within each task.")] = 3,
    max_self_correction_retries: Annotated[
        int, Parameter(help="Max self-correction retries when self-correction is enabled.")
    ] = 3,
    self_correction: Annotated[
        bool,
        Parameter(help="Enable self-correction.", negative_bool=()),
    ] = False,
    max_concurrency: Annotated[int, Parameter(help="Maximum parallel tasks to run.")] = 1,
) -> None:
    _run(
        CommonConfig(
            domain_key=domain_key,
            experiment_name=experiment_name,
            model=model,
            max_tasks=max_tasks,
            tool_retries=tool_retries,
            max_self_correction_retries=max_self_correction_retries,
            self_correction=self_correction,
            max_concurrency=max_concurrency,
        ),
        mode="basic",
    )


@app.command(name="ts", help="Run tool-suggest mode.")
def ts(
    *,
    domain_key: Annotated[
        Literal["pg", "fs"],
        Parameter(name="--domain", help="Domain to run tasks from: 'pg' or 'fs'."),
    ],
    experiment_name: Annotated[str, Parameter(help="Experiment name. Use it to differentiate runs.")],
    model: Annotated[str, Parameter(help="Model name to use (default: 'openai:gpt-4.1').")] = "openai:gpt-4.1",
    grouper_kind: Annotated[
        Literal["ho", "cv"],
        Parameter(name="--grouper", help="Grouper: 'ho' (hold-out) or 'cv' (cross-validation)."),
    ] = "ho",
    ho_ratio: Annotated[float, Parameter(help="Hold-out test ratio. Used when --grouper ho.")] = 0.2,
    cv_splits: Annotated[int, Parameter(help="Number of CV folds. Used when --grouper cv.")] = 5,
    random_state: Annotated[int | None, Parameter(help="Random state for train/test splits.")] = None,
    repos_dir: Annotated[
        Path,
        Parameter(help="Directory for persistent tool-suggest JSON repositories."),
    ] = Path("tool_suggest_repos"),
    max_tasks: Annotated[int | None, Parameter(help="Limit number of tasks to run.")] = None,
    tool_retries: Annotated[int, Parameter(help="Tool call retries within each task.")] = 3,
    max_self_correction_retries: Annotated[
        int, Parameter(help="Max self-correction retries when self-correction is enabled.")
    ] = 3,
    self_correction: Annotated[
        bool,
        Parameter(help="Enable self-correction.", negative_bool=()),
    ] = False,
    max_concurrency: Annotated[int, Parameter(help="Maximum parallel tasks to run.")] = 1,
    selection_target_size: Annotated[
        int,
        Parameter(help="Upper bound for number of samples used for AutoIntent training."),
    ] = 100,
    formatter_max_len: Annotated[
        int,
        Parameter(help="Maximum number of tokens for a single training sample."),
    ] = 1000,
) -> None:
    _run(
        TSConfig(
            domain_key=domain_key,
            experiment_name=experiment_name,
            model=model,
            max_tasks=max_tasks,
            tool_retries=tool_retries,
            max_self_correction_retries=max_self_correction_retries,
            self_correction=self_correction,
            max_concurrency=max_concurrency,
            grouper_kind=grouper_kind,
            ho_ratio=ho_ratio,
            cv_splits=cv_splits,
            random_state=random_state,
            repos_dir=repos_dir,
            selection_target_size=selection_target_size,
            formatter_max_len=formatter_max_len,
        ),
        mode="ts",
    )


@app.command(name="ts-repro", help="Run tool-suggest in JSONL reproduction mode.")
def ts_repro(
    *,
    domain_key: Annotated[
        Literal["pg", "fs"],
        Parameter(name="--domain", help="Domain to run tasks from: 'pg' or 'fs'."),
    ],
    experiment_name: Annotated[str, Parameter(help="Experiment name. Use it to differentiate runs.")],
    jsonl_repo: Annotated[Path, Parameter(help="Path to the JSONL repository to reproduce from.")],
    model: Annotated[str, Parameter(help="Model name to use (default: 'openai:gpt-4.1').")] = "openai:gpt-4.1",
    grouper_kind: Annotated[
        Literal["ho", "cv"],
        Parameter(name="--grouper", help="Grouper: 'ho' (hold-out) or 'cv' (cross-validation)."),
    ] = "ho",
    ho_ratio: Annotated[float, Parameter(help="Hold-out test ratio. Used when --grouper ho.")] = 0.2,
    cv_splits: Annotated[int, Parameter(help="Number of CV folds. Used when --grouper cv.")] = 5,
    random_state: Annotated[int | None, Parameter(help="Random state for train/test splits.")] = None,
    repos_dir: Annotated[
        Path,
        Parameter(help="Directory for persistent tool-suggest JSON repositories."),
    ] = Path("tool_suggest_repos"),
    max_tasks: Annotated[int | None, Parameter(help="Limit number of tasks to run.")] = None,
    tool_retries: Annotated[int, Parameter(help="Tool call retries within each task.")] = 3,
    max_self_correction_retries: Annotated[
        int, Parameter(help="Max self-correction retries when self-correction is enabled.")
    ] = 3,
    self_correction: Annotated[
        bool,
        Parameter(help="Enable self-correction.", negative_bool=()),
    ] = False,
    max_concurrency: Annotated[int, Parameter(help="Maximum parallel tasks to run.")] = 1,
    selection_target_size: Annotated[
        int,
        Parameter(help="Upper bound for number of samples used for AutoIntent training."),
    ] = 100,
    formatter_max_len: Annotated[
        int,
        Parameter(help="Maximum number of tokens for a single training sample."),
    ] = 1000,
) -> None:
    _run(
        TSReproConfig(
            domain_key=domain_key,
            experiment_name=experiment_name,
            model=model,
            max_tasks=max_tasks,
            tool_retries=tool_retries,
            max_self_correction_retries=max_self_correction_retries,
            self_correction=self_correction,
            max_concurrency=max_concurrency,
            grouper_kind=grouper_kind,
            ho_ratio=ho_ratio,
            cv_splits=cv_splits,
            random_state=random_state,
            repos_dir=repos_dir,
            selection_target_size=selection_target_size,
            formatter_max_len=formatter_max_len,
            jsonl_repo=jsonl_repo,
        ),
        mode="ts-repro",
    )


def _run(cfg: CommonConfig, *, mode: Literal["basic", "ts", "ts-repro"]) -> None:
    _init_logfire()
    agent_obj = _build_agent(mode, cfg.model)
    domain_obj = _build_domain(cfg.domain_key, cfg.tool_retries)
    grouper_obj = _build_grouper(mode, cfg)
    deps_maker, start_training_cb, start_testing_cb = _build_deps(mode, cfg)

    is_repro_mode = mode == "ts-repro"
    run_result_processor = tool_suggest_run_result_processor if mode != "basic" else None
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
        usage_limits=UsageLimits(request_limit=50),
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


def _build_agent(mode: Literal["basic", "ts", "ts-repro"], model: str) -> "Agent[Any, Any]":
    if mode == "basic":
        return create_basic_agent(model=model)
    logger.debug("Creating ts agent")
    return create_tool_suggest_agent(model=model)


def _build_grouper(mode: Literal["basic", "ts", "ts-repro"], cfg: CommonConfig) -> Grouper:
    if mode == "basic":
        return PlainGrouper()
    if not isinstance(cfg, TSConfig):
        raise TypeError("TS mode requires TSConfig")
    if cfg.grouper_kind == "ho":
        return HoldOutGrouper(test_ratio=cfg.ho_ratio, random_state=cfg.random_state)
    return CVGrouper(n_splits=cfg.cv_splits, random_state=cfg.random_state)


def _build_deps(
    mode: Literal["basic", "ts", "ts-repro"], cfg: CommonConfig
) -> tuple[Any, "TrainingTestingCallback | None", "TrainingTestingCallback | None"]:
    if mode == "basic":
        return create_basic_deps_maker(), None, None
    if mode == "ts":
        if not isinstance(cfg, TSConfig):
            raise TypeError("ts mode requires TSConfig")
        return create_phase_scoped_tool_suggest_deps(
            cfg.repos_dir / cfg.experiment_name,
            multilabel=True,
            formatter_max_len=cfg.formatter_max_len,
            selection_target_size=cfg.selection_target_size,
        )
    if not isinstance(cfg, TSReproConfig):
        raise TypeError("ts-repro mode requires TSReproConfig")
    return create_jsonl_repo_tool_suggest_deps(
        experiment_name=cfg.experiment_name,
        jsonl_path=cfg.jsonl_repo,
        output_dir=cfg.repos_dir / cfg.experiment_name,
        multilabel=True,
        formatter_max_len=cfg.formatter_max_len,
        selection_target_size=cfg.selection_target_size,
    )


if __name__ == "__main__":
    app()
