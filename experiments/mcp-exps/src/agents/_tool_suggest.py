from __future__ import annotations

import re
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, assert_never

import logfire
import tiktoken
from autointent import OptimizationConfig
from autointent.configs import EmbedderConfig, LoggingConfig, OpenaiEmbeddingConfig, SentenceTransformerEmbeddingConfig
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent, RunContext, ToolDefinition
from pydantic_ai.messages import ModelResponse
from tool_suggest import LocalBackendConfig, ToolSuggestClient, ToolSuggestConfig
from tool_suggest.services.embedder import OpenAIEmbedder, SentenceTransformerEmbedder
from tool_suggest.services.formatter import SampleFormatter
from tool_suggest.services.repository import JSONFileRepository
from tool_suggest.services.selector import GreedySelector
from tool_suggest.services.suggester import AutoIntentSuggester

from src.history_processors import truncate_tool_returns
from src.tools import change_output_limit, get_thoughts, record_intermediate_speculations

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable
    from uuid import UUID

    from autointent.configs import EmbedderConfig
    from mcp_evals.task import Task
    from mcp_evals.types import DepsMaker, TrainingTestingCallback
    from mcp_evals.types import RunContext as EvalsContext
    from pydantic_ai.run import AgentRunResult
    from tool_suggest.services.embedder import BaseEmbedder

type EmbBackend = Literal["openai", "st"]


@dataclass
class TSAgentState:
    tool_suggest_client: ToolSuggestClient
    speculations: list[str] = field(default_factory=list)
    tool_return_limit: int = 10_000
    top_k: int | None = None


def _build_embedding_resources(
    *, emb_backend: EmbBackend, emb_model: str
) -> tuple[BaseEmbedder, Callable[[str], int] | None, EmbedderConfig]:
    """Instantiate embedder(s) and token counter based on CLI settings."""
    tool_suggest_embedder: BaseEmbedder
    ai_embedder_config: EmbedderConfig
    if emb_backend == "openai":
        tool_suggest_embedder = OpenAIEmbedder(model=emb_model)
        token_counter = _build_tiktoken_counter(model=emb_model)
        ai_embedder_config = OpenaiEmbeddingConfig(model_name=emb_model)
    elif emb_backend == "st":
        tool_suggest_embedder = SentenceTransformerEmbedder(model_name=emb_model)
        # SampleFormatter token budget is only an approximation; we keep the default fast counter.
        token_counter = None
        ai_embedder_config = SentenceTransformerEmbeddingConfig(model_name=emb_model)
    else:
        assert_never(emb_backend)

    return (
        tool_suggest_embedder,
        token_counter,
        ai_embedder_config,
    )


def create_tool_suggest_agent(model: str) -> Agent[TSAgentState, str]:
    """Create agent that uses tool suggestion service for reducing context size.

    Details:
    - Training data is collected via run result processor (record from result.all_messages()
      with parent_context compression), not event_stream_handler.
    - suggest in Agent.__init__(prepare_tools=...)

    Note:
        Training of tool suggester is to be launched on behalf of an admin,
        that has access to control tool suggester service.

    """
    load_dotenv()
    agent = Agent(
        model,
        system_prompt=(
            "You are an autonomous worker that can use tools to complete tasks. "
            "You must solve the task before sumbitting the final answer (or report that it is not possible). "
            "Don't ask for confirmations; make your own decisions. "
            "After you submit `final_result`, you will be strictly judged and evaluated."
        ),
        tools=[record_intermediate_speculations, get_thoughts, change_output_limit],
        prepare_tools=suggest_tools,
        history_processors=[truncate_tool_returns],
        deps_type=TSAgentState,
        retries=5,
    )

    @agent.instructions
    def current_tool_return_limit(ctx: RunContext[TSAgentState]) -> str:
        return f"Current tool return limit is {ctx.deps.tool_return_limit} (in chars)."

    return agent


async def suggest_tools(ctx: RunContext[TSAgentState], tool_defs: list[ToolDefinition]) -> list[ToolDefinition] | None:
    if not ctx.deps.tool_suggest_client.is_trained:
        logger.debug("Suggester is not trained yet")
        return tool_defs
    with logfire.span("Gettings suggestions from suggester") as span:
        messages = ctx.messages
        suggestions = await ctx.deps.tool_suggest_client.suggest(context=messages, top_k=ctx.deps.top_k)
        names = [s.id for s in suggestions]
        selected = sorted((t for t in tool_defs if t.name in names), key=lambda t: names.index(t.name))
        span.set_attribute("tools_suggested", [s.name for s in selected])
    return selected


def _tool_names_from_response(msg: ModelResponse) -> list[str]:
    """Collect tool names from a ModelResponse's tool-call parts."""
    return [part.tool_name for part in msg.parts if hasattr(part, "tool_name") and part.tool_name]


async def tool_suggest_run_result_processor(_task: Task[Any, Any], result: AgentRunResult[Any], deps: object) -> None:
    """Record tool-suggest samples from result.all_messages() with parent_context."""
    if not isinstance(deps, TSAgentState):
        return
    client = deps.tool_suggest_client
    messages = result.all_messages()
    # Collect (end_index_excl, selected_tools) for each ModelResponse that contains tool calls.
    steps: list[tuple[int, list[str]]] = []
    for i, msg in enumerate(messages):
        if isinstance(msg, ModelResponse):
            tool_names = _tool_names_from_response(msg)
            if tool_names:
                steps.append((i + 1, tool_names))
    if not steps:
        return
    last_sample_id: UUID | None = None
    prev_end = 0
    for end_idx, selected_tools in steps:
        context_slice = messages[prev_end:end_idx]
        if last_sample_id is None:
            sample_id = await client.record(
                context=context_slice,
                selected_tools=selected_tools,
                parent_context=None,
            )
        else:
            sample_id = await client.record(
                context=context_slice,
                selected_tools=selected_tools,
                parent_context=last_sample_id,
            )
        last_sample_id = sample_id
        prev_end = end_idx


def create_phase_scoped_tool_suggest_deps(
    experiment_name: str,
    output_dir: Path,
    formatter_max_len: int,
    selection_target_size: int,
    min_samples_per_tool: int = 3,
    emb_backend: EmbBackend = "openai",
    emb_model: str = "text-embedding-3-small",
    multilabel: bool = False,
    top_k: int | None = None,
) -> tuple[DepsMaker, TrainingTestingCallback, TrainingTestingCallback]:
    """Build phase-scoped deps: same client/repo for all tasks in a training+testing phase.

    Returns (deps_maker, start_training, start_testing). start_training(phase_name) creates
    TSAgentState (repo at output_dir/{phase_name}.json) and stores it in a shared ref; if the
    file exists, the repo loads it (idempotent resume). start_testing(phase_name) calls
    await client.train() so the tool-suggest service is ready for test tasks.
    DepsMaker yields the current phase deps for every task (no per-task collection).
    """

    def _sanitize_phase_name(name: str) -> str:
        """Make phase name safe for use in filenames."""
        return re.sub(r"[^\w\-]", "_", name) or "phase"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedder, token_counter, emb_config = _build_embedding_resources(emb_backend=emb_backend, emb_model=emb_model)
    ai_config = _get_ai_config(
        experiment_name=experiment_name,
        ai_embedder_config=emb_config,
    )

    # Mutable ref holding current phase's TSAgentState (or None before first start_training).
    phase_deps_ref: list[TSAgentState | None] = [None]

    async def start_training(run_context: EvalsContext) -> None:
        phase_name = run_context.phase_name
        logger.info("Collect training samples... (phase={})", phase_name)
        collection_name = _sanitize_phase_name(phase_name)
        file_path = output_dir / f"{collection_name}.json"
        repository = JSONFileRepository(
            collection_name=collection_name,
            file_path=file_path,
        )
        formatter = SampleFormatter(max_len=formatter_max_len, token_counter=token_counter)
        backend_config = LocalBackendConfig(
            repository=repository,
            suggester=AutoIntentSuggester(
                formatter=formatter, multilabel=multilabel, config=ai_config, emergency_toolset="full"
            ),
            selector=GreedySelector(
                embedder=embedder,
                min_samples_per_tool=min_samples_per_tool,
                formatter=formatter,
                min_target_size=selection_target_size,
            ),
        )
        config = ToolSuggestConfig(
            collection_name=collection_name,
            local_backend=backend_config,
        )
        client = ToolSuggestClient(config=config)
        phase_deps_ref[0] = TSAgentState(tool_suggest_client=client, speculations=[], top_k=top_k)
        logger.success("Data collection is set up! (file={})", file_path)

    async def start_testing(run_context: EvalsContext) -> None:
        phase_name = run_context.phase_name
        logger.info("Start training tool suggester on collected data (phase={})", phase_name)
        with logfire.span("Training tool suggester"):
            state = phase_deps_ref[0]
            if state is not None:
                logger.debug("Training...")
                await state.tool_suggest_client.train()
                logger.debug("Trained!")

    def deps_maker(_task: Task[Any, Any]) -> AbstractAsyncContextManager[object]:
        @asynccontextmanager
        async def cm() -> AsyncGenerator[object]:
            state = phase_deps_ref[0]
            if state is None:
                raise RuntimeError("Phase deps not set; start_training must run first.")
            state.speculations.clear()
            yield state

        return cm()

    return (deps_maker, start_training, start_testing)


def create_jsonl_repo_tool_suggest_deps(  # noqa: C901, PLR0915
    experiment_name: str,
    jsonl_path: Path,
    output_dir: Path,
    formatter_max_len: int,
    selection_target_size: int,
    emb_backend: EmbBackend = "openai",
    emb_model: str = "text-embedding-3-small",
    min_samples_per_tool: int = 3,
    multilabel: bool = False,
    top_k: int | None = None,
) -> tuple[DepsMaker, TrainingTestingCallback, TrainingTestingCallback]:
    """Build tool-suggest deps from an existing JSONL repository.

    Initializes the repository from a JSONL file. In `start_testing`, it filters
    samples by `case_name` from the `run_context.phase_to_tasks` and trains the suggester.
    """

    def _sanitize_phase_name(name: str) -> str:
        return re.sub(r"[^\w\-]", "_", name) or "phase"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedder, token_counter, emb_config = _build_embedding_resources(emb_backend=emb_backend, emb_model=emb_model)
    ai_config = _get_ai_config(
        experiment_name=experiment_name,
        ai_embedder_config=emb_config,
    )
    is_already_trained = (ai_config.logging_config.dirpath / experiment_name).exists()

    phase_deps_ref: list[TSAgentState | None] = [None]

    async def start_training(run_context: EvalsContext) -> None:
        phase_name = run_context.phase_name
        logger.info("Skipping setup in start_training for JSONL replay mode (phase={})", phase_name)

    async def start_testing(run_context: EvalsContext) -> None:
        phase_name = run_context.phase_name
        logger.info("Preparing filtered JSONL repo and training suggester (phase={})", phase_name)
        collection_name = _sanitize_phase_name(phase_name)
        formatter = SampleFormatter(max_len=formatter_max_len, token_counter=token_counter)

        source_repo = JSONFileRepository(
            collection_name=f"{collection_name}_source",
            file_path=jsonl_path,
        )
        filtered_file_path = output_dir / f"{collection_name}.jsonl"
        if filtered_file_path.exists() and not is_already_trained:
            raise FileExistsError

        dest_repo = JSONFileRepository(
            collection_name=collection_name,
            file_path=filtered_file_path,
        )

        if not is_already_trained:
            train_tasks = run_context.get_training_tasks()
            train_case_names = {task.name for task in train_tasks}
            logger.debug("Filtering by case names: {}", train_case_names)

            copied_count = 0
            async for batch in source_repo.get_batches(batch_size=64, resolve_links=False):
                filtered_batch = [s for s in batch if s.data.get("case_name", "") in train_case_names]
                if filtered_batch:
                    await dest_repo.add_bulk(filtered_batch)
                    copied_count += len(filtered_batch)

            logger.info("Copied {} samples into filtered repo: {}", copied_count, filtered_file_path)
        else:
            logger.debug("Skipped repo filtering")

        backend_config = LocalBackendConfig(
            repository=dest_repo,
            suggester=AutoIntentSuggester(
                formatter=formatter, multilabel=multilabel, config=ai_config, emergency_toolset="full"
            ),
            selector=GreedySelector(
                embedder=embedder,
                formatter=formatter,
                min_samples_per_tool=min_samples_per_tool,
                min_target_size=selection_target_size,
            ),
        )
        config = ToolSuggestConfig(
            collection_name=collection_name,
            local_backend=backend_config,
        )
        client = ToolSuggestClient(config=config)
        phase_deps_ref[0] = TSAgentState(tool_suggest_client=client, speculations=[], top_k=top_k)

        if not is_already_trained:
            with logfire.span("Training tool suggester"):
                logger.debug("Training...")
                await client.train()
                logger.debug("Trained!")
        else:
            logger.debug("Skipped training")

    def deps_maker(_task: Task[Any, Any]) -> AbstractAsyncContextManager[object]:
        @asynccontextmanager
        async def cm() -> AsyncGenerator[object]:
            state = phase_deps_ref[0]
            if state is None:
                raise RuntimeError("Phase deps not set; start_training must run first.")
            state.speculations.clear()
            yield state

        return cm()

    return (deps_maker, start_training, start_testing)


def _get_ai_config(experiment_name: str, *, ai_embedder_config: EmbedderConfig) -> OptimizationConfig:
    ai_config = OptimizationConfig.from_preset("classic-light")
    ai_config.logging_config = LoggingConfig(
        dump_modules=True, clear_ram=True, project_dir="./.autointent_runs", run_name=experiment_name
    )
    ai_config.embedder_config = ai_embedder_config
    return ai_config


def _build_tiktoken_counter(model: str) -> Callable[[str], int]:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    def counter(text: str) -> int:
        return len(encoding.encode(text))

    return counter
