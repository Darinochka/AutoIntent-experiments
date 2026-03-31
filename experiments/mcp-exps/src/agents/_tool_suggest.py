from __future__ import annotations

import re
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from uuid import UUID

    from mcp_evals.task import Task
    from mcp_evals.types import DepsMaker, TrainingTestingCallback
    from mcp_evals.types import RunContext as EvalsContext
    from pydantic_ai.run import AgentRunResult

import logfire
from autointent import OptimizationConfig
from autointent.configs import LoggingConfig
from dotenv import load_dotenv
from loguru import logger
from pydantic import ValidationError
from pydantic_ai import Agent, RunContext, ToolDefinition
from pydantic_ai.messages import ModelResponse
from tool_suggest import LocalBackendConfig, ToolSuggestClient, ToolSuggestConfig
from tool_suggest.models import Sample
from tool_suggest.services.embedder import SentenceTransformerEmbedder
from tool_suggest.services.formatter import SampleFormatter
from tool_suggest.services.repository import JSONFileRepository
from tool_suggest.services.selector import GreedySelector
from tool_suggest.services.suggester import AutoIntentSuggester

from src.history_processors import truncate_tool_returns
from src.tools import change_output_limit, get_thoughts, record_intermediate_speculations


@dataclass
class TSAgentState:
    tool_suggest_client: ToolSuggestClient
    speculations: list[str] = field(default_factory=list)
    tool_return_limit: int = 10_000


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
        suggestions = await ctx.deps.tool_suggest_client.suggest(context=messages)
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
    output_dir: Path,
    multilabel: bool = False,
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
    embedder = SentenceTransformerEmbedder(device="mps")

    ai_config = OptimizationConfig.from_preset("classic-light")
    ai_config.logging_config = LoggingConfig(dump_modules=True, clear_ram=True)

    # Mutable ref holding current phase's TSAgentState (or None before first start_training).
    phase_deps_ref: list[TSAgentState | None] = [None]

    async def start_training(phase_name: str) -> None:
        logger.info("Collect training samples... (phase={})", phase_name)
        collection_name = _sanitize_phase_name(phase_name)
        file_path = output_dir / f"{collection_name}.json"
        repository = JSONFileRepository(
            collection_name=collection_name,
            file_path=file_path,
        )
        formatter = SampleFormatter(max_len=1000)
        backend_config = LocalBackendConfig(
            repository=repository,
            suggester=AutoIntentSuggester(formatter=formatter, multilabel=multilabel, config=ai_config),
            selector=GreedySelector(embedder=embedder, formatter=formatter, target_size=15),  # NOTE: test value
        )
        config = ToolSuggestConfig(
            collection_name=collection_name,
            local_backend=backend_config,
        )
        client = ToolSuggestClient(config=config)
        phase_deps_ref[0] = TSAgentState(tool_suggest_client=client, speculations=[])
        logger.success("Data collection is set up! (file={})", file_path)

    async def start_testing(phase_name: str) -> None:
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


def create_jsonl_repo_tool_suggest_deps(  # noqa: C901
    jsonl_path: Path,
    output_dir: Path,
    multilabel: bool = False,
) -> tuple[DepsMaker, TrainingTestingCallback, TrainingTestingCallback]:
    """Build tool-suggest deps from an existing JSONL repository.

    Initializes the repository from a JSONL file. In `start_testing`, it filters
    samples by `case_name` from the `run_context.phase_to_tasks` and trains the suggester.
    """

    def _sanitize_phase_name(name: str) -> str:
        return re.sub(r"[^\w\-]", "_", name) or "phase"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedder = SentenceTransformerEmbedder(device="mps")

    ai_config = OptimizationConfig.from_preset("classic-light")
    ai_config.logging_config = LoggingConfig(dump_modules=True, clear_ram=True)

    phase_deps_ref: list[TSAgentState | None] = [None]

    async def start_training(phase_name: str, _: RunContext) -> None:
        logger.info("Setting up tool-suggest from JSONL... (phase={})", phase_name)
        collection_name = _sanitize_phase_name(phase_name)
        # We use a temporary file for the phase-specific repository
        file_path = output_dir / f"{collection_name}.json"
        repository = JSONFileRepository(
            collection_name=collection_name,
            file_path=file_path,
        )
        formatter = SampleFormatter(max_len=1000)
        backend_config = LocalBackendConfig(
            repository=repository,
            suggester=AutoIntentSuggester(formatter=formatter, multilabel=multilabel, config=ai_config),
            selector=GreedySelector(embedder=embedder, formatter=formatter, target_size=15),
        )
        config = ToolSuggestConfig(
            collection_name=collection_name,
            local_backend=backend_config,
        )
        client = ToolSuggestClient(config=config)
        phase_deps_ref[0] = TSAgentState(tool_suggest_client=client, speculations=[])

    async def start_testing(phase_name: str, run_context: EvalsContext) -> None:
        logger.info("Loading and filtering samples from JSONL (phase={})", phase_name)
        state = phase_deps_ref[0]
        if state is None:
            logger.error("Phase deps not set; start_training must run first.")
            return

        # 1. Get the list of task names (case_names) for the current training phase
        train_tasks = run_context["phase_to_tasks"].get(phase_name, [])
        train_case_names = {task.name for task in train_tasks}
        logger.debug("Training on tasks: {}", train_case_names)

        # 2. Load samples from JSONL and filter by case_name
        filtered_samples: list[Sample] = []
        with jsonl_path.open("r") as f:
            for line in f:
                try:
                    sample = Sample.model_validate_json(line)
                except ValidationError as e:
                    logger.warning("Failed to parse line from JSONL: {}", e)
                else:
                    if sample.data.get("case_name", "") in train_case_names:
                        filtered_samples.append(sample)

        if not filtered_samples:
            logger.warning("No samples found in JSONL for the current training tasks.")
        else:
            logger.info("Found {} samples for training.", len(filtered_samples))
            # 3. Inject these samples into the repository
            # repository.add_samples is not directly available on client, but on backend
            await state.tool_suggest_client.backend.repository.add_samples(filtered_samples)

        # 4. Train the suggester
        with logfire.span("Training tool suggester"):
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
