"""Phase-scoped deps with local JSON repo + AutoIntent + greedy selector."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import logfire
from loguru import logger
from tool_suggest import LocalBackendConfig, ToolSuggestClient, ToolSuggestConfig
from tool_suggest.services.formatter import SampleFormatter
from tool_suggest.services.repository import JSONFileRepository
from tool_suggest.services.selector import GreedySelector
from tool_suggest.services.session_memory import InMemorySessionMemory
from tool_suggest.services.suggester import AutoIntentSuggester

from .embedding import build_embedding_resources, get_ai_config
from .phase_deps import make_phase_deps_maker
from .phase_names import sanitize_phase_name
from .types import EmbBackend, TSAgentState

if TYPE_CHECKING:
    from mcp_evals.types import DepsMaker, TrainingTestingCallback
    from mcp_evals.types import RunContext as EvalsContext


def create_phase_scoped_tool_suggest_deps(  # noqa: PLR0913
    experiment_name: str,
    output_dir: Path,
    formatter_max_len: int,
    max_oos_fraction: float,
    selection_target_size: int | None = None,
    min_samples_per_tool: int = 3,
    emb_backend: EmbBackend = "openai",
    emb_model: str = "text-embedding-3-small",
    multilabel: bool = False,
    top_k: int | None = None,
    *,
    suggest_session_tracking: bool = False,
) -> tuple[DepsMaker, TrainingTestingCallback, TrainingTestingCallback]:
    """Build phase-scoped deps: same client/repo for all tasks in a training+testing phase.

    Returns (deps_maker, start_training, start_testing). start_training(phase_name) creates
    TSAgentState (repo at output_dir/{phase_name}.json) and stores it in a shared ref; if the
    file exists, the repo loads it (idempotent resume). start_testing(phase_name) calls
    await client.train() so the tool-suggest service is ready for test tasks.
    DepsMaker yields the current phase deps for every task (no per-task collection).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedder, token_counter, emb_config = build_embedding_resources(emb_backend=emb_backend, emb_model=emb_model)
    ai_config = get_ai_config(
        experiment_name=experiment_name,
        ai_embedder_config=emb_config,
    )

    phase_deps_ref: list[TSAgentState | None] = [None]

    async def start_training(run_context: EvalsContext) -> None:
        phase_name = run_context.phase_name
        logger.info("Collect training samples... (phase={})", phase_name)
        collection_name = sanitize_phase_name(phase_name)
        file_path = output_dir / f"{collection_name}.json"
        repository = JSONFileRepository(
            collection_name=collection_name,
            file_path=file_path,
        )
        formatter = SampleFormatter(max_len=formatter_max_len, token_counter=token_counter)
        session_memory = InMemorySessionMemory() if suggest_session_tracking else None
        backend_config = LocalBackendConfig(
            repository=repository,
            suggester=AutoIntentSuggester(
                formatter=formatter,
                multilabel=multilabel,
                config=ai_config,
                emergency_toolset="full",
                under_represented_behavior="emergency_only",
                max_oos_fraction=max_oos_fraction,
            ),
            selector=GreedySelector(
                embedder=embedder,
                min_samples_per_tool=min_samples_per_tool,
                formatter=formatter,
                min_target_size=selection_target_size,
            ),
            session_memory=session_memory,
        )
        config = ToolSuggestConfig(
            collection_name=collection_name,
            local_backend=backend_config,
        )
        client = ToolSuggestClient(config=config)
        phase_deps_ref[0] = TSAgentState(
            tool_suggest_client=client,
            speculations=[],
            top_k=top_k,
            use_suggest_session_tracking=suggest_session_tracking,
        )
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

    return (make_phase_deps_maker(phase_deps_ref), start_training, start_testing)
