"""Tool-suggest deps built from an existing JSONL repository (reproduction mode)."""

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

from .constants import IGNORED_TOOL_LABELS
from .embedding import build_embedding_resources, get_ai_config
from .phase_deps import make_phase_deps_maker
from .phase_names import sanitize_phase_name
from .types import EmbBackend, TSAgentState

if TYPE_CHECKING:
    from mcp_evals.types import DepsMaker, TrainingTestingCallback
    from mcp_evals.types import RunContext as EvalsContext


def create_jsonl_repo_tool_suggest_deps(  # noqa: PLR0913
    experiment_name: str,
    jsonl_path: Path,
    output_dir: Path,
    formatter_max_len: int,
    max_oos_fraction: float,
    selection_target_size: int | None = None,
    emb_backend: EmbBackend = "openai",
    emb_model: str = "text-embedding-3-small",
    min_samples_per_tool: int = 3,
    multilabel: bool = False,
    top_k: int | None = None,
    *,
    suggest_session_tracking: bool = False,
) -> tuple[DepsMaker, TrainingTestingCallback, TrainingTestingCallback]:
    """Build tool-suggest deps from an existing JSONL repository.

    Initializes the repository from a JSONL file. In `start_testing`, it filters
    samples by `case_name` from the `run_context.phase_to_tasks` and trains the suggester.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedder, token_counter, emb_config = build_embedding_resources(emb_backend=emb_backend, emb_model=emb_model)
    ai_config = get_ai_config(
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
        collection_name = sanitize_phase_name(phase_name)
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
                filtered_batch = []
                for s in batch:
                    if s.data.get("case_name", "") not in train_case_names:
                        continue

                    tools = [t for t in s.tools if t not in IGNORED_TOOL_LABELS]
                    filtered_batch.append(
                        s.model_copy(
                            update={
                                "tools": tools,
                                "is_out_of_scope": not tools,
                            }
                        )
                    )
                if filtered_batch:
                    await dest_repo.add_bulk(filtered_batch)
                    copied_count += len(filtered_batch)

            logger.info("Copied {} samples into filtered repo: {}", copied_count, filtered_file_path)
        else:
            logger.debug("Skipped repo filtering")

        session_memory = InMemorySessionMemory() if suggest_session_tracking else None
        backend_config = LocalBackendConfig(
            repository=dest_repo,
            suggester=AutoIntentSuggester(
                formatter=formatter,
                multilabel=multilabel,
                config=ai_config,
                emergency_toolset="full",
                under_represented_behavior="always_include",
                max_oos_fraction=max_oos_fraction,
            ),
            selector=GreedySelector(
                embedder=embedder,
                formatter=formatter,
                min_samples_per_tool=min_samples_per_tool,
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

        if not is_already_trained:
            with logfire.span("Training tool suggester"):
                logger.debug("Training...")
                await client.train()
                logger.debug("Trained!")
        else:
            logger.debug("Skipped training")

    return (make_phase_deps_maker(phase_deps_ref), start_training, start_testing)
