"""Phase-scoped deps talking to a remote ToolSuggest HTTP service."""

from __future__ import annotations

from typing import TYPE_CHECKING

import logfire
from loguru import logger
from tool_suggest import ToolSuggestClient, ToolSuggestConfig

from .phase_deps import make_phase_deps_maker
from .phase_names import sanitize_phase_name
from .types import TSAgentState

if TYPE_CHECKING:
    from mcp_evals.types import DepsMaker, TrainingTestingCallback
    from mcp_evals.types import RunContext as EvalsContext


def create_remote_phase_scoped_tool_suggest_deps(
    service_url: str,
    top_k: int | None = None,
) -> tuple[DepsMaker, TrainingTestingCallback, TrainingTestingCallback]:
    """Same phase lifecycle as local mode, but storage and ML run on ``service_url``.

    Each eval phase uses ``sanitize_phase_name(phase_name)`` as the remote collection name.
    Suggester/selector/repository types are whatever the server was configured with.
    """
    base_url = service_url.rstrip("/")
    phase_deps_ref: list[TSAgentState | None] = [None]

    async def start_training(run_context: EvalsContext) -> None:
        phase_name = run_context.phase_name
        collection_name = sanitize_phase_name(phase_name)
        logger.info(
            "Remote tool-suggest: collect training samples (phase={}, collection={})",
            phase_name,
            collection_name,
        )
        config = ToolSuggestConfig(collection_name=collection_name, service_url=base_url)
        client = ToolSuggestClient(config=config)
        phase_deps_ref[0] = TSAgentState(tool_suggest_client=client, speculations=[], top_k=top_k)
        logger.success(
            "Remote data collection client ready (service_url={}, collection={})",
            base_url,
            collection_name,
        )

    async def start_testing(run_context: EvalsContext) -> None:
        phase_name = run_context.phase_name
        logger.info("Remote tool-suggest: train suggester (phase={})", phase_name)
        with logfire.span("Training tool suggester (remote)"):
            state = phase_deps_ref[0]
            if state is not None:
                logger.debug("Training via HTTP...")
                await state.tool_suggest_client.train()
                logger.debug("Trained!")

    return (make_phase_deps_maker(phase_deps_ref), start_training, start_testing)
