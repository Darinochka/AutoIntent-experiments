"""Shared deps_maker factory for phase-scoped tool-suggest modes."""

from __future__ import annotations

import uuid
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from mcp_evals.task import Task
    from mcp_evals.types import DepsMaker

    from .types import TSAgentState


def make_phase_deps_maker(phase_deps_ref: list[TSAgentState | None]) -> DepsMaker:
    """Return a DepsMaker that yields the current phase's :class:`TSAgentState`."""

    def deps_maker(_task: Task[Any, Any]) -> AbstractAsyncContextManager[object]:
        @asynccontextmanager
        async def cm() -> AsyncGenerator[object]:
            state = phase_deps_ref[0]
            if state is None:
                msg = "Phase deps not set; start_training must run first."
                raise RuntimeError(msg)
            state.speculations.clear()
            if state.use_suggest_session_tracking:
                state.suggest_session_id = str(uuid.uuid4())
            else:
                state.suggest_session_id = None
            yield state

        return cm()

    return deps_maker
