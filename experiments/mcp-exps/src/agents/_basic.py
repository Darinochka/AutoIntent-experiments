from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from mcp_evals.task import Task
from mcp_evals.types import DepsMaker
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from src.agents.worker_capabilities import WORKER_SYSTEM_PROMPT, shared_worker_capabilities


class BasicAgentState(BaseModel):
    speculations: list[str] = Field(default_factory=list)
    tool_return_limit: int = 10_000


def create_basic_agent(model: str = "openai:gpt-4.1") -> Agent[BasicAgentState, str]:
    load_dotenv()
    return Agent(
        model,
        system_prompt=WORKER_SYSTEM_PROMPT,
        capabilities=shared_worker_capabilities(),
        deps_type=BasicAgentState,
    )


def create_basic_deps_maker() -> DepsMaker:
    """Deps maker for basic agent: fresh AgentState (empty speculations) per task."""

    def deps_maker(_task: Task[Any, Any]) -> AbstractAsyncContextManager[BasicAgentState]:
        @asynccontextmanager
        async def cm() -> AsyncGenerator[BasicAgentState]:
            yield BasicAgentState()

        return cm()

    return deps_maker
