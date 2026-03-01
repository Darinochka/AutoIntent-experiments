from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from mcp_evals.task import Task
from mcp_evals.types import DepsMaker
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from src.history_processors import truncate_tool_returns
from src.tools import change_output_limit, get_thoughts, record_intermediate_speculations


class BasicAgentState(BaseModel):
    speculations: list[str] = Field(default_factory=list)
    tool_return_limit: int = 10_000


def create_basic_agent(model: str = "openai:gpt-4.1") -> Agent[BasicAgentState, str]:
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
        history_processors=[truncate_tool_returns],
        deps_type=BasicAgentState,
    )

    @agent.instructions
    def current_tool_return_limit(ctx: RunContext[BasicAgentState]) -> str:
        return f"Current tool return limit is {ctx.deps.tool_return_limit} (in chars)."

    return agent


def create_basic_deps_maker() -> DepsMaker:
    """Deps maker for basic agent: fresh AgentState (empty speculations) per task."""

    def deps_maker(_task: Task[Any, Any]) -> AbstractAsyncContextManager[BasicAgentState]:
        @asynccontextmanager
        async def cm() -> AsyncGenerator[BasicAgentState]:
            yield BasicAgentState()

        return cm()

    return deps_maker
