# ruff: noqa: D103, D101, D100
import asyncio

import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, ToolOutput
from pydantic_ai.models.openrouter import OpenRouterModelSettings

from src.history_processors import truncate_tool_returns
from src.tools import change_output_limit, get_thoughts, record_intermediate_speculations

settings = OpenRouterModelSettings(
    openrouter_reasoning={
        "effort": "none",
    },
    openrouter_usage={
        "include": True,
    },
)


class BasicAgentState(BaseModel):
    speculations: list[str] = Field(default_factory=list)
    tool_return_limit: int = 10_000


class Finish(BaseModel):
    success: bool


async def main() -> None:
    agent = create_basic_agent(model="openrouter:qwen/qwen3.5-397b-a17b")
    deps = BasicAgentState()
    logfire.configure(send_to_logfire="if-token-present", scrubbing=False)
    logfire.instrument_pydantic_ai()
    result = await agent.run(
        "Write a few brief thoughts about what makes a good debugging workflow.",
        deps=deps,
    )
    logger.info(result.output)
    logger.info(deps.__repr__())


def create_basic_agent(model: str = "openai:gpt-4.1") -> Agent[BasicAgentState, Finish]:
    load_dotenv()
    agent = Agent(
        model,
        system_prompt=(
            "You are an autonomous worker that can use tools to complete tasks. "
            "You must solve the task before sumbitting the final answer (or report that it is not possible). "
            "Don't ask for confirmations; make your own decisions. "
            "After you submit `finish`, you will be strictly judged and evaluated."
        ),
        tools=[record_intermediate_speculations, get_thoughts, change_output_limit],
        history_processors=[truncate_tool_returns],
        deps_type=BasicAgentState,
        output_type=ToolOutput(Finish, name="finish"),
        model_settings=settings,
    )

    @agent.instructions
    def current_tool_return_limit(ctx: RunContext[BasicAgentState]) -> str:
        return f"Current tool return limit is {ctx.deps.tool_return_limit} (in chars)."

    return agent


if __name__ == "__main__":
    asyncio.run(main())
