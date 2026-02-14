from dotenv import load_dotenv
from pydantic_ai import Agent

from src.history_processors import truncate_tool_returns
from src.tools import intermediate_speculations


def create_basic_agent(model: str = "openai:gpt-4.1") -> Agent[None, str]:
    load_dotenv()
    return Agent(
        model,
        system_prompt=(
            "You are a helpful assistant that can use tools to complete tasks. "
            "You can provide text messages beside the final answer as a means of "
            "intermediate speculations and reasoning."
        ),
        tools=[intermediate_speculations],
        history_processors=[truncate_tool_returns],
    )
