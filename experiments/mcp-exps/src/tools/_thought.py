"""Tool for intermediate thoughts."""

from typing import Protocol

from pydantic_ai import RunContext


class StateWithSpeculations(Protocol):
    speculations: list[str]


def record_intermediate_speculations(ctx: RunContext[StateWithSpeculations], thought: str) -> str:
    """Record intermediate speculations.

    This function is intended to use by AI agents to help them better understand current context.
    It is not necessary to use it after each step.

    Args:
       ctx: agent run context
       thought: the thoughts to record.

    Return:
        "Thought is recorded" message
    """
    ctx.deps.speculations.append(thought)
    return "Thought is recorded"


def get_thoughts(ctx: RunContext[StateWithSpeculations], n: int = 10) -> list[str]:
    """Return the last n recorded speculations (most recent last). Use to review your previous reasoning."""
    if n <= 0:
        return []
    return ctx.deps.speculations[-n:]
