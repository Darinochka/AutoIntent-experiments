"""Collection of various agents."""

from ._basic import BasicAgentState, create_basic_agent, create_basic_deps_maker
from ._tool_suggest import (
    TSAgentState,
    create_jsonl_repo_tool_suggest_deps,
    create_phase_scoped_tool_suggest_deps,
    create_tool_suggest_agent,
    tool_suggest_run_result_processor,
)

__all__ = [
    "BasicAgentState",
    "TSAgentState",
    "create_basic_agent",
    "create_basic_deps_maker",
    "create_jsonl_repo_tool_suggest_deps",
    "create_phase_scoped_tool_suggest_deps",
    "create_tool_suggest_agent",
    "tool_suggest_run_result_processor",
]
