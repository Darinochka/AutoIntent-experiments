"""Tool-suggest-backed agent, recording, and phase-scoped dependency factories."""

from .agent import create_tool_suggest_agent
from .jsonl_repro import create_jsonl_repo_tool_suggest_deps
from .phase_remote import create_remote_phase_scoped_tool_suggest_deps
from .phase_scoped import create_phase_scoped_tool_suggest_deps
from .recording import tool_suggest_run_result_processor
from .types import EmbBackend, TSAgentState

__all__ = [
    "EmbBackend",
    "TSAgentState",
    "create_jsonl_repo_tool_suggest_deps",
    "create_phase_scoped_tool_suggest_deps",
    "create_remote_phase_scoped_tool_suggest_deps",
    "create_tool_suggest_agent",
    "tool_suggest_run_result_processor",
]
