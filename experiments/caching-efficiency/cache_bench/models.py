"""Pydantic output model and message builders for the structured-output cache scenarios.

The output model lives in an importable module on purpose: ``PydanticModelDumper.load``
re-imports the class by its ``module``/``name``, so disk round-trips only work for models
that can be imported by path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from autointent.generation.chat_templates import Message


class IntentPrediction(BaseModel):
    """A representative structured LLM output: a predicted intent with rationale."""

    label: str
    confidence: float
    rationale: str


def build_messages(user_text: str, *, padding: int = 0) -> list[Message]:
    """Build a realistic 2-message chat payload.

    Args:
        user_text: The user turn content.
        padding: Extra characters appended to the system prompt to grow the payload size
            (used to study how key-hashing cost scales with message size).

    Returns:
        A list of ``Message`` dicts (JSON-serializable).
    """
    system = "You are an intent classifier. Respond with the intent label, confidence and rationale."
    if padding:
        system = system + " " + ("x" * padding)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ]


def make_prediction(seed: int) -> IntentPrediction:
    """Build a deterministic prediction result for a given seed."""
    return IntentPrediction(
        label=f"intent_{seed % 50}",
        confidence=0.5 + (seed % 50) / 100.0,
        rationale=f"matched pattern #{seed}",
    )
