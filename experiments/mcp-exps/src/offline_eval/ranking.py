"""Full tool rankings for :class:`KNNSuggester` and :class:`AutoIntentSuggester` (offline eval)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, TypeGuard

from tool_suggest.services.suggester.autointent import AutoIntentSuggester
from tool_suggest.services.suggester.knn import KNNSuggester

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai.messages import ModelMessage
    from tool_suggest.services.suggester.base import BaseSuggester


def _is_autointent(s: BaseSuggester) -> TypeGuard[AutoIntentSuggester]:
    return isinstance(s, AutoIntentSuggester)


def _is_knn(s: BaseSuggester) -> TypeGuard[KNNSuggester]:
    return isinstance(s, KNNSuggester)


async def full_ranked_tool_ids(
    suggester: BaseSuggester,
    context: Sequence[ModelMessage],
) -> list[str]:
    """Return all tools in descending score order (best first).

    * **KNN**: uses :meth:`suggest_detailed` with ``top_k=None``.
    * **AutoIntent**: ranks by raw pipeline scores for every intent in ``_tool_names``.
    """
    if _is_knn(suggester):
        result = await suggester.suggest_detailed(context, top_k=None)
        return [s.id for s in result.suggestions]

    if _is_autointent(suggester):
        return await _autointent_full_rank_ids(suggester, context)

    msg = f"Unsupported suggester for full ranking: {type(suggester).__name__}"
    raise TypeError(msg)


async def _autointent_full_rank_ids(
    suggester: AutoIntentSuggester,
    context: Sequence[ModelMessage],
) -> list[str]:
    """Rank tools by classifier score (all labels), not only thresholded positives."""

    def _sync() -> list[str]:
        if not suggester.is_trained or suggester._pipeline is None:  # noqa: SLF001
            return []
        query_text = suggester.formatter.format(list(context))
        if not query_text:
            return []
        output = suggester._pipeline.predict_with_metadata([query_text])  # noqa: SLF001
        if not output.utterances:
            return []
        scores = output.utterances[0].score
        names = suggester._tool_names  # noqa: SLF001
        pairs = [(names[i], float(scores[i])) for i in range(min(len(names), len(scores)))]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return [p[0] for p in pairs]

    return await asyncio.to_thread(_sync)


def assert_suggester_supported(suggester: BaseSuggester) -> None:
    """Raise if ranking is not implemented for this suggester type."""
    if _is_knn(suggester) or _is_autointent(suggester):
        return
    msg = f"Only KNNSuggester and AutoIntentSuggester are supported, got {type(suggester).__name__}"
    raise TypeError(msg)
