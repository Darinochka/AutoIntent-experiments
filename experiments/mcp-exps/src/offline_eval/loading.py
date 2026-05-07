"""Load and normalize :class:`tool_suggest.models.Sample` rows from a JSONL repository."""

from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from tool_suggest.models import Sample

from src.agents._tool_suggest.constants import IGNORED_TOOL_LABELS


def iter_jsonl_samples(path: Path) -> Iterator[Sample]:
    """Yield :class:`Sample` objects from a newline-delimited JSON file."""
    with path.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            yield Sample.model_validate_json(line)


def load_jsonl_samples(path: Path) -> list[Sample]:
    """Load all samples from a tool-suggest JSONL repository."""
    return list(iter_jsonl_samples(path))


def normalize_sample_tools(sample: Sample) -> Sample | None:
    """Drop ignored tool labels; return ``None`` if the row becomes invalid."""
    tools = [t for t in sample.tools if t not in IGNORED_TOOL_LABELS]
    if sample.is_out_of_scope:
        if tools:
            return None
        return sample
    if not tools:
        return None
    if tools == sample.tools:
        return sample
    return sample.model_copy(update={"tools": tools})


def load_and_normalize(path: Path) -> list[Sample]:
    """Load JSONL and drop invalid / empty-label rows (after filtering ignored names)."""
    out: list[Sample] = []
    for raw in load_jsonl_samples(path):
        norm = normalize_sample_tools(raw)
        if norm is not None:
            out.append(norm)
    return out


def group_samples_by_task(samples: list[Sample], *, task_key: str = "case_name") -> dict[str, list[Sample]]:
    r"""Group samples by ``sample.data[task_key]``; missing key uses ``__missing_task__``."""
    groups: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        key = s.data.get(task_key) if isinstance(s.data, dict) else None
        gk = str(key) if key is not None and str(key) else "__missing_task__"
        groups[gk].append(s)
    return dict(groups)
