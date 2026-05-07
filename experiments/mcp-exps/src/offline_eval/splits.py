"""Task-level train/test splits (holdout and k-fold) for offline evaluation."""

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import KFold, train_test_split  # type: ignore[import-untyped]


@dataclass(frozen=True)
class TaskSplit:
    """Train and test task ids (``case_name`` strings or synthetic group keys)."""

    train_task_ids: list[str]
    test_task_ids: list[str]


def _task_ids_sorted[T](task_to_samples: dict[str, list[T]]) -> list[str]:
    return sorted(task_to_samples.keys())


def build_holdout_split[T](
    task_to_samples: dict[str, list[T]],
    *,
    test_size: float,
    random_state: int,
) -> TaskSplit:
    """Single holdout split on **tasks** (not individual samples)."""
    ids = _task_ids_sorted(task_to_samples)
    if len(ids) < 2:  # noqa: PLR2004
        msg = "Need at least two task groups for holdout split"
        raise ValueError(msg)
    train_ids, test_ids = train_test_split(
        ids,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    return TaskSplit(train_task_ids=sorted(train_ids), test_task_ids=sorted(test_ids))


def build_cv_splits[T](
    task_to_samples: dict[str, list[T]],
    *,
    n_splits: int,
    random_state: int,
) -> list[TaskSplit]:
    """K-fold cross-validation on **task** ids; every task appears in test exactly once."""
    ids = _task_ids_sorted(task_to_samples)
    if n_splits < 2:  # noqa: PLR2004
        msg = "n_splits must be >= 2"
        raise ValueError(msg)
    if len(ids) < n_splits:
        msg = f"Need at least n_splits={n_splits} task groups, got {len(ids)}"
        raise ValueError(msg)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds: list[TaskSplit] = []
    arr = np.array(ids, dtype=object)
    for train_idx, test_idx in kf.split(arr):
        train_ids = [str(arr[i]) for i in train_idx]
        test_ids = [str(arr[i]) for i in test_idx]
        folds.append(TaskSplit(train_task_ids=sorted(train_ids), test_task_ids=sorted(test_ids)))
    return folds


def samples_for_tasks[T](task_to_samples: dict[str, list[T]], task_ids: list[str]) -> list[T]:
    """Concatenate samples belonging to the given task ids (stable order by task id)."""
    out: list[T] = []
    for tid in sorted(task_ids):
        out.extend(task_to_samples.get(tid, []))
    return out
