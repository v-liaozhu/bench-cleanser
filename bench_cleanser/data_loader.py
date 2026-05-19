"""Load SWE-bench datasets from HuggingFace and normalize to TaskRecord objects."""

from __future__ import annotations

import logging

from datasets import load_dataset

from bench_cleanser.models import TaskRecord

logger = logging.getLogger(__name__)


def _load_dataset_as_records(
    name: str, max_tasks: int, split: str = "test"
) -> list[TaskRecord]:
    """Load a HuggingFace dataset and convert rows to TaskRecord objects."""
    ds = load_dataset(name, split=split)
    records: list[TaskRecord] = []
    for row in ds:
        if len(records) >= max_tasks:
            break
        records.append(TaskRecord.from_dict(row))
    return records


def load_swebench_verified(max_tasks: int = 500) -> list[TaskRecord]:
    """Load from the SWE-bench Verified dataset (500 tasks).

    This is the primary target dataset for contamination analysis.

    Args:
        max_tasks: Maximum number of task records to return.

    Returns:
        A list of up to *max_tasks* TaskRecord objects.
    """
    return _load_dataset_as_records("princeton-nlp/SWE-bench_Verified", max_tasks)


def load_swebench_pro(max_tasks: int = 500) -> list[TaskRecord]:
    """Load from the SWE-bench Pro dataset.

    SWE-bench Pro contains harder, professionally curated tasks.

    Args:
        max_tasks: Maximum number of task records to return.

    Returns:
        A list of up to *max_tasks* TaskRecord objects.
    """
    return _load_dataset_as_records("ScaleAI/SWE-bench_Pro", max_tasks)


def load_swebench_live(max_tasks: int = 500, split: str = "verified") -> list[TaskRecord]:
    """Load from the SWE-bench Live dataset.

    SWE-bench Live is a continuously updated benchmark of real-world
    software-engineering issue resolutions.

    Available splits: test (~1k), lite (~300), verified (~500), full (~1.89k).

    See: https://huggingface.co/datasets/SWE-bench-Live/SWE-bench-Live

    Args:
        max_tasks: Maximum number of task records to return.
        split: HuggingFace dataset split to load (default: ``verified``).

    Returns:
        A list of up to *max_tasks* TaskRecord objects.
    """
    return _load_dataset_as_records(
        "SWE-bench-Live/SWE-bench-Live", max_tasks, split=split
    )


def load_all(max_per_dataset: int = 500) -> list[TaskRecord]:
    """Load from SWE-bench Verified and SWE-bench Pro, concatenated.

    Args:
        max_per_dataset: Maximum number of records to load from each dataset.

    Returns:
        Combined list of TaskRecord objects from both datasets.
    """
    verified = load_swebench_verified(max_tasks=max_per_dataset)
    pro = load_swebench_pro(max_tasks=max_per_dataset)
    return verified + pro


def load_single_task(instance_id: str) -> TaskRecord | None:
    """Search all datasets for a specific instance_id.

    Args:
        instance_id: The unique instance identifier to look for.

    Returns:
        The matching TaskRecord, or None if not found.
    """
    for name, split in [
        ("princeton-nlp/SWE-bench_Verified", "test"),
        ("ScaleAI/SWE-bench_Pro", "test"),
        ("SWE-bench-Live/SWE-bench-Live", "verified"),
    ]:
        try:
            ds = load_dataset(name, split=split)
            for row in ds:
                if row.get("instance_id") == instance_id:
                    return TaskRecord.from_dict(row)
        except Exception as exc:
            logger.warning(
                "Failed loading dataset %s (%s) while searching for %s: %s",
                name,
                split,
                instance_id,
                exc,
                exc_info=True,
            )
            continue
    return None
