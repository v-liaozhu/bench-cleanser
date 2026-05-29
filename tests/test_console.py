"""Tests for the canonical terminal frontend in ``bench_cleanser._console``."""

from __future__ import annotations

import logging

from bench_cleanser._console import (
    ALL_FUSION_VERDICTS,
    ALL_LEAKAGE_PATTERNS,
    setup_logging,
    truncate_status,
)
from bench_cleanser.fusion import FusionVerdict
from bench_cleanser.trajectory.models import LeakagePattern


def test_truncate_status_empty() -> None:
    assert truncate_status([]) == ""


def test_truncate_status_fits() -> None:
    assert truncate_status(["a:1", "b:2"], max_width=80) == "a:1 b:2"


def test_truncate_status_clips_on_token_boundary() -> None:
    parts = ["alpha:1", "beta:22", "gammagamma:333", "deltadelta:4444"]
    result = truncate_status(parts, max_width=20)
    assert result.endswith("…")
    head = result.removesuffix(" …").removesuffix("…").rstrip()
    for tok in head.split(" "):
        assert tok in parts, f"{tok!r} was split mid-word"
    assert len(result) <= 20


def test_truncate_status_first_token_too_long() -> None:
    result = truncate_status(["abcdefghijklmnop"], max_width=5)
    assert result == "…"


def test_setup_logging_is_idempotent() -> None:
    setup_logging(verbose=False)
    setup_logging(verbose=True)
    setup_logging(verbose=False)
    rich_handlers = [
        h for h in logging.getLogger().handlers
        if h.__class__.__name__ == "RichHandler"
    ]
    assert len(rich_handlers) == 1


def test_all_leakage_patterns_matches_enum() -> None:
    assert ALL_LEAKAGE_PATTERNS == tuple(v.value for v in LeakagePattern)


def test_all_fusion_verdicts_matches_enum() -> None:
    assert ALL_FUSION_VERDICTS == tuple(v.value for v in FusionVerdict)
