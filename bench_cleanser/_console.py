"""Canonical terminal frontend for bench-cleanser CLIs.

Every entry point in :mod:`bench_cleanser.cli` calls :func:`setup_logging`
exactly once at start-up. That installs a single :class:`rich.logging.RichHandler`
on the root logger writing to a shared :class:`rich.console.Console`. The same
``Console`` is then handed to any live :class:`rich.progress.Progress` widget,
so log records never overwrite or interleave with progress lines.

The module also exposes :data:`ALL_LEAKAGE_PATTERNS` / :data:`ALL_FUSION_VERDICTS`,
the canonical lists used by the overview tables — sourced from the enums so a
new value cannot be silently dropped from a summary.
"""
from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

from bench_cleanser.fusion import FusionVerdict
from bench_cleanser.trajectory.models import LeakagePattern

__all__ = [
    "ALL_FUSION_VERDICTS",
    "ALL_LEAKAGE_PATTERNS",
    "get_console",
    "setup_logging",
    "truncate_status",
]


ALL_LEAKAGE_PATTERNS: tuple[str, ...] = tuple(v.value for v in LeakagePattern)
ALL_FUSION_VERDICTS: tuple[str, ...] = tuple(v.value for v in FusionVerdict)


_CONSOLE: Console | None = None
_LOGGING_CONFIGURED = False


def get_console() -> Console:
    """Return the process-wide :class:`Console`, lazily constructing one."""
    global _CONSOLE
    if _CONSOLE is None:
        _CONSOLE = Console(stderr=True, highlight=False)
    return _CONSOLE


def setup_logging(verbose: bool = False) -> Console:
    """Install a single RichHandler on the root logger and return the console.

    Idempotent: calling twice in the same process does not stack handlers.
    Returns the shared :class:`Console` so callers can pass it to Progress
    widgets and to ``console.print`` calls.
    """
    global _LOGGING_CONFIGURED
    console = get_console()
    if _LOGGING_CONFIGURED:
        logging.getLogger().setLevel(logging.DEBUG if verbose else logging.INFO)
        return console

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        show_path=False,
        omit_repeated_times=False,
        rich_tracebacks=True,
        markup=False,
        log_time_format="%H:%M:%S",
    )
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Mute third-party HTTP noise — at INFO every LLM call logs a line.
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "azure"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _LOGGING_CONFIGURED = True
    return console


def truncate_status(parts: list[str], max_width: int = 80) -> str:
    """Join *parts* with single spaces, truncating on a whole-token boundary.

    Never splits a token mid-word. When the joined string exceeds *max_width*
    characters (Rich markup is included in the width, which is a conservative
    over-count), the result is the longest token prefix that fits, followed by
    an ellipsis. An empty input returns an empty string.
    """
    if not parts:
        return ""
    full = " ".join(parts)
    if len(full) <= max_width:
        return full
    kept: list[str] = []
    used = 0
    sep = 0  # leading space for every token after the first
    ellipsis = " …"
    budget = max_width - len(ellipsis)
    for tok in parts:
        need = sep + len(tok)
        if used + need > budget:
            break
        kept.append(tok)
        used += need
        sep = 1
    if not kept:
        return ellipsis.lstrip()
    return " ".join(kept) + ellipsis
