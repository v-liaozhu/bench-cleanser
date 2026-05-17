"""Loadable LLM prompts.

Prompts live in this package as plain markdown so they are diffable, reviewable
in PRs, and editable without touching Python code. Use :func:`load` to read a
prompt by short name.

Naming: one file per prompt, lowercase ``snake_case.md`` — e.g.
``task_classifier.md``, ``intent_extraction.md``, ``trajectory_analysis.md``.
"""

from __future__ import annotations

from importlib.resources import files

__all__ = ["load"]


def load(name: str) -> str:
    """Return the markdown contents of the prompt named *name*.

    *name* is the file stem without extension. Raises ``FileNotFoundError``
    if no matching prompt exists.

    Example::

        from bench_cleanser.prompts import load
        SYSTEM_PROMPT = load("task_classifier")
    """
    resource = files(__name__).joinpath(f"{name}.md")
    if not resource.is_file():
        raise FileNotFoundError(f"prompt not found: {name}.md")
    return resource.read_text(encoding="utf-8").rstrip("\n")
