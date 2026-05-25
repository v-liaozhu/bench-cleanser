"""Parse unified diff strings into structured PatchHunk objects.

Handles the unified diff format found in SWE-bench ``patch`` and
``test_patch`` fields, including multi-file diffs, binary file markers,
and ``\\ No newline at end of file`` notices.
"""

from __future__ import annotations

import re

from bench_cleanser.models import PatchHunk

_DIFF_GIT_RE = re.compile(r"^diff --git a/(.+?) b/(.+)$")
_MINUS_FILE_RE = re.compile(r"^--- (?:a/)?(.+)$")
_PLUS_FILE_RE = re.compile(r"^\+\+\+ (?:b/)?(.+)$")
_HUNK_HEADER_RE = re.compile(
    r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@(.*)$"
)


def parse_patch(patch_text: str) -> list[PatchHunk]:
    """Parse a unified diff string into a list of :class:`PatchHunk` objects.

    Handles:
    * Multi-file diffs (``diff --git a/path b/path`` headers).
    * File paths from ``---``/``+++`` lines when ``diff --git`` is absent.
    * Function context extracted from ``@@ ... @@ function_name`` headers.
    * Context, added, and removed line separation.
    * Binary file notices and ``\\ No newline at end of file`` markers.
    * Empty or ``None`` input.

    Parameters
    ----------
    patch_text:
        The unified diff string to parse.

    Returns
    -------
    list[PatchHunk]
        One entry per hunk found in the diff.
    """
    if not patch_text or not patch_text.strip():
        return []

    lines = patch_text.splitlines(keepends=True)

    hunks: list[PatchHunk] = []

    # State while walking lines
    current_file: str | None = None
    hunk_index_in_file: int = 0

    # Accumulation for the current hunk
    hunk_header: str | None = None
    hunk_func_ctx: str = ""
    hunk_added: list[str] = []
    hunk_removed: list[str] = []
    hunk_context: list[str] = []
    hunk_raw_lines: list[str] = []
    in_hunk: bool = False

    def _flush_hunk() -> None:
        """Emit the currently accumulated hunk, if any."""
        nonlocal hunk_header, hunk_func_ctx
        nonlocal hunk_added, hunk_removed, hunk_context, hunk_raw_lines
        nonlocal in_hunk

        if hunk_header is None or current_file is None:
            return

        hunks.append(
            PatchHunk(
                file_path=current_file,
                hunk_index=hunk_index_in_file - 1,  # already incremented
                header=hunk_header.rstrip("\n").rstrip("\r"),
                added_lines=list(hunk_added),
                removed_lines=list(hunk_removed),
                context_lines=list(hunk_context),
                function_context=hunk_func_ctx,
                raw_diff="".join(hunk_raw_lines),
            )
        )

        # Reset hunk accumulators
        hunk_header = None
        hunk_func_ctx = ""
        hunk_added = []
        hunk_removed = []
        hunk_context = []
        hunk_raw_lines = []
        in_hunk = False

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        stripped = line.rstrip("\n").rstrip("\r")

        # --- diff --git header --------------------------------------------
        m_git = _DIFF_GIT_RE.match(stripped)
        if m_git:
            # Flush any in-progress hunk from the previous file
            _flush_hunk()
            current_file = m_git.group(2)
            hunk_index_in_file = 0
            in_hunk = False
            idx += 1
            continue

        # --- minus (old) file ---------------------------------------------
        m_minus = _MINUS_FILE_RE.match(stripped)
        if m_minus and not in_hunk:
            # Only update file if we didn't already get it from diff --git
            if current_file is None:
                minus_path = m_minus.group(1)
                if minus_path != "/dev/null":
                    current_file = minus_path
            idx += 1
            continue

        # --- plus (new) file ----------------------------------------------
        m_plus = _PLUS_FILE_RE.match(stripped)
        if m_plus and not in_hunk:
            plus_path = m_plus.group(1)
            if plus_path != "/dev/null":
                current_file = plus_path
            # If both --- and +++ were /dev/null something is odd, but we
            # just leave current_file as-is.
            idx += 1
            continue

        # --- hunk header (@@ ... @@) --------------------------------------
        m_hunk = _HUNK_HEADER_RE.match(stripped)
        if m_hunk:
            # Flush previous hunk for the same file if any
            _flush_hunk()

            hunk_header = stripped
            # The text after the closing @@ is the optional function context
            hunk_func_ctx = m_hunk.group(1).strip()
            hunk_index_in_file += 1
            hunk_raw_lines = [line]
            in_hunk = True
            idx += 1
            continue

        # --- inside a hunk body -------------------------------------------
        if in_hunk:
            # "\ No newline at end of file" -- keep in raw, skip otherwise
            if stripped.startswith("\\"):
                hunk_raw_lines.append(line)
                idx += 1
                continue

            if line.startswith("+"):
                hunk_added.append(stripped[1:])
                hunk_raw_lines.append(line)
            elif line.startswith("-"):
                hunk_removed.append(stripped[1:])
                hunk_raw_lines.append(line)
            elif line.startswith(" ") or stripped == "":
                # Context line (starts with a space) or a blank line within
                # the hunk.  A truly blank line (no leading space) can appear
                # in some diffs when trailing whitespace was stripped.
                content = stripped[1:] if line.startswith(" ") else stripped
                hunk_context.append(content)
                hunk_raw_lines.append(line)
            else:
                # Non-diff line encountered while inside a hunk -- this means
                # the hunk body has ended (e.g. a new diff --git line that we
                # will handle on the next iteration).  Do NOT advance idx so
                # the outer loop re-processes this line.
                _flush_hunk()
                continue

            idx += 1
            continue

        # --- any other line (index, mode, Binary files, etc.) -------------
        idx += 1

    # Flush the very last hunk
    _flush_hunk()

    return hunks


def get_files_from_patch(patch_text: str) -> list[str]:
    """Return the list of file paths touched by a unified diff.

    This is a lightweight extraction that does **not** build full
    :class:`PatchHunk` objects.  Paths are returned in the order they
    first appear, without duplicates.

    Parameters
    ----------
    patch_text:
        The unified diff string.

    Returns
    -------
    list[str]
        Unique file paths in order of first appearance.
    """
    if not patch_text or not patch_text.strip():
        return []

    seen: set[str] = set()
    result: list[str] = []

    for raw_line in patch_text.splitlines():
        line = raw_line.rstrip("\n").rstrip("\r")

        m_git = _DIFF_GIT_RE.match(line)
        if m_git:
            path = m_git.group(2)
            if path not in seen:
                seen.add(path)
                result.append(path)
            continue

        # Fall back to +++ lines for diffs without ``diff --git``
        m_plus = _PLUS_FILE_RE.match(line)
        if m_plus:
            path = m_plus.group(1)
            if path != "/dev/null" and path not in seen:
                seen.add(path)
                result.append(path)

    return result
