"""Stage 3: Structural diff analysis using astred_core.

Parses source files before and after the gold patch to extract:
- Changed blocks (functions/classes) with edit status
- Test functions with extracted assertions
- Call graph edges between tests and changed source

Falls back to Python ``ast`` module when astred_core is unavailable.
"""

from __future__ import annotations

import ast
import logging
import pathlib
import re
import subprocess
import tempfile
from typing import Sequence

from bench_cleanser.models import (
    AssertionDetail,
    AssertionVerdict,
    ChangedBlock,
    ParsedTask,
    PatchHunk,
    StructuralDiff,
    TestBlock,
)
from bench_cleanser.static_analysis import extract_assertions, extract_test_calls

logger = logging.getLogger(__name__)

# Try importing astred_core – it requires pythonnet and .NET runtime
try:
    import astred_core
    from astred_core import (
        BlockEditStatus,
        BlockFunction,
        CodeGraph,
        CodeGraphEdits,
        GraphLanguage,
        XAst,
    )
    ASTRED_AVAILABLE = True
except ImportError:
    ASTRED_AVAILABLE = False
    logger.info("astred_core not available; falling back to Python ast")


# ── Public API ────────────────────────────────────────────────────────


def compute_structural_diff(
    parsed_task: ParsedTask,
    repo_path: pathlib.Path | None,
) -> StructuralDiff:
    """Compute structural diff for a parsed task.

    If *repo_path* is provided and astred_core is available, uses
    astred_core for full structural analysis.  Otherwise falls back to
    Python ``ast`` for basic extraction.
    """
    instance_id = parsed_task.record.instance_id

    if ASTRED_AVAILABLE and repo_path and repo_path.exists():
        try:
            return _compute_with_astred(parsed_task, repo_path)
        except Exception:
            logger.warning(
                "%s: astred_core failed, falling back to ast",
                instance_id,
                exc_info=True,
            )

    return _compute_with_python_ast(parsed_task, repo_path)


# ── astred_core implementation ────────────────────────────────────────


def _compute_with_astred(
    parsed_task: ParsedTask,
    repo_path: pathlib.Path,
) -> StructuralDiff:
    """Full structural analysis using astred_core."""
    instance_id = parsed_task.record.instance_id

    # Collect source files touched by the gold patch
    patch_files = list(set(h.file_path for h in parsed_task.patch_hunks))
    python_patch_files = [f for f in patch_files if f.endswith(".py")]

    # Build CodeGraph from the pre-patch repo
    abs_paths = [str(repo_path / f) for f in python_patch_files if (repo_path / f).exists()]

    if not abs_paths:
        logger.debug("%s: no Python patch files found in repo", instance_id)
        return _compute_with_python_ast(parsed_task, repo_path)

    # Build pre-patch graph
    pre_graph = CodeGraph.build_from_paths(abs_paths, GraphLanguage.PYTHON)

    # Apply gold patch to get post-patch files
    post_paths = _apply_patch_to_tempdir(repo_path, parsed_task.record.patch, python_patch_files)

    if not post_paths:
        logger.debug("%s: patch application failed, falling back", instance_id)
        return _compute_with_python_ast(parsed_task, repo_path)

    # Build post-patch graph and compute edits
    edits = CodeGraphEdits.build(pre_graph, post_paths)

    # Extract changed blocks from edit script
    changed_blocks = _extract_changed_blocks_astred(edits, pre_graph, python_patch_files)

    # Extract test blocks
    test_blocks = _extract_test_blocks(parsed_task, repo_path)

    # Build call edges
    call_edges = _build_call_edges(test_blocks, changed_blocks)

    return StructuralDiff(
        instance_id=instance_id,
        changed_blocks=changed_blocks,
        test_blocks=test_blocks,
        call_edges=call_edges,
        astred_available=True,
    )


def _extract_changed_blocks_astred(
    edits: CodeGraphEdits,
    pre_graph: CodeGraph,
    patch_files: list[str],
) -> list[ChangedBlock]:
    """Extract changed blocks from astred_core CodeGraphEdits."""
    changed: list[ChangedBlock] = []

    for file_path in patch_files:
        # Get file block from graph
        file_block = pre_graph.get_file_block_with_path(file_path)
        if file_block is None:
            continue

        # Find all function and class blocks in the file
        top_blocks = file_block.find_top_descendants()
        if top_blocks is None:
            continue

        for block in top_blocks:
            status = block.edit_status
            if status is None:
                continue

            # Check if block was modified
            status_str = str(status)
            if "INSERT" in status_str or "DELETE" in status_str or "UPDATE" in status_str:
                block_type = _astred_block_type(block)
                block_name = block.name or "(anonymous)"

                pre_source = ""
                post_source = ""
                try:
                    pre_source = block.get_text() or ""
                except Exception:
                    pass

                edit_status_label = "UPDATE"
                if "INSERT" in status_str:
                    edit_status_label = "INSERT"
                elif "DELETE" in status_str:
                    edit_status_label = "DELETE"

                changed.append(ChangedBlock(
                    file_path=file_path,
                    block_name=block_name,
                    block_type=block_type,
                    edit_status=edit_status_label,
                    pre_source=pre_source,
                    post_source=post_source,
                ))

    return changed


def _astred_block_type(block) -> str:
    """Map astred_core block class to a string label."""
    type_name = type(block).__name__
    mapping = {
        "BlockFunction": "function",
        "BlockClass": "class",
        "BlockClassDeclaration": "class",
        "BlockStatement": "statement",
        "BlockImport": "import",
        "BlockAssignment": "assignment",
        "BlockVariable": "variable",
        "BlockAttribute": "attribute",
    }
    return mapping.get(type_name, "other")


def _apply_patch_to_tempdir(
    repo_path: pathlib.Path,
    patch_text: str,
    python_files: list[str],
) -> list[str]:
    """Apply the gold patch and return paths to post-patch files.

    Creates temporary copies of files, applies the patch, and returns
    the paths to the modified files.
    """
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="bench_cleanser_"))
    result_paths: list[str] = []

    try:
        # Copy original files to tmpdir preserving directory structure
        for rel_path in python_files:
            src = repo_path / rel_path
            if not src.exists():
                continue
            dst = tmpdir / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(src.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")

        # Write patch to temp file
        patch_file = tmpdir / "_patch.diff"
        patch_file.write_text(patch_text, encoding="utf-8")

        # Apply patch
        try:
            subprocess.run(
                ["git", "apply", "--allow-empty", str(patch_file)],
                cwd=str(tmpdir),
                capture_output=True,
                timeout=30,
                check=False,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # If git apply fails, try manual application
            pass

        # Collect post-patch file paths
        for rel_path in python_files:
            post_file = tmpdir / rel_path
            if post_file.exists():
                result_paths.append(str(post_file))

    except Exception:
        logger.debug("Patch application to tmpdir failed", exc_info=True)

    return result_paths


# ── Python ast fallback ───────────────────────────────────────────────


def _compute_with_python_ast(
    parsed_task: ParsedTask,
    repo_path: pathlib.Path | None,
) -> StructuralDiff:
    """Fallback structural analysis using Python's ast module."""
    instance_id = parsed_task.record.instance_id

    # Extract changed blocks from patch hunks
    changed_blocks = _extract_changed_blocks_from_hunks(parsed_task.patch_hunks, repo_path)

    # Extract test blocks
    test_blocks = _extract_test_blocks(parsed_task, repo_path)

    # Build call edges
    call_edges = _build_call_edges(test_blocks, changed_blocks)

    return StructuralDiff(
        instance_id=instance_id,
        changed_blocks=changed_blocks,
        test_blocks=test_blocks,
        call_edges=call_edges,
        astred_available=False,
    )


def _extract_changed_blocks_from_hunks(
    hunks: list[PatchHunk],
    repo_path: pathlib.Path | None,
) -> list[ChangedBlock]:
    """Extract changed blocks by parsing patch hunks with Python ast."""
    changed: list[ChangedBlock] = []
    seen: set[tuple[str, str]] = set()

    for hunk in hunks:
        if not hunk.file_path.endswith(".py"):
            continue

        # Use function context from the @@ header
        func_name = hunk.function_context.strip()
        if func_name:
            # Clean up function context (e.g., "def foo(...):" → "foo")
            func_name = _clean_function_context(func_name)

        # Determine edit status from hunk content
        has_added = bool(hunk.added_lines)
        has_removed = bool(hunk.removed_lines)
        if has_added and has_removed:
            edit_status = "UPDATE"
        elif has_added:
            edit_status = "INSERT"
        elif has_removed:
            edit_status = "DELETE"
        else:
            continue

        # Try to read full function source from repo
        pre_source = ""
        if repo_path and func_name:
            file_path = repo_path / hunk.file_path
            if file_path.exists():
                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                    pre_source = _extract_function_source_ast(content, func_name)
                except OSError:
                    pass

        # Determine block type from diff content heuristics
        block_type = _infer_block_type(hunk, func_name)

        key = (hunk.file_path, func_name or f"hunk_{hunk.hunk_index}")
        if key not in seen:
            seen.add(key)
            changed.append(ChangedBlock(
                file_path=hunk.file_path,
                block_name=func_name or f"(hunk {hunk.hunk_index})",
                block_type=block_type,
                edit_status=edit_status,
                pre_source=pre_source,
            ))

    return changed


def _clean_function_context(ctx: str) -> str:
    """Extract the function/class/method name from a @@ context header."""
    # Patterns like "def foo(...):" or "class Foo(...):" or "    def bar(self):"
    m = re.search(r"(?:def|class)\s+(\w+)", ctx)
    if m:
        return m.group(1)
    # If it's just a name
    m = re.match(r"(\w+)", ctx.strip())
    return m.group(1) if m else ctx.strip()


def _infer_block_type(hunk: PatchHunk, func_name: str) -> str:
    """Infer block type from hunk content."""
    all_lines = "\n".join(hunk.added_lines + hunk.removed_lines + hunk.context_lines)
    if re.search(r"^\s*class\s+", all_lines, re.MULTILINE):
        return "class"
    if re.search(r"^\s*(?:async\s+)?def\s+", all_lines, re.MULTILINE):
        return "function"
    if hunk.is_init_file:
        return "import"
    return "statement"


def _extract_function_source_ast(content: str, func_name: str) -> str:
    """Extract function source using Python ast."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return ""

    lines = content.splitlines(keepends=True)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == func_name:
                start = node.lineno - 1
                end = node.end_lineno or (start + 1)
                return "".join(lines[start:end])
        elif isinstance(node, ast.ClassDef):
            if node.name == func_name:
                start = node.lineno - 1
                end = node.end_lineno or (start + 1)
                return "".join(lines[start:end])
    return ""


# ── Test block extraction ─────────────────────────────────────────────


def _extract_test_blocks(
    parsed_task: ParsedTask,
    repo_path: pathlib.Path | None,
) -> list[TestBlock]:
    """Extract F2P test functions with assertions."""
    test_blocks: list[TestBlock] = []

    for th in parsed_task.f2p_test_hunks:
        # Get test source: prefer code_context post-patch, fallback to reconstructed
        test_source = th.full_source
        if th.code_context and th.code_context.post_patch_test_source:
            test_source = th.code_context.post_patch_test_source

        # Extract assertions using existing static_analysis module
        raw_assertions = extract_assertions(test_source)
        assertion_details = [
            AssertionDetail(
                statement=a.statement,
                verdict=AssertionVerdict.ON_TOPIC,  # default; overwritten by Stage 4
                reason="",
            )
            for a in raw_assertions
        ]

        # Extract called functions
        called_funcs = extract_test_calls(test_source)

        test_blocks.append(TestBlock(
            test_id=th.full_test_id,
            test_name=th.test_name,
            file_path=th.file_path,
            full_source=test_source,
            assertions=assertion_details,
            called_functions=called_funcs,
        ))

    # Also handle F2P tests with no matching hunk (they exist in the repo)
    for test_id in parsed_task.f2p_tests_with_no_hunk:
        if not repo_path:
            continue
        # Try to find the test source from the repo
        test_source = _find_test_source_in_repo(test_id, repo_path)
        if not test_source:
            continue

        # Extract test name from the test ID
        test_name = _test_name_from_id(test_id)

        raw_assertions = extract_assertions(test_source)
        assertion_details = [
            AssertionDetail(
                statement=a.statement,
                verdict=AssertionVerdict.ON_TOPIC,
                reason="",
            )
            for a in raw_assertions
        ]
        called_funcs = extract_test_calls(test_source)

        test_blocks.append(TestBlock(
            test_id=test_id,
            test_name=test_name,
            file_path="",
            full_source=test_source,
            assertions=assertion_details,
            called_functions=called_funcs,
        ))

    return test_blocks


def _find_test_source_in_repo(test_id: str, repo_path: pathlib.Path) -> str:
    """Try to find a test function's source in the repo by its test ID.

    Test IDs look like:
    - "tests.model_forms.tests.FormFieldCallbackTests.test_custom_callback_in_meta"
    - "test_custom_callback_in_meta (model_forms.tests.FormFieldCallbackTests)"
    """
    from bench_cleanser.code_visitor import extract_function_source

    # Try both ID formats
    test_name = _test_name_from_id(test_id)
    file_path = _file_path_from_test_id(test_id)

    if file_path and test_name:
        full_path = repo_path / file_path
        if full_path.exists():
            try:
                content = full_path.read_text(encoding="utf-8", errors="replace")
                return extract_function_source(content, test_name)
            except OSError:
                pass

    return ""


def _test_name_from_id(test_id: str) -> str:
    """Extract just the test function name from a test ID."""
    # Format: "test_foo (module.Class)" or "module.Class.test_foo"
    if " (" in test_id:
        return test_id.split(" (")[0].split(".")[-1]
    return test_id.split(".")[-1].split("[")[0]


def _file_path_from_test_id(test_id: str) -> str:
    """Try to extract a file path from a test ID."""
    # Format: "tests/foo/test_bar.py::test_baz"
    if "::" in test_id:
        return test_id.split("::")[0]

    # Format: "test_foo (module.tests.ClassName)"
    if " (" in test_id:
        module = test_id.split("(")[1].rstrip(")")
        parts = module.split(".")
        # Convert module path to file path guess
        # e.g. "model_forms.tests.FormFieldCallbackTests" → "tests/model_forms/tests.py"
        # This is approximate; the pipeline's parsed data is more reliable
        if len(parts) >= 2:
            file_parts = parts[:-1]  # drop the class name
            return "/".join(file_parts) + ".py"

    return ""


# ── Call graph construction ───────────────────────────────────────────


def _build_call_edges(
    test_blocks: list[TestBlock],
    changed_blocks: list[ChangedBlock],
) -> list[tuple[str, str]]:
    """Build call edges: (test_name, changed_block_name) pairs.

    A test is linked to a changed block if it calls a function with the
    same name as the changed block.
    """
    changed_names = {cb.block_name for cb in changed_blocks}
    edges: list[tuple[str, str]] = []

    for tb in test_blocks:
        for call_name in tb.called_functions:
            # Match by base name (last part of dotted name)
            base_call = call_name.split(".")[-1]
            if base_call in changed_names:
                edges.append((tb.test_name, base_call))

    return edges
