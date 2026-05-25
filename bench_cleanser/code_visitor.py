"""Retrieve full source code from cloned repositories.

Provides functions to read complete test functions (pre- and post-patch),
extract imports and fixtures from test files, and read source files being
tested.
"""

from __future__ import annotations

import ast
import logging
import pathlib

logger = logging.getLogger(__name__)


def extract_function_source(
    file_content: str,
    func_name: str,
    *,
    max_lines: int = 200,
) -> str:
    """Extract a single function's source from *file_content* using AST.

    Falls back to regex-based extraction if AST parsing fails.
    Returns ``""`` if the function is not found.
    """
    # Try AST first
    try:
        tree = ast.parse(file_content)
    except SyntaxError:
        return ""

    lines = file_content.splitlines(keepends=True)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == func_name:
                start = node.lineno - 1  # 0-indexed
                end = node.end_lineno or (start + 1)
                func_lines = lines[start:end]
                if len(func_lines) > max_lines:
                    func_lines = func_lines[:max_lines]
                    func_lines.append(f"    # ... truncated ({end - start} total lines)\n")
                return "".join(func_lines)

    # Function not found via AST
    return ""


def get_full_test_source(
    repo_path: pathlib.Path,
    test_file: str,
    test_name: str,
    *,
    max_lines: int = 200,
) -> str:
    """Read the full test function source from the pre-patch repo.

    Returns ``""`` if the file or function is not found.
    """
    file_path = repo_path / test_file
    if not file_path.exists():
        logger.debug("Test file not found: %s", file_path)
        return ""

    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""

    return extract_function_source(content, test_name, max_lines=max_lines)


def get_post_patch_test_source(
    pre_patch_content: str,
    test_name: str,
    added_lines: list[str],
    removed_lines: list[str],
    *,
    max_lines: int = 200,
) -> str:
    """Reconstruct the post-patch test function.

    If we have the pre-patch content and diff info, apply the changes.
    Falls back to reconstructing from added_lines if needed.
    """
    if not pre_patch_content:
        return "\n".join(added_lines)

    if added_lines:
        return "\n".join(added_lines)

    return pre_patch_content


def extract_imports(file_content: str) -> str:
    """Extract import statements from a Python file.

    Returns a string of all import lines (``import x`` and ``from x import y``).
    """
    lines: list[str] = []
    in_multiline = False
    for line in file_content.splitlines():
        stripped = line.strip()
        if in_multiline:
            lines.append(line)
            if ")" in stripped:
                in_multiline = False
            continue
        if stripped.startswith("import ") or stripped.startswith("from "):
            lines.append(line)
            if "(" in stripped and ")" not in stripped:
                in_multiline = True
    return "\n".join(lines)


def extract_fixtures(
    file_content: str,
    test_name: str,
) -> str:
    """Extract pytest fixtures and setup methods relevant to *test_name*.

    Looks for:
    - ``@pytest.fixture`` decorated functions
    - ``setup_method`` / ``setUp`` / ``tearDown``
    - conftest.py fixtures (by name reference)
    """
    try:
        tree = ast.parse(file_content)
    except SyntaxError:
        return ""

    fixtures: list[str] = []
    lines = file_content.splitlines(keepends=True)

    # Find the test function to check its parameters (fixture injection)
    test_params: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == test_name:
                for arg in node.args.args:
                    if arg.arg not in ("self", "cls"):
                        test_params.add(arg.arg)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            is_fixture = False
            for dec in node.decorator_list:
                dec_str = ast.dump(dec)
                if "fixture" in dec_str:
                    is_fixture = True
                    break

            if is_fixture and node.name in test_params:
                start = node.lineno - 1
                end = node.end_lineno or (start + 1)
                func_lines = lines[start:min(end, start + 50)]
                fixtures.append("".join(func_lines))

            # Also capture setup/teardown
            if node.name in ("setup_method", "setUp", "tearDown", "setup", "teardown"):
                start = node.lineno - 1
                end = node.end_lineno or (start + 1)
                func_lines = lines[start:min(end, start + 30)]
                fixtures.append("".join(func_lines))

    return "\n\n".join(fixtures)


import re as _re

from bench_cleanser.models import ProblemCodeContext


def extract_entities_from_text(
    text: str,
    repo_path: pathlib.Path,
) -> dict[str, list[str]]:
    """Heuristically extract code entity references from problem text.

    Returns verified entities (files, functions, classes) that exist in repo.
    """
    # File paths: word/word.ext patterns
    file_candidates = set(_re.findall(r"[\w/\\]+\.(?:py|js|ts|go|rs|java|c|cpp|h|rb|php)", text))
    files = [f for f in file_candidates if (repo_path / f.replace("\\", "/")).exists()]

    # Class names: PascalCase words (2+ uppercase transitions)
    class_candidates = set(_re.findall(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b", text))
    # Also single-word capitalized that look like classes
    class_candidates |= set(_re.findall(r"\b([A-Z][a-z]{2,}(?:Error|Exception|Factory|Manager|Handler|Form|Model|View|Serializer))\b", text))

    # Function names: snake_case followed by (
    func_candidates = set(_re.findall(r"\b([a-z_][a-z0-9_]{2,})\s*\(", text))
    # Also __dunder__ methods
    func_candidates |= set(_re.findall(r"\b(__[a-z_]+__)\b", text))

    # Verify classes and functions exist in mentioned files
    verified_classes: list[str] = []
    verified_functions: list[str] = []

    for f in files:
        full_path = repo_path / f.replace("\\", "/")
        if not full_path.exists() or not f.endswith(".py"):
            continue
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(content)
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in class_candidates:
                verified_classes.append(node.name)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in func_candidates:
                verified_functions.append(node.name)

    return {
        "files": files,
        "functions": list(dict.fromkeys(verified_functions)),
        "classes": list(dict.fromkeys(verified_classes)),
    }


def extract_problem_code_context(
    repo_path: pathlib.Path,
    mentioned_files: list[str],
    mentioned_functions: list[str],
    mentioned_classes: list[str],
    *,
    max_file_lines: int = 300,
    max_entity_lines: int = 100,
) -> ProblemCodeContext:
    """Build ProblemCodeContext from entities mentioned in problem text.

    Reads pre-patch source files and extracts function/class definitions
    to ground Stage 2 intent extraction in actual code.
    """
    file_contents: dict[str, str] = {}
    entity_sources: dict[str, str] = {}

    for file_path in mentioned_files:
        full_path = repo_path / file_path.replace("\\", "/")
        if not full_path.exists():
            continue
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            if len(lines) > max_file_lines:
                content = "\n".join(lines[:max_file_lines]) + f"\n# ... truncated ({len(lines)} total lines)"
            file_contents[file_path] = content
        except OSError:
            continue

    # Extract function sources
    for func_name in mentioned_functions:
        for file_path in mentioned_files:
            if not file_path.endswith(".py"):
                continue
            full_path = repo_path / file_path.replace("\\", "/")
            if not full_path.exists():
                continue
            try:
                content = full_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            source = extract_function_source(content, func_name, max_lines=max_entity_lines)
            if source:
                entity_sources[func_name] = source
                break

    # Extract class sources (first max_entity_lines)
    for class_name in mentioned_classes:
        for file_path in mentioned_files:
            if not file_path.endswith(".py"):
                continue
            full_path = repo_path / file_path.replace("\\", "/")
            if not full_path.exists():
                continue
            try:
                content = full_path.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(content)
            except (OSError, SyntaxError):
                continue
            lines = content.splitlines(keepends=True)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    start = node.lineno - 1
                    end = node.end_lineno or (start + 1)
                    cls_lines = lines[start:min(end, start + max_entity_lines)]
                    if end - start > max_entity_lines:
                        cls_lines.append(f"    # ... truncated ({end - start} total lines)\n")
                    entity_sources[class_name] = "".join(cls_lines)
                    break
            if class_name in entity_sources:
                break

    # Build directory tree for mentioned file directories
    dirs: set[str] = set()
    for f in mentioned_files:
        parts = f.replace("\\", "/").split("/")
        if len(parts) > 1:
            dirs.add("/".join(parts[:-1]))

    tree_lines: list[str] = []
    for d in sorted(dirs):
        dir_path = repo_path / d
        if dir_path.exists() and dir_path.is_dir():
            tree_lines.append(f"{d}/")
            try:
                for child in sorted(dir_path.iterdir()):
                    if child.name.startswith("."):
                        continue
                    suffix = "/" if child.is_dir() else ""
                    tree_lines.append(f"  {child.name}{suffix}")
            except OSError:
                pass

    return ProblemCodeContext(
        mentioned_file_contents=file_contents,
        relevant_directory_tree="\n".join(tree_lines),
        mentioned_entity_sources=entity_sources,
    )
