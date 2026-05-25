"""AST-based static analysis for understanding test behavior.

Extracts function calls, assertions, and import chains from test source
code to identify what source code a test actually exercises.
"""

from __future__ import annotations

import ast
import logging
import pathlib
import re
from collections.abc import Sequence

from bench_cleanser.code_visitor import extract_function_source
from bench_cleanser.models import Assertion, CallTarget, TestedFunction

logger = logging.getLogger(__name__)

# unittest assertion methods
_UNITTEST_ASSERT_METHODS = {
    "assertEqual",
    "assertNotEqual",
    "assertTrue",
    "assertFalse",
    "assertIs",
    "assertIsNot",
    "assertIsNone",
    "assertIsNotNone",
    "assertIn",
    "assertNotIn",
    "assertIsInstance",
    "assertNotIsInstance",
    "assertRaises",
    "assertRaisesRegex",
    "assertWarns",
    "assertWarnsRegex",
    "assertAlmostEqual",
    "assertNotAlmostEqual",
    "assertGreater",
    "assertGreaterEqual",
    "assertLess",
    "assertLessEqual",
    "assertRegex",
    "assertNotRegex",
    "assertCountEqual",
    "assertMultiLineEqual",
    "assertSequenceEqual",
    "assertListEqual",
    "assertTupleEqual",
    "assertSetEqual",
    "assertDictEqual",
}


def extract_test_calls(test_source: str) -> list[str]:
    """Extract all function/method call names from test source via AST.

    Returns a list of call names (e.g., ``["Run", "capsys.readouterr", "assert"]``).
    Names are deduplicated but order is preserved.
    """
    try:
        tree = ast.parse(test_source)
    except SyntaxError:
        return []

    calls: list[str] = []
    seen: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name and name not in seen:
                seen.add(name)
                calls.append(name)

    return calls


def _call_name(node: ast.Call) -> str:
    """Extract a human-readable name from a Call node."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts: list[str] = [func.attr]
        val = func.value
        while isinstance(val, ast.Attribute):
            parts.append(val.attr)
            val = val.value
        if isinstance(val, ast.Name):
            parts.append(val.id)
        parts.reverse()
        return ".".join(parts)
    return ""


def extract_assertions(test_source: str) -> list[Assertion]:
    """Extract structured assertions from test source.

    Handles both ``assert`` statements and unittest-style method calls.
    """
    try:
        tree = ast.parse(test_source)
    except SyntaxError:
        return []

    assertions: list[Assertion] = []
    source_lines = test_source.splitlines()

    for node in ast.walk(tree):
        # Plain assert statements
        if isinstance(node, ast.Assert):
            stmt = _get_source_line(source_lines, node.lineno)
            # Try to get a readable target
            target_str = _unparse_safe(node.test)
            expected = ""
            # Check for comparisons: assert x == y
            if isinstance(node.test, ast.Compare):
                if node.test.comparators:
                    expected = _unparse_safe(node.test.comparators[0])
                target_str = _unparse_safe(node.test.left)

            assertions.append(Assertion(
                statement=stmt.strip(),
                assertion_type="assert",
                target_expression=target_str,
                expected_value=expected,
            ))

        # Method-based assertions (self.assertEqual, pytest.raises, etc.)
        if isinstance(node, ast.Call):
            name = _call_name(node)
            short_name = name.split(".")[-1] if "." in name else name
            if short_name in _UNITTEST_ASSERT_METHODS:
                stmt = _get_source_line(source_lines, node.lineno)
                target = ""
                expected = ""
                if node.args:
                    target = _unparse_safe(node.args[0])
                if len(node.args) >= 2:
                    expected = _unparse_safe(node.args[1])
                assertions.append(Assertion(
                    statement=stmt.strip(),
                    assertion_type=short_name,
                    target_expression=target,
                    expected_value=expected,
                ))

        # pytest.raises context manager
        if isinstance(node, ast.With):
            for item in node.items:
                if isinstance(item.context_expr, ast.Call):
                    name = _call_name(item.context_expr)
                    if "raises" in name or "warns" in name:
                        stmt = _get_source_line(source_lines, node.lineno)
                        exc_type = ""
                        if item.context_expr.args:
                            exc_type = _unparse_safe(item.context_expr.args[0])
                        assertions.append(Assertion(
                            statement=stmt.strip(),
                            assertion_type="pytest.raises" if "raises" in name else "pytest.warns",
                            target_expression=exc_type,
                            expected_value="",
                        ))

    return assertions


def _unparse_safe(node: ast.AST | None) -> str:
    """Convert an AST node back to source, with fallback."""
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return ast.dump(node)


def _get_source_line(lines: list[str], lineno: int) -> str:
    """Get a source line by 1-based line number."""
    if 0 < lineno <= len(lines):
        return lines[lineno - 1]
    return ""


def resolve_imports(
    file_content: str,
    repo_path: pathlib.Path,
) -> dict[str, str]:
    """Map imported names to file paths within the repo.

    Returns a dict from import name to resolved file path
    (relative to repo root).
    """
    try:
        tree = ast.parse(file_content)
    except SyntaxError:
        return {}

    mapping: dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name
                resolved = _resolve_module_path(alias.name, repo_path)
                if resolved:
                    mapping[name] = resolved

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                base_resolved = _resolve_module_path(node.module, repo_path)
                for alias in node.names:
                    name = alias.asname or alias.name
                    if base_resolved:
                        mapping[name] = base_resolved

    return mapping


def _resolve_module_path(module: str, repo_path: pathlib.Path) -> str:
    """Try to find the file for a Python module within the repo."""
    parts = module.split(".")
    # Try as a package directory
    candidates = [
        "/".join(parts) + ".py",
        "/".join(parts) + "/__init__.py",
    ]
    for candidate in candidates:
        if (repo_path / candidate).exists():
            return candidate
    return ""


def identify_tested_functions(
    test_source: str,
    import_map: dict[str, str],
    patch_files: Sequence[str],
    repo_path: pathlib.Path,
    *,
    max_source_lines: int = 200,
) -> list[TestedFunction]:
    """Identify source functions that the test exercises.

    Cross-references function calls in the test with files modified
    by the gold patch to find functions that are both called by the
    test AND live in patched files.
    """
    calls = extract_test_calls(test_source)
    results: list[TestedFunction] = []
    seen: set[str] = set()

    # Normalize patch file paths for matching
    patch_file_set = {_normalize_path(f) for f in patch_files}

    for call_name in calls:
        # Skip common builtins / test helpers
        base_name = call_name.split(".")[-1]
        if base_name in ("print", "len", "str", "int", "float", "list", "dict",
                         "set", "tuple", "type", "isinstance", "hasattr",
                         "getattr", "setattr", "range", "enumerate", "zip",
                         "format", "repr", "super", "open", "sorted"):
            continue

        # Check if the call is directly an imported name
        top_name = call_name.split(".")[0]
        if top_name in import_map:
            source_file = import_map[top_name]
            normalized = _normalize_path(source_file)

            if normalized in patch_file_set and base_name not in seen:
                seen.add(base_name)
                # Read the function source
                full_path = repo_path / source_file
                if full_path.exists():
                    try:
                        content = full_path.read_text(encoding="utf-8", errors="replace")
                        func_source = extract_function_source(
                            content,
                            base_name,
                            max_lines=max_source_lines,
                        )
                    except OSError:
                        func_source = ""
                else:
                    func_source = ""

                results.append(TestedFunction(
                    name=base_name,
                    file_path=source_file,
                    source=func_source,
                    is_modified_by_patch=True,
                ))

    # Also check if any call names match function names in patch hunks
    # by searching patch files directly for matching function defs
    for pf in patch_files:
        if pf.endswith(".py"):
            full_path = repo_path / pf
            if not full_path.exists():
                continue
            try:
                content = full_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            for call_name in calls:
                base_name = call_name.split(".")[-1]
                if base_name in seen:
                    continue
                # Check if this function exists in the patch file
                pattern = re.compile(
                    rf"^\s*(?:async\s+)?def\s+{re.escape(base_name)}\s*\(",
                    re.MULTILINE,
                )
                if pattern.search(content):
                    seen.add(base_name)
                    func_source = extract_function_source(
                        content,
                        base_name,
                        max_lines=max_source_lines,
                    )
                    results.append(TestedFunction(
                        name=base_name,
                        file_path=pf,
                        source=func_source,
                        is_modified_by_patch=True,
                    ))

    return results


def _normalize_path(p: str) -> str:
    """Normalize a file path for comparison."""
    return p.replace("\\", "/").lstrip("/").rstrip("/")


def build_call_targets(
    test_source: str,
    import_map: dict[str, str],
    patch_files: Sequence[str],
) -> list[CallTarget]:
    """Build a list of CallTarget objects from test source.

    Each call is annotated with whether it targets a file modified
    by the gold patch.
    """
    calls = extract_test_calls(test_source)
    patch_file_set = {_normalize_path(f) for f in patch_files}
    targets: list[CallTarget] = []

    # Get line numbers via AST
    line_map: dict[str, int] = {}
    try:
        tree = ast.parse(test_source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _call_name(node)
                if name and name not in line_map:
                    line_map[name] = node.lineno
    except SyntaxError:
        pass

    for call_name in calls:
        top_name = call_name.split(".")[0]
        resolved_file = import_map.get(top_name, "")
        is_in_patch = _normalize_path(resolved_file) in patch_file_set if resolved_file else False

        targets.append(CallTarget(
            name=call_name,
            module=top_name if top_name in import_map else "",
            file_path=resolved_file,
            line_number=line_map.get(call_name, 0),
            is_in_patch=is_in_patch,
        ))

    return targets
