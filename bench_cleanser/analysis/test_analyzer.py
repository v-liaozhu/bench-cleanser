"""Stage 4B: F2P test intent matching (batched).

All F2P tests are classified as ALIGNED, TANGENTIAL, or UNRELATED in a
SINGLE batched LLM call. Individual assertions get ON_TOPIC / OFF_TOPIC
verdicts. Code context from Stage 1.5 and structural analysis from Stage 3
enrich the prompt.

Uses structured output with strict JSON schema enforcement.
Assertion extraction uses AST-only for Python source.
"""

from __future__ import annotations

import ast
import logging

from bench_cleanser.llm_client import LLMClient
from bench_cleanser.models import (
    AssertionVerdict,
    AssertionVerdictReport,
    IntentStatement,
    ParsedTask,
    StructuralDiff,
    TestAnalysis,
    TestHunk,
    TestModificationType,
    TestVerdict,
    TestVerdictReport,
)
from bench_cleanser.prompts import load as _load_prompt
from bench_cleanser.schemas import BatchTestVerdictsResponse

logger = logging.getLogger(__name__)

BATCH_TEST_SYSTEM_PROMPT = _load_prompt("test_classifier")


def _count_assertions_ast(source: str) -> list[str]:
    """Extract assertion statements from test source using Python AST only.

    No regex patterns. Returns assertion strings.
    For non-Python source that fails AST parsing, returns an empty list
    and the LLM will analyze the raw source directly.
    """
    lines = []
    for line in source.splitlines():
        clean = line
        if clean.startswith("+"):
            clean = clean[1:]
        lines.append(clean)
    cleaned = "\n".join(lines)

    try:
        tree = ast.parse(cleaned)
    except SyntaxError:
        # Non-Python or malformed — let the LLM handle assertion analysis
        # directly from the raw source code
        return []

    assertions: list[str] = []

    UNITTEST_ASSERT_METHODS = {
        "assertEqual", "assertRaises", "assertTrue", "assertFalse",
        "assertIn", "assertNotIn", "assertIsNone", "assertIsNotNone",
        "assertAlmostEqual", "assertGreater", "assertLess", "assertRegex",
        "assertNotEqual", "assertIs", "assertIsNot", "assertCountEqual",
        "assertSequenceEqual", "assertListEqual", "assertDictEqual",
        "assertRaisesRegex", "assertWarns", "assertWarnsRegex",
        "assertGreaterEqual", "assertLessEqual", "assertNotRegex",
        "assertIsInstance", "assertNotIsInstance", "assertMultiLineEqual",
        "assertTupleEqual", "assertSetEqual", "assertNotAlmostEqual",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            try:
                assertions.append(ast.unparse(node))
            except Exception:
                assertions.append(f"assert at line {node.lineno}")
        elif isinstance(node, ast.Call):
            func = node.func
            name = ""
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            if name.startswith("assert") or name in UNITTEST_ASSERT_METHODS:
                try:
                    assertions.append(ast.unparse(node))
                except Exception:
                    assertions.append(f"{name}() at line {node.lineno}")
        elif isinstance(node, ast.With):
            for item in node.items:
                if isinstance(item.context_expr, ast.Call):
                    func = item.context_expr.func
                    call_name = ""
                    if isinstance(func, ast.Attribute):
                        call_name = func.attr
                    elif isinstance(func, ast.Name):
                        call_name = func.id
                    if "raises" in call_name.lower() or "warns" in call_name.lower():
                        try:
                            assertions.append(ast.unparse(item.context_expr))
                        except Exception:
                            assertions.append(f"{call_name}() at line {node.lineno}")

    return assertions


def _build_batch_test_prompt(
    intent: IntentStatement,
    test_hunks: list[TestHunk],
    problem_statement: str = "",
    structural_diff: StructuralDiff | None = None,
) -> str:
    """Build user prompt with all tests for batch classification."""
    parts: list[str] = []

    # Intent section
    parts.append(
        "=== INTENT (from problem statement only — no gold patch was seen) ===\n"
        f"Core requirement: {intent.core_requirement}\n\n"
        f"Behavioral contract: {intent.behavioral_contract}\n\n"
        "Acceptance criteria:\n"
        + "\n".join(f"  {i+1}. {c}" for i, c in enumerate(intent.acceptance_criteria))
        + "\n\n"
        f"Out of scope: {intent.out_of_scope}"
    )

    if intent.decomposition:
        d = intent.decomposition
        parts.append(
            "=== PROBLEM DECOMPOSITION ===\n"
            f"Bug description: {d.bug_description}\n"
            f"Reporter's suggested fix: {d.suggested_fix or '(none)'}\n"
            f"Legitimacy: {d.legitimacy}"
        )

    if problem_statement:
        parts.append(f"=== PROBLEM STATEMENT (full text) ===\n{problem_statement}")

    # All tests
    for i, test_hunk in enumerate(test_hunks):
        assertions = _count_assertions_ast(test_hunk.full_source)
        is_modified = test_hunk.modification_type == TestModificationType.MODIFIED

        test_section = (
            f"=== TEST {i} ===\n"
            f"File: {test_hunk.file_path}\n"
            f"Test name: {test_hunk.test_name}\n"
            f"Test ID: {test_hunk.full_test_id}\n"
            f"Modification type: {test_hunk.modification_type.value}\n"
            f"Is modified (pre-existing): {is_modified}\n"
        )

        ctx = test_hunk.code_context

        # Pre-patch source (for modification analysis)
        if ctx and ctx.pre_patch_test_source:
            test_section += f"\n--- Pre-patch test source (BEFORE this PR) ---\n{ctx.pre_patch_test_source}\n"

        # Post-patch source
        test_section += f"\n--- Post-patch test source (AFTER this PR) ---\n{test_hunk.full_source}\n"

        # Assertions numbered for reference
        if assertions:
            test_section += "\n--- Assertions (number each verdict by index) ---\n"
            for j, a in enumerate(assertions):
                test_section += f"  [{j}] {a}\n"
        else:
            test_section += "\n--- No assertions extracted via AST (analyze from source directly) ---\n"

        # Tested functions (code context)
        if ctx and ctx.tested_functions:
            test_section += "\n--- Tested source functions ---\n"
            for tf in ctx.tested_functions:
                tag = "MODIFIED BY GOLD PATCH" if tf.is_modified_by_patch else "not modified"
                test_section += f"Function: {tf.name} (file: {tf.file_path}, {tag})\n"
                if tf.source:
                    test_section += f"Source:\n{tf.source}\n\n"

        # Call targets
        if ctx and ctx.call_targets:
            test_section += "\n--- Call analysis ---\nFunctions called by this test:\n"
            for ct in ctx.call_targets:
                tag = "IN GOLD PATCH" if ct.is_in_patch else "not in patch"
                loc = f" in {ct.file_path}" if ct.file_path else ""
                test_section += f"  - {ct.name}{loc} ({tag})\n"

        # Structural context
        if structural_diff:
            struct_ctx = ""
            for edge_test, edge_func in structural_diff.call_edges:
                if edge_test == test_hunk.test_name:
                    for cb in structural_diff.changed_blocks:
                        if cb.block_name == edge_func:
                            struct_ctx += (
                                f"Calls -> {cb.block_name} ({cb.block_type}, "
                                f"{cb.edit_status}) in {cb.file_path}\n"
                            )
            if struct_ctx:
                test_section += f"\n--- Structural context ---\n{struct_ctx}"

        parts.append(test_section)

    return "\n\n".join(parts)


async def analyze_tests(
    parsed: ParsedTask,
    intent: IntentStatement,
    llm: LLMClient,
    structural_diff: StructuralDiff | None = None,
) -> TestAnalysis:
    """Stage 4B: classify all F2P tests against intent in a single batched LLM call."""
    all_test_hunks = list(parsed.f2p_test_hunks)
    problem_statement = parsed.record.full_problem_context

    # Handle unmatched F2P tests (tests with no hunk — not modified, likely exercise gold patch)
    unmatched_verdicts: list[TestVerdictReport] = []
    for test_id in parsed.f2p_tests_with_no_hunk:
        parts = test_id.split("::")
        test_name = parts[-1].split("[")[0] if parts else test_id
        unmatched_verdicts.append(TestVerdictReport(
            test_id=test_id, test_name=test_name,
            intent_match=TestVerdict.ALIGNED, evidence_strength="weak",
            reasoning="F2P test with no matching hunk — test was not modified, likely exercises gold patch behavior",
            is_modified=False, modification_aligned=True, assertion_verdicts=[],
        ))

    if not all_test_hunks:
        # Only unmatched tests
        aligned = sum(1 for v in unmatched_verdicts if v.intent_match == TestVerdict.ALIGNED)
        return TestAnalysis(
            total_tests=len(unmatched_verdicts),
            aligned_count=aligned, tangential_count=0, unrelated_count=0,
            total_assertions=0, on_topic_assertions=0, off_topic_assertions=0,
            has_modified_tests=False, test_verdicts=unmatched_verdicts,
        )

    user_prompt = _build_batch_test_prompt(
        intent, all_test_hunks,
        problem_statement=problem_statement,
        structural_diff=structural_diff,
    )

    result: BatchTestVerdictsResponse = await llm.query_structured(
        BATCH_TEST_SYSTEM_PROMPT,
        user_prompt,
        BatchTestVerdictsResponse,
    )

    # Map results to TestVerdictReport objects
    test_verdicts: list[TestVerdictReport] = []
    result_by_index = {v.test_index: v for v in result.verdicts}

    for i, test_hunk in enumerate(all_test_hunks):
        is_modified = test_hunk.modification_type == TestModificationType.MODIFIED
        assertions = _count_assertions_ast(test_hunk.full_source)

        verdict_item = result_by_index.get(i)
        if verdict_item is None:
            logger.warning(
                "Missing verdict for test %d (%s) — defaulting to TANGENTIAL",
                i, test_hunk.test_name,
            )
            test_verdicts.append(TestVerdictReport(
                test_id=test_hunk.full_test_id, test_name=test_hunk.test_name,
                intent_match=TestVerdict.TANGENTIAL, evidence_strength="weak",
                reasoning="No verdict returned by LLM for this test",
                is_modified=is_modified, modification_aligned=True,
            ))
            continue

        try:
            test_verdict = TestVerdict(verdict_item.test_verdict)
        except ValueError:
            test_verdict = TestVerdict.TANGENTIAL

        # Build per-assertion verdicts
        assertion_verdicts: list[AssertionVerdictReport] = []
        raw_by_index = {av.index: av for av in verdict_item.assertion_verdicts}

        for j, assertion_str in enumerate(assertions):
            av_item = raw_by_index.get(j)
            if av_item:
                try:
                    av = AssertionVerdict(av_item.verdict)
                except ValueError:
                    av = AssertionVerdict.ON_TOPIC
                assertion_verdicts.append(AssertionVerdictReport(
                    statement=assertion_str, verdict=av, reason=av_item.reason,
                ))
            else:
                assertion_verdicts.append(AssertionVerdictReport(
                    statement=assertion_str, verdict=AssertionVerdict.ON_TOPIC,
                    reason="No per-assertion verdict from LLM",
                ))

        test_verdicts.append(TestVerdictReport(
            test_id=test_hunk.full_test_id, test_name=test_hunk.test_name,
            intent_match=test_verdict, evidence_strength=verdict_item.evidence_strength,
            reasoning=verdict_item.reasoning, is_modified=is_modified,
            modification_aligned=verdict_item.is_modification_aligned,
            assertion_verdicts=assertion_verdicts,
        ))

    # Combine with unmatched verdicts
    test_verdicts.extend(unmatched_verdicts)

    aligned = sum(1 for v in test_verdicts if v.intent_match == TestVerdict.ALIGNED)
    tangential = sum(1 for v in test_verdicts if v.intent_match == TestVerdict.TANGENTIAL)
    unrelated = sum(1 for v in test_verdicts if v.intent_match == TestVerdict.UNRELATED)

    total_assertions = sum(len(v.assertion_verdicts) for v in test_verdicts)
    on_topic = sum(v.on_topic_count for v in test_verdicts)
    off_topic = sum(v.off_topic_count for v in test_verdicts)
    has_modified = any(v.is_modified for v in test_verdicts)

    return TestAnalysis(
        total_tests=len(test_verdicts),
        aligned_count=aligned,
        tangential_count=tangential,
        unrelated_count=unrelated,
        total_assertions=total_assertions,
        on_topic_assertions=on_topic,
        off_topic_assertions=off_topic,
        has_modified_tests=has_modified,
        test_verdicts=test_verdicts,
    )
