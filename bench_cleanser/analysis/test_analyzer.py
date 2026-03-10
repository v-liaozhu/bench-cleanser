"""Stage 4: Test patch analysis.

Combines deterministic new-vs-modified detection with LLM scope alignment
and AST-based assertion analysis.  When code visitation data (CodeContext)
is available the LLM receives full pre-/post-patch test source, the source
code of functions being tested, structured assertions, and call analysis.
"""

from __future__ import annotations

import ast
import logging

from bench_cleanser.llm_client import LLMClient
from bench_cleanser.models import (
    AssertionVerdict,
    AssertionVerdictReport,
    CodeContext,
    ExcessTestDetail,
    IntentStatement,
    ParsedTask,
    ScopeAnalysis,
    StructuralDiff,
    TestAnalysis,
    TestClassification,
    TestHunk,
    TestModificationType,
    TestReport,
    TestVerdict,
    TestVerdictReport,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

# Enhanced prompt used when full code context is available
TEST_ALIGNMENT_SYSTEM_PROMPT_ENHANCED = """\
You are a senior open-source maintainer and expert code reviewer.  Your job
is to assess whether an F2P (fail-to-pass) test is **legitimately aligned**
with the task's problem statement or whether it tests behavior the task
never asked for.

You are given:
  1. The task scope — core requirement, behavioral contract, affected
     components, and what is explicitly out of scope.
  2. The FULL pre-patch test (if the test existed before the PR).
  3. The FULL post-patch test (after the test_patch is applied).
  4. A clear diff showing what was added/removed.
  5. The actual source code of functions the test calls that live in files
     modified by the gold patch.
  6. A structured list of every assertion in the post-patch test.
  7. A call analysis listing every function the test invokes and whether
     that function lives in a gold-patch file.

**Analysis steps (follow carefully):**

A) READ the task scope.  Form a mental model of what behavior the fix/feature
   MUST deliver and nothing more.

B) If the test is MODIFIED (existed before the PR):
   - Compare the pre-patch and post-patch test source line by line.
   - Identify EXACTLY which lines were changed.
   - For each change, ask: "Does the problem statement require this change?"
   - If the modification adds assertions on behavior NOT described in the
     problem statement, that is a **SNEAKY_MODIFICATION** — the test looks
     legitimate because it already existed, but the PR author silently
     altered it to assert on new behavior the task never asked for.

C) For EACH assertion in the test:
   - Trace what source function it exercises (use the call analysis and
     tested source code).
   - Determine: does the problem statement require verifying this behavior?
   - Mark the assertion as IN-SCOPE or OUT-OF-SCOPE.

D) For each tested function that is modified by the gold patch:
   - Read its source code.
   - Ask: does the test verify the *described* fix in this function, or
     does it verify *other* behavioral changes introduced by the patch?

E) Classify the overall test:
   - ALIGNED: Every assertion directly verifies the described fix/feature.
   - PARTIALLY_ALIGNED: Most assertions are in-scope but ≥1 checks behavior
     beyond the problem statement.
   - MISALIGNED: The test primarily verifies behavior NOT described in the
     problem statement.
   - SNEAKY_MODIFICATION: The test existed before and was modified to assert
     on changed behavior not described in the task.  This is the most
     insidious form of contamination.

Respond in JSON:
{
  "classification": "ALIGNED|PARTIALLY_ALIGNED|MISALIGNED|SNEAKY_MODIFICATION",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<concise but specific; cite lines/assertions/functions>",
  "misaligned_assertions": <int>,
  "total_assertions": <int>,
  "key_changes_from_original": "<summary of what changed, or 'N/A' for NEW tests>"
}
"""

# Fallback prompt when no code context is available (diff-only)
TEST_ALIGNMENT_SYSTEM_PROMPT_BASIC = """\
You are an expert software engineer reviewing a test function from a patch.
You have been given:
  1. A scope analysis describing what the task actually asks for.
  2. The source code of a test function from the F2P (fail-to-pass) test set.
  3. Whether the test is NEW (added by the patch) or MODIFIED (existed before
     and was changed by the patch).

Think through this step by step:
1. Review the scope analysis to understand what behavior the task requires.
2. Read the test function carefully, identifying each assertion.
3. For each assertion, determine whether it tests behavior described in the
   problem statement or behavior that goes beyond it.
4. If the test is MODIFIED, pay special attention: what was changed from the
   original test? Does the modification test new out-of-scope behavior?
5. Count how many assertions check out-of-scope behavior.

Classify the test as one of:
  - ALIGNED: The test directly verifies the described fix/feature.
  - PARTIALLY_ALIGNED: The test mostly verifies described behavior but
    includes some assertions beyond scope.
  - MISALIGNED: The test verifies behavior NOT described in the problem
    statement.
  - SNEAKY_MODIFICATION: The test existed before and was modified to
    assert on changed behavior not described in the task.

Respond in JSON:
{
  "classification": "ALIGNED|PARTIALLY_ALIGNED|MISALIGNED|SNEAKY_MODIFICATION",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<brief explanation>",
  "misaligned_assertions": <int, count of assertions checking out-of-scope behavior>
}
"""


# ---------------------------------------------------------------------------
# Assertion counting (local AST)
# ---------------------------------------------------------------------------

def _count_assertions(source: str) -> tuple[int, list[str]]:
    """Count assert statements in test source via AST parsing.

    Returns (total_count, list_of_assertion_strings).
    """
    # Clean up the source for parsing -- remove diff prefixes
    lines = []
    for line in source.splitlines():
        clean = line
        if clean.startswith("+"):
            clean = clean[1:]
        lines.append(clean)
    cleaned = "\n".join(lines)

    assertions: list[str] = []
    try:
        tree = ast.parse(cleaned)
    except SyntaxError:
        # Fall back to regex
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("assert"):
                assertions.append(stripped)
        return len(assertions), assertions

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
            if name.startswith("assert") or name in (
                "assertEqual",
                "assertRaises",
                "assertTrue",
                "assertFalse",
                "assertIn",
                "assertNotIn",
                "assertIsNone",
                "assertIsNotNone",
                "assertAlmostEqual",
                "assertGreater",
                "assertLess",
                "assertRegex",
                "assertNotEqual",
                "assertIs",
                "assertIsNot",
                "assertCountEqual",
                "assertSequenceEqual",
                "assertListEqual",
                "assertDictEqual",
            ):
                try:
                    assertions.append(ast.unparse(node))
                except Exception:
                    assertions.append(f"{name}() at line {node.lineno}")

    return len(assertions), assertions


# ---------------------------------------------------------------------------
# Prompt building — enhanced (with CodeContext)
# ---------------------------------------------------------------------------

def _truncate(text: str, max_lines: int = 200) -> str:
    """Truncate source text to *max_lines*."""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[:max_lines]) + f"\n[...truncated, {len(lines) - max_lines} more lines...]"


def _build_enhanced_prompt(
    scope: ScopeAnalysis,
    test_hunk: TestHunk,
    ctx: CodeContext,
    problem_statement: str = "",
) -> str:
    """Build a rich user prompt using full code context."""
    parts: list[str] = []

    # --- Task scope ---
    parts.append(
        f"=== TASK SCOPE ===\n"
        f"Core requirement: {scope.core_requirement}\n"
        f"Behavioral contract: {scope.behavioral_contract}\n"
        f"Affected components: {', '.join(scope.affected_components)}\n"
        f"Out of scope: {scope.out_of_scope}\n"
        f"Ambiguity score: {scope.ambiguity_score:.2f}"
    )

    # --- Problem statement (truncated) ---
    if problem_statement:
        ps_trunc = _truncate(problem_statement, max_lines=40)
        parts.append(f"=== PROBLEM STATEMENT ===\n{ps_trunc}")

    # --- Test metadata ---
    parts.append(
        f"=== TEST METADATA ===\n"
        f"File: {test_hunk.file_path}\n"
        f"Test name: {test_hunk.test_name}\n"
        f"Test ID: {test_hunk.full_test_id}\n"
        f"Modification type: {test_hunk.modification_type.value}"
    )

    # --- Full pre-patch test ---
    if ctx.pre_patch_test_source:
        parts.append(
            f"=== FULL PRE-PATCH TEST (before this PR) ===\n"
            f"{_truncate(ctx.pre_patch_test_source)}"
        )
    else:
        parts.append(
            "=== FULL PRE-PATCH TEST ===\n"
            "[Not found — test is likely NEW, did not exist before the PR]"
        )

    # --- Full post-patch test ---
    if ctx.post_patch_test_source:
        parts.append(
            f"=== FULL POST-PATCH TEST (after this PR) ===\n"
            f"{_truncate(ctx.post_patch_test_source)}"
        )
    elif test_hunk.full_source:
        parts.append(
            f"=== POST-PATCH TEST SOURCE (from diff) ===\n"
            f"{_truncate(test_hunk.full_source)}"
        )

    # --- Diff ---
    diff_lines: list[str] = []
    if test_hunk.removed_lines:
        for line in test_hunk.removed_lines:
            diff_lines.append(f"- {line}")
    if test_hunk.added_lines:
        for line in test_hunk.added_lines:
            diff_lines.append(f"+ {line}")
    if diff_lines:
        parts.append(
            f"=== DIFF (what changed in the test) ===\n"
            + "\n".join(diff_lines)
        )

    # --- Test context ---
    context_parts: list[str] = []
    if ctx.test_file_imports:
        context_parts.append(f"Imports:\n{_truncate(ctx.test_file_imports, 30)}")
    if ctx.test_file_fixtures:
        context_parts.append(f"Fixtures/setup:\n{_truncate(ctx.test_file_fixtures, 40)}")
    if context_parts:
        parts.append("=== TEST CONTEXT ===\n" + "\n\n".join(context_parts))

    # --- Tested source code ---
    if ctx.tested_functions:
        tf_parts: list[str] = []
        for tf in ctx.tested_functions:
            tf_entry = (
                f"Function: {tf.name}\n"
                f"File: {tf.file_path}\n"
                f"Modified by gold patch: {'YES' if tf.is_modified_by_patch else 'NO'}\n"
            )
            if tf.source:
                tf_entry += f"Source (pre-patch):\n{_truncate(tf.source)}"
            else:
                tf_entry += "Source: [not available]"
            tf_parts.append(tf_entry)
        parts.append("=== TESTED SOURCE CODE ===\n" + "\n\n---\n".join(tf_parts))

    # --- Call analysis ---
    if ctx.call_targets:
        call_lines: list[str] = []
        for ct in ctx.call_targets:
            patch_tag = "IN GOLD PATCH" if ct.is_in_patch else "not in patch"
            loc = f" in {ct.file_path}" if ct.file_path else ""
            call_lines.append(f"  - {ct.name}{loc} ({patch_tag})")
        parts.append(
            "=== CALL ANALYSIS ===\n"
            "Functions called by this test:\n" + "\n".join(call_lines)
        )

    # --- Structured assertions ---
    if ctx.assertions:
        assert_lines: list[str] = []
        for a in ctx.assertions:
            entry = f"  - [{a.assertion_type}] {a.target_expression}"
            if a.expected_value:
                entry += f" == {a.expected_value}"
            assert_lines.append(entry)
        parts.append(
            "=== ASSERTIONS IN POST-PATCH TEST ===\n" + "\n".join(assert_lines)
        )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Prompt building — basic (diff-only, no CodeContext)
# ---------------------------------------------------------------------------

def _build_basic_prompt(
    scope: ScopeAnalysis,
    test_hunk: TestHunk,
    problem_statement: str = "",
) -> str:
    """Build the user prompt when no CodeContext is available."""
    text = (
        f"=== TASK SCOPE ===\n"
        f"Core requirement: {scope.core_requirement}\n"
        f"Behavioral contract: {scope.behavioral_contract}\n"
        f"Affected components: {', '.join(scope.affected_components)}\n"
        f"Out of scope: {scope.out_of_scope}\n\n"
    )

    if problem_statement:
        ps_trunc = _truncate(problem_statement, max_lines=30)
        text += f"=== PROBLEM STATEMENT ===\n{ps_trunc}\n\n"

    text += (
        f"=== TEST FUNCTION ===\n"
        f"File: {test_hunk.file_path}\n"
        f"Test name: {test_hunk.test_name}\n"
        f"Test ID: {test_hunk.full_test_id}\n"
        f"Modification type: {test_hunk.modification_type.value}\n\n"
        f"Test source (from added lines):\n{test_hunk.full_source}\n"
    )

    if test_hunk.removed_lines:
        text += (
            f"\n\nRemoved lines (original test code):\n"
            + "\n".join(test_hunk.removed_lines)
        )

    return text


# ---------------------------------------------------------------------------
# Single test analysis
# ---------------------------------------------------------------------------

async def _analyze_single_test(
    test_hunk: TestHunk,
    scope: ScopeAnalysis,
    llm: LLMClient,
    problem_statement: str = "",
) -> TestReport:
    """Analyze a single F2P test for scope alignment."""
    # Deterministic signal: if the test is MODIFIED, it's a strong C3 indicator
    is_modified = test_hunk.modification_type == TestModificationType.MODIFIED

    # Count assertions via AST
    assertion_count, _assertion_strs = _count_assertions(test_hunk.full_source)

    # Choose prompt strategy based on code context availability
    ctx = test_hunk.code_context
    if ctx is not None:
        system_prompt = TEST_ALIGNMENT_SYSTEM_PROMPT_ENHANCED
        user_prompt = _build_enhanced_prompt(
            scope, test_hunk, ctx,
            problem_statement=problem_statement,
        )
        logger.info(
            "Enhanced prompt for %s: %d tested funcs, %d assertions, "
            "pre-patch=%s, post-patch=%s",
            test_hunk.test_name,
            len(ctx.tested_functions),
            len(ctx.assertions),
            "yes" if ctx.pre_patch_test_source else "no",
            "yes" if ctx.post_patch_test_source else "no",
        )
    else:
        system_prompt = TEST_ALIGNMENT_SYSTEM_PROMPT_BASIC
        user_prompt = _build_basic_prompt(
            scope, test_hunk,
            problem_statement=problem_statement,
        )
        logger.info(
            "Basic prompt for %s (no code context available)",
            test_hunk.test_name,
        )

    result = await llm.query_json(system_prompt, user_prompt)

    classification_str = result.get("classification", "PARTIALLY_ALIGNED")
    misaligned_count = int(result.get("misaligned_assertions", 0))

    # If deterministic detection says MODIFIED but LLM says ALIGNED
    # and found no misaligned assertions, the modification is benign —
    # keep it as ALIGNED (don't punish aligned test updates).
    # Otherwise, override toward SNEAKY_MODIFICATION.
    if is_modified:
        if classification_str == "ALIGNED" and misaligned_count == 0:
            # LLM confirms modification is fully aligned — trust it
            pass
        elif classification_str == "ALIGNED":
            classification_str = "PARTIALLY_ALIGNED"
        if classification_str in ("MISALIGNED", "PARTIALLY_ALIGNED"):
            classification_str = "SNEAKY_MODIFICATION"

    try:
        classification = TestClassification(classification_str)
    except ValueError:
        classification = TestClassification.PARTIALLY_ALIGNED

    return TestReport(
        test_id=test_hunk.full_test_id,
        test_name=test_hunk.test_name,
        modification_type=test_hunk.modification_type,
        classification=classification,
        confidence=float(result.get("confidence", 0.5)),
        reasoning=result.get("reasoning", ""),
        is_modified_existing=is_modified,
        assertion_count=assertion_count,
        misaligned_assertion_count=misaligned_count,
    )


async def _analyze_unmatched_test(
    test_id: str,
    scope: ScopeAnalysis,
    llm: LLMClient,
) -> TestReport:
    """Create a report for an F2P test with no matching test hunk.

    These tests exist in the F2P list but were NOT modified in the test patch.
    This is interesting: the test existed before the PR, passed before, and
    now must fail-to-pass. This could mean the gold patch changes behavior
    that this test depends on.
    """
    parts = test_id.split("::")
    test_name = parts[-1].split("[")[0] if parts else test_id

    return TestReport(
        test_id=test_id,
        test_name=test_name,
        modification_type=TestModificationType.UNKNOWN,
        classification=TestClassification.ALIGNED,
        confidence=0.3,
        reasoning=(
            "F2P test with no matching hunk in test_patch — "
            "test was not modified, likely exercises new behavior from the gold patch"
        ),
        is_modified_existing=False,
        assertion_count=0,
        misaligned_assertion_count=0,
    )


# ---------------------------------------------------------------------------
# Stage 4 entry point
# ---------------------------------------------------------------------------

async def analyze_tests(
    parsed: ParsedTask,
    scope: ScopeAnalysis,
    llm: LLMClient,
) -> TestAnalysis:
    """Run Stage 4 test analysis on all F2P tests.

    Combines deterministic new-vs-modified detection with LLM scope
    alignment analysis and AST-based assertion counting.
    """
    test_reports: list[TestReport] = []
    problem_statement = parsed.record.problem_statement

    # Analyze F2P tests that have matching hunks
    for test_hunk in parsed.f2p_test_hunks:
        report = await _analyze_single_test(
            test_hunk, scope, llm,
            problem_statement=problem_statement,
        )
        test_reports.append(report)

    # Analyze F2P tests without matching hunks
    for test_id in parsed.f2p_tests_with_no_hunk:
        report = await _analyze_unmatched_test(test_id, scope, llm)
        test_reports.append(report)

    aligned = sum(
        1
        for r in test_reports
        if r.classification == TestClassification.ALIGNED
    )
    misaligned = sum(
        1
        for r in test_reports
        if r.classification
        in (TestClassification.MISALIGNED, TestClassification.PARTIALLY_ALIGNED)
    )
    sneaky = sum(
        1
        for r in test_reports
        if r.classification == TestClassification.SNEAKY_MODIFICATION
    )

    total = len(test_reports) or 1
    overtest_score = (misaligned + sneaky) / total
    sneaky_score = sneaky / total

    return TestAnalysis(
        instance_id=parsed.record.instance_id,
        test_reports=test_reports,
        total_f2p_tests=len(test_reports),
        aligned_count=aligned,
        misaligned_count=misaligned,
        sneaky_mod_count=sneaky,
        overtest_score=overtest_score,
        sneaky_test_mod_score=sneaky_score,
    )


# ═══════════════════════════════════════════════════════════════════════
# v2 Intent Matching: ALIGNED / TANGENTIAL / UNRELATED + per-assertion
# ═══════════════════════════════════════════════════════════════════════


TEST_INTENT_SYSTEM_PROMPT = """\
You are an expert software engineer performing **intent matching** between a
problem description and a test function.

You are given:
  1. The ground truth **intent** extracted from the problem statement — including
     core requirement, behavioral contract, acceptance criteria, and out-of-scope.
  2. The full source of an F2P (fail-to-pass) test function.
  3. A list of assertions extracted from the test, numbered for reference.
  4. Whether the test is NEW (added by the patch) or MODIFIED (existed before).
  5. Optionally, structural context (call graph, changed functions).

Your task has TWO parts:

**PART 1 — Test-level verdict:**
Classify the test as a whole:

  **ALIGNED**: The test as a whole targets the described problem.  Its primary
    purpose is to verify behavior from the acceptance criteria.

  **TANGENTIAL**: The test partially targets the problem but includes
    significant behavior beyond the acceptance criteria.  A test that checks
    the described fix AND also checks unrelated inheritance/edge cases is
    TANGENTIAL.

  **UNRELATED**: The test does not target the described problem at all.

**PART 2 — Per-assertion verdicts:**
For EACH assertion (by index), classify:

  **ON_TOPIC**: This assertion checks behavior described in the acceptance
    criteria or directly required to verify the fix.

  **OFF_TOPIC**: This assertion checks behavior NOT described in the problem
    statement.  Examples: checking inheritance behavior when the problem only
    asks about a specific attribute, checking error messages when the problem
    only asks about correct behavior, checking side effects not mentioned.

Important guidelines:
- Focus on the ACCEPTANCE CRITERIA.  An assertion is ON_TOPIC only if it
  verifies (or directly supports verification of) at least one criterion.
- If the test was MODIFIED from a pre-existing test, note whether the
  modifications are aligned with the acceptance criteria.
- Be conservative with ON_TOPIC — if an assertion checks behavior that is
  plausible but not described, it is OFF_TOPIC.

Respond in JSON:
{
  "test_verdict": "ALIGNED|TANGENTIAL|UNRELATED",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<concise explanation>",
  "is_modification_aligned": <bool, true if modifications target described behavior>,
  "assertion_verdicts": [
    {"index": 0, "verdict": "ON_TOPIC|OFF_TOPIC", "reason": "<brief reason>"},
    {"index": 1, "verdict": "ON_TOPIC|OFF_TOPIC", "reason": "<brief reason>"},
    ...
  ]
}
"""


def _build_test_intent_prompt(
    intent: IntentStatement,
    test_hunk: TestHunk,
    assertions: list[str],
    structural_context: str = "",
    problem_statement: str = "",
) -> str:
    """Build the user prompt for v2 test intent matching."""
    parts: list[str] = []

    parts.append(
        "=== INTENT (from problem statement only) ===\n"
        f"Core requirement: {intent.core_requirement}\n"
        f"Behavioral contract: {intent.behavioral_contract}\n"
        f"Acceptance criteria:\n"
        + "\n".join(f"  - {c}" for c in intent.acceptance_criteria) + "\n"
        f"Out of scope: {intent.out_of_scope}"
    )

    if problem_statement:
        ps = _truncate(problem_statement, max_lines=30)
        parts.append(f"=== PROBLEM STATEMENT ===\n{ps}")

    parts.append(
        "=== TEST METADATA ===\n"
        f"File: {test_hunk.file_path}\n"
        f"Test name: {test_hunk.test_name}\n"
        f"Test ID: {test_hunk.full_test_id}\n"
        f"Modification type: {test_hunk.modification_type.value}"
    )

    # Pre/post source if available
    ctx = test_hunk.code_context
    if ctx and ctx.pre_patch_test_source:
        parts.append(
            f"=== PRE-PATCH TEST (before this PR) ===\n"
            f"{_truncate(ctx.pre_patch_test_source)}"
        )

    parts.append(
        f"=== POST-PATCH TEST SOURCE ===\n"
        f"{_truncate(test_hunk.full_source)}"
    )

    # Numbered assertions
    if assertions:
        assert_lines = [f"  [{i}] {a}" for i, a in enumerate(assertions)]
        parts.append(
            "=== ASSERTIONS (number each verdict by index) ===\n"
            + "\n".join(assert_lines)
        )

    if structural_context:
        parts.append(
            "=== STRUCTURAL CONTEXT ===\n" + structural_context
        )

    return "\n\n".join(parts)


async def _analyze_test_v2(
    test_hunk: TestHunk,
    intent: IntentStatement,
    llm: LLMClient,
    structural_diff: StructuralDiff | None = None,
    problem_statement: str = "",
) -> TestVerdictReport:
    """Analyze a single F2P test using v2 intent matching."""
    is_modified = test_hunk.modification_type == TestModificationType.MODIFIED

    # Count assertions
    _count, assertion_strs = _count_assertions(test_hunk.full_source)

    # Build structural context
    structural_context = ""
    if structural_diff:
        # Find call edges for this test
        for edge_test, edge_func in structural_diff.call_edges:
            if edge_test == test_hunk.test_name:
                for cb in structural_diff.changed_blocks:
                    if cb.block_name == edge_func:
                        structural_context += (
                            f"Calls → {cb.block_name} ({cb.block_type}, {cb.edit_status}) "
                            f"in {cb.file_path}\n"
                        )

    user_prompt = _build_test_intent_prompt(
        intent, test_hunk, assertion_strs,
        structural_context=structural_context,
        problem_statement=problem_statement,
    )

    result = await llm.query_json(TEST_INTENT_SYSTEM_PROMPT, user_prompt)

    # Parse test-level verdict
    verdict_str = result.get("test_verdict", "TANGENTIAL")
    try:
        test_verdict = TestVerdict(verdict_str)
    except ValueError:
        test_verdict = TestVerdict.TANGENTIAL

    # Parse per-assertion verdicts
    assertion_verdicts: list[AssertionVerdictReport] = []
    raw_verdicts = result.get("assertion_verdicts", [])
    for i, assertion_str in enumerate(assertion_strs):
        # Find matching verdict from LLM response
        av = AssertionVerdict.ON_TOPIC  # default
        reason = ""
        for rv in raw_verdicts:
            if rv.get("index") == i:
                try:
                    av = AssertionVerdict(rv.get("verdict", "ON_TOPIC"))
                except ValueError:
                    av = AssertionVerdict.ON_TOPIC
                reason = rv.get("reason", "")
                break

        assertion_verdicts.append(AssertionVerdictReport(
            statement=assertion_str,
            verdict=av,
            reason=reason,
        ))

    modification_aligned = result.get("is_modification_aligned", True)

    return TestVerdictReport(
        test_id=test_hunk.full_test_id,
        test_name=test_hunk.test_name,
        intent_match=test_verdict,
        confidence=float(result.get("confidence", 0.5)),
        reasoning=result.get("reasoning", ""),
        is_modified=is_modified,
        modification_aligned=modification_aligned,
        assertion_verdicts=assertion_verdicts,
    )


async def _analyze_unmatched_test_v2(
    test_id: str,
    intent: IntentStatement,
) -> TestVerdictReport:
    """Create a verdict for an F2P test with no matching test hunk."""
    parts = test_id.split("::")
    test_name = parts[-1].split("[")[0] if parts else test_id

    return TestVerdictReport(
        test_id=test_id,
        test_name=test_name,
        intent_match=TestVerdict.ALIGNED,
        confidence=0.3,
        reasoning=(
            "F2P test with no matching hunk in test_patch — "
            "test was not modified, likely exercises new behavior from the gold patch"
        ),
        is_modified=False,
        modification_aligned=True,
        assertion_verdicts=[],
    )


async def analyze_tests_v2(
    parsed: ParsedTask,
    intent: IntentStatement,
    llm: LLMClient,
    structural_diff: StructuralDiff | None = None,
) -> ExcessTestDetail:
    """Run v2 test intent matching on all F2P tests.

    Returns ExcessTestDetail with per-test ALIGNED/TANGENTIAL/UNRELATED
    verdicts, per-assertion ON_TOPIC/OFF_TOPIC, and the computed excess_test_score.
    """
    test_verdicts: list[TestVerdictReport] = []
    problem_statement = parsed.record.problem_statement

    # Analyze F2P tests with matching hunks
    for test_hunk in parsed.f2p_test_hunks:
        verdict = await _analyze_test_v2(
            test_hunk, intent, llm, structural_diff,
            problem_statement=problem_statement,
        )
        test_verdicts.append(verdict)

    # Analyze F2P tests without matching hunks
    for test_id in parsed.f2p_tests_with_no_hunk:
        verdict = await _analyze_unmatched_test_v2(test_id, intent)
        test_verdicts.append(verdict)

    # Compute counts
    aligned = sum(1 for v in test_verdicts if v.intent_match == TestVerdict.ALIGNED)
    tangential = sum(1 for v in test_verdicts if v.intent_match == TestVerdict.TANGENTIAL)
    unrelated = sum(1 for v in test_verdicts if v.intent_match == TestVerdict.UNRELATED)

    total_assertions = sum(len(v.assertion_verdicts) for v in test_verdicts) or 1
    on_topic = sum(v.on_topic_count for v in test_verdicts)
    off_topic = sum(v.off_topic_count for v in test_verdicts)

    has_modified = any(v.is_modified for v in test_verdicts)

    # Compute score:
    # off_topic assertions count directly
    # unrelated tests contribute their share of assertions too
    avg_assertions_per_test = total_assertions / (len(test_verdicts) or 1)
    unrelated_assertion_equiv = unrelated * avg_assertions_per_test
    score = min(1.0, (off_topic + unrelated_assertion_equiv) / total_assertions)

    return ExcessTestDetail(
        score=score,
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
