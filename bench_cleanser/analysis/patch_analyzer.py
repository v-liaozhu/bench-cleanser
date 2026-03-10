"""Stage 3: Gold patch hunk-by-hunk analysis.

Each hunk in the gold patch is classified as IN_SCOPE, BORDERLINE,
OUT_OF_SCOPE, or INFRASTRUCTURE relative to the task scope determined
in Stage 2.
"""

from __future__ import annotations

import logging
from typing import Any

from bench_cleanser.llm_client import LLMClient
from bench_cleanser.models import (
    ChangedBlock,
    ExcessPatchDetail,
    HunkClassification,
    HunkReport,
    HunkVerdict,
    IntentStatement,
    ParsedTask,
    PatchAnalysis,
    PatchHunk,
    PatchVerdict,
    ScopeAnalysis,
    StructuralDiff,
)
from bench_cleanser.parsing.patch_parser import hunk_is_import_only

logger = logging.getLogger(__name__)

PATCH_SYSTEM_PROMPT = """\
You are an expert software engineer reviewing a code change (patch hunk).
You have been given:
  1. A scope analysis describing what the task actually asks for.
  2. A single hunk from the gold (reference) patch.

Think through this step by step:
1. Read the scope analysis to understand what the task requires.
2. Examine the hunk's file path, function context, and actual code changes.
3. Determine whether each changed line is necessary to fulfill the task scope.
4. Consider whether the change could be a refactor, style fix, or unrelated feature.

Classify this hunk as one of:
  - IN_SCOPE: The change is directly required by the task description.
  - BORDERLINE: The change is arguably related but not strictly required.
  - OUT_OF_SCOPE: The change is unrelated to what the task asks for (e.g.,
    refactoring, style changes, unrelated features, documentation updates).
  - INFRASTRUCTURE: Boilerplate changes required to support in-scope changes
    (e.g., imports, __init__.py updates, test configuration).

Respond in JSON:
{
  "classification": "IN_SCOPE|BORDERLINE|OUT_OF_SCOPE|INFRASTRUCTURE",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<brief explanation>"
}
"""


def _apply_heuristics(hunk: PatchHunk) -> HunkReport | None:
    """Apply deterministic heuristics before LLM classification.

    Returns a HunkReport if the hunk can be classified heuristically,
    or None if it needs LLM analysis.
    """
    # __init__.py with only import changes
    if hunk.is_init_file and hunk_is_import_only(hunk):
        return HunkReport(
            hunk_index=hunk.hunk_index,
            file_path=hunk.file_path,
            classification=HunkClassification.INFRASTRUCTURE,
            confidence=0.95,
            reasoning="__init__.py hunk with only import/export changes",
            is_heuristic=True,
        )

    # Documentation files
    if hunk.is_doc_file:
        return HunkReport(
            hunk_index=hunk.hunk_index,
            file_path=hunk.file_path,
            classification=HunkClassification.OUT_OF_SCOPE,
            confidence=0.90,
            reasoning="Change in documentation/changelog file",
            is_heuristic=True,
        )

    return None


def _build_user_prompt(scope: ScopeAnalysis, hunk: PatchHunk) -> str:
    """Build the user prompt for hunk classification."""
    return (
        f"=== TASK SCOPE ===\n"
        f"Core requirement: {scope.core_requirement}\n"
        f"Behavioral contract: {scope.behavioral_contract}\n"
        f"Affected components: {', '.join(scope.affected_components)}\n"
        f"Out of scope: {scope.out_of_scope}\n\n"
        f"=== PATCH HUNK ===\n"
        f"File: {hunk.file_path}\n"
        f"Function context: {hunk.function_context}\n"
        f"Lines added: {len(hunk.added_lines)}\n"
        f"Lines removed: {len(hunk.removed_lines)}\n\n"
        f"Diff:\n{hunk.raw_diff}\n"
    )


async def _classify_hunk(
    hunk: PatchHunk,
    scope: ScopeAnalysis,
    llm: LLMClient,
) -> HunkReport:
    """Classify a single patch hunk, using heuristics first, then LLM."""
    # Try heuristic classification
    heuristic_result = _apply_heuristics(hunk)
    if heuristic_result is not None:
        return heuristic_result

    # Fall back to LLM
    user_prompt = _build_user_prompt(scope, hunk)
    result = await llm.query_json(PATCH_SYSTEM_PROMPT, user_prompt)

    classification_str = result.get("classification", "BORDERLINE")
    try:
        classification = HunkClassification(classification_str)
    except ValueError:
        classification = HunkClassification.BORDERLINE

    return HunkReport(
        hunk_index=hunk.hunk_index,
        file_path=hunk.file_path,
        classification=classification,
        confidence=float(result.get("confidence", 0.5)),
        reasoning=result.get("reasoning", ""),
        is_heuristic=False,
    )


async def analyze_patch(
    parsed: ParsedTask,
    scope: ScopeAnalysis,
    llm: LLMClient,
) -> PatchAnalysis:
    """Run Stage 3 patch analysis on all hunks in the gold patch.

    Each hunk is classified independently.  Heuristic pre-filters handle
    obvious cases; the rest go to the LLM.
    """
    hunk_reports: list[HunkReport] = []

    for hunk in parsed.patch_hunks:
        report = await _classify_hunk(hunk, scope, llm)
        hunk_reports.append(report)

    in_scope = sum(
        1 for r in hunk_reports if r.classification == HunkClassification.IN_SCOPE
    )
    out_of_scope = sum(
        1 for r in hunk_reports if r.classification == HunkClassification.OUT_OF_SCOPE
    )
    borderline = sum(
        1 for r in hunk_reports if r.classification == HunkClassification.BORDERLINE
    )
    infrastructure = sum(
        1 for r in hunk_reports if r.classification == HunkClassification.INFRASTRUCTURE
    )

    total = len(hunk_reports) or 1
    overpatch_score = (out_of_scope + 0.5 * borderline) / total

    return PatchAnalysis(
        instance_id=parsed.record.instance_id,
        hunk_reports=hunk_reports,
        total_hunks=len(hunk_reports),
        in_scope_count=in_scope,
        out_of_scope_count=out_of_scope,
        borderline_count=borderline,
        infrastructure_count=infrastructure,
        overpatch_score=overpatch_score,
    )


# ═══════════════════════════════════════════════════════════════════════
# v2 Intent Matching: REQUIRED / ANCILLARY / UNRELATED
# ═══════════════════════════════════════════════════════════════════════


PATCH_INTENT_SYSTEM_PROMPT = """\
You are an expert software engineer performing **intent matching** between a
problem description and a code change.

You are given:
  1. The ground truth **intent** extracted from the problem statement — including
     core requirement, behavioral contract, acceptance criteria, and out-of-scope.
  2. A single hunk from the gold (reference) patch with its diff.
  3. Optionally, the full source of the function/class being modified (from
     structural analysis).

Your task: Classify this hunk as one of:

  **REQUIRED**: The change directly implements behavior described in the
    acceptance criteria.  A correct solution MUST include this change (or an
    equivalent).  Examples: fixing the reported bug, adding the requested
    feature, updating the specific function mentioned.

  **ANCILLARY**: The change supports the fix but is NOT described in the problem.
    It is reasonable infrastructure.  Examples: adding imports for new code,
    updating __init__.py exports, adding type annotations for changed functions,
    adjusting configuration that enables the fix.

  **UNRELATED**: The change modifies behavior that is NOT described in the
    problem statement and NOT required to support the fix.  Examples:
    refactoring other code, fixing unrelated bugs, adding unrelated features,
    changing documentation, style changes to untouched code.

Important guidelines:
- Focus on the ACCEPTANCE CRITERIA — each criterion is a specific testable behavior
  from the problem statement.
- A hunk is REQUIRED only if removing it would break at least one acceptance criterion.
- Infrastructure changes (imports, __init__.py, test config) are ANCILLARY, not UNRELATED.
- When uncertain between REQUIRED and ANCILLARY, prefer REQUIRED.
- When uncertain between ANCILLARY and UNRELATED, prefer ANCILLARY.

Respond in JSON:
{
  "verdict": "REQUIRED|ANCILLARY|UNRELATED",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<brief explanation citing specific acceptance criteria if applicable>"
}
"""


def _build_intent_hunk_prompt(
    intent: IntentStatement,
    hunk: PatchHunk,
    structural_context: str = "",
) -> str:
    """Build user prompt for v2 patch intent matching."""
    parts: list[str] = []

    parts.append(
        "=== INTENT (from problem statement only) ===\n"
        f"Core requirement: {intent.core_requirement}\n"
        f"Behavioral contract: {intent.behavioral_contract}\n"
        f"Acceptance criteria:\n"
        + "\n".join(f"  - {c}" for c in intent.acceptance_criteria) + "\n"
        f"Out of scope: {intent.out_of_scope}"
    )

    parts.append(
        "=== PATCH HUNK ===\n"
        f"File: {hunk.file_path}\n"
        f"Function context: {hunk.function_context}\n"
        f"Lines added: {len(hunk.added_lines)}\n"
        f"Lines removed: {len(hunk.removed_lines)}\n\n"
        f"Diff:\n{hunk.raw_diff}"
    )

    if structural_context:
        parts.append(
            "=== STRUCTURAL CONTEXT (full function source) ===\n"
            + structural_context
        )

    return "\n\n".join(parts)


def _apply_heuristics_v2(hunk: PatchHunk) -> HunkVerdict | None:
    """Heuristic pre-filters for v2 patch verdicts."""
    if hunk.is_init_file and hunk_is_import_only(hunk):
        return HunkVerdict(
            hunk_index=hunk.hunk_index,
            file_path=hunk.file_path,
            verdict=PatchVerdict.ANCILLARY,
            confidence=0.95,
            reasoning="__init__.py hunk with only import/export changes",
            is_heuristic=True,
        )

    if hunk.is_doc_file:
        return HunkVerdict(
            hunk_index=hunk.hunk_index,
            file_path=hunk.file_path,
            verdict=PatchVerdict.UNRELATED,
            confidence=0.90,
            reasoning="Change in documentation/changelog file",
            is_heuristic=True,
        )

    return None


async def _classify_hunk_v2(
    hunk: PatchHunk,
    intent: IntentStatement,
    llm: LLMClient,
    structural_diff: StructuralDiff | None = None,
) -> HunkVerdict:
    """Classify a single hunk using v2 intent matching."""
    heuristic = _apply_heuristics_v2(hunk)
    if heuristic is not None:
        return heuristic

    # Find structural context for this hunk
    structural_context = ""
    if structural_diff:
        for cb in structural_diff.changed_blocks:
            if cb.file_path == hunk.file_path and cb.pre_source:
                structural_context += f"--- {cb.block_name} ({cb.block_type}, {cb.edit_status}) ---\n"
                structural_context += cb.pre_source + "\n\n"

    user_prompt = _build_intent_hunk_prompt(intent, hunk, structural_context)
    result = await llm.query_json(PATCH_INTENT_SYSTEM_PROMPT, user_prompt)

    verdict_str = result.get("verdict", "ANCILLARY")
    try:
        verdict = PatchVerdict(verdict_str)
    except ValueError:
        verdict = PatchVerdict.ANCILLARY

    return HunkVerdict(
        hunk_index=hunk.hunk_index,
        file_path=hunk.file_path,
        verdict=verdict,
        confidence=float(result.get("confidence", 0.5)),
        reasoning=result.get("reasoning", ""),
        is_heuristic=False,
    )


async def analyze_patch_v2(
    parsed: ParsedTask,
    intent: IntentStatement,
    llm: LLMClient,
    structural_diff: StructuralDiff | None = None,
) -> ExcessPatchDetail:
    """Run v2 patch intent matching on all hunks in the gold patch.

    Returns ExcessPatchDetail with per-hunk REQUIRED/ANCILLARY/UNRELATED
    verdicts and the computed excess_patch_score.
    """
    hunk_verdicts: list[HunkVerdict] = []

    for hunk in parsed.patch_hunks:
        verdict = await _classify_hunk_v2(hunk, intent, llm, structural_diff)
        hunk_verdicts.append(verdict)

    required = sum(1 for v in hunk_verdicts if v.verdict == PatchVerdict.REQUIRED)
    ancillary = sum(1 for v in hunk_verdicts if v.verdict == PatchVerdict.ANCILLARY)
    unrelated = sum(1 for v in hunk_verdicts if v.verdict == PatchVerdict.UNRELATED)
    total = len(hunk_verdicts) or 1

    # Score: unrelated counts fully, ancillary counts half
    score = (unrelated + 0.5 * ancillary) / total

    return ExcessPatchDetail(
        score=score,
        total_hunks=len(hunk_verdicts),
        required_count=required,
        ancillary_count=ancillary,
        unrelated_count=unrelated,
        hunk_verdicts=hunk_verdicts,
    )
