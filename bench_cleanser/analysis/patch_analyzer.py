"""Stage 4A: Gold patch intent matching (batched).

All hunks in the gold patch are classified as REQUIRED, ANCILLARY, or
UNRELATED in a SINGLE batched LLM call against the intent from Stage 2.
Structural context from Stage 3 is included when available.

Uses structured output with strict JSON schema enforcement.
No regex heuristics — all classification goes through the LLM.
"""

from __future__ import annotations

import logging

from bench_cleanser.llm_client import LLMClient
from bench_cleanser.models import (
    HunkVerdict,
    IntentStatement,
    ParsedTask,
    PatchAnalysis,
    PatchHunk,
    PatchVerdict,
    StructuralDiff,
)
from bench_cleanser.prompts import load as _load_prompt
from bench_cleanser.schemas import BatchPatchVerdictsResponse

logger = logging.getLogger(__name__)

BATCH_PATCH_SYSTEM_PROMPT = _load_prompt("patch_classifier")


def _build_batch_patch_prompt(
    intent: IntentStatement,
    hunks: list[PatchHunk],
    structural_diff: StructuralDiff | None = None,
) -> str:
    """Build user prompt with all hunks for batch classification."""
    parts: list[str] = []

    # Intent section
    parts.append(
        "=== INTENT (from problem statement only — no gold patch was seen) ===\n"
        f"Core requirement: {intent.core_requirement}\n\n"
        f"Behavioral contract: {intent.behavioral_contract}\n\n"
        "Acceptance criteria:\n"
        + "\n".join(f"  {i+1}. {c}" for i, c in enumerate(intent.acceptance_criteria)) + "\n\n"
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

    # All hunks
    for i, hunk in enumerate(hunks):
        hunk_section = (
            f"=== HUNK {i} ===\n"
            f"File: {hunk.file_path}\n"
            f"Function context: {hunk.function_context}\n"
            f"Is test file: {hunk.is_test_file}\n"
            f"Is __init__.py: {hunk.is_init_file}\n"
            f"Is doc/changelog: {hunk.is_doc_file}\n"
            f"Lines added: {len(hunk.added_lines)}\n"
            f"Lines removed: {len(hunk.removed_lines)}\n\n"
            f"Diff:\n{hunk.raw_diff}"
        )

        # Add structural context if available
        if structural_diff:
            for cb in structural_diff.changed_blocks:
                if cb.file_path == hunk.file_path and cb.pre_source:
                    hunk_section += (
                        f"\n\nFull function source (pre-patch):\n"
                        f"--- {cb.block_name} ({cb.block_type}, {cb.edit_status}) ---\n"
                        f"{cb.pre_source}"
                    )

        parts.append(hunk_section)

    return "\n\n".join(parts)


async def analyze_patch(
    parsed: ParsedTask,
    intent: IntentStatement,
    llm: LLMClient,
    structural_diff: StructuralDiff | None = None,
) -> PatchAnalysis:
    """Stage 4A: classify all gold patch hunks against intent in a single batched LLM call."""
    if not parsed.patch_hunks:
        return PatchAnalysis(
            total_hunks=0, required_count=0,
            ancillary_count=0, unrelated_count=0,
        )

    user_prompt = _build_batch_patch_prompt(
        intent, parsed.patch_hunks, structural_diff,
    )

    result: BatchPatchVerdictsResponse = await llm.query_structured(
        BATCH_PATCH_SYSTEM_PROMPT,
        user_prompt,
        BatchPatchVerdictsResponse,
    )

    # Map results to HunkVerdict objects
    hunk_verdicts: list[HunkVerdict] = []
    result_by_index = {v.hunk_index: v for v in result.verdicts}

    for i, hunk in enumerate(parsed.patch_hunks):
        verdict_item = result_by_index.get(i)
        if verdict_item is None:
            logger.warning(
                "Missing verdict for hunk %d (%s) — defaulting to ANCILLARY",
                i, hunk.file_path,
            )
            hunk_verdicts.append(HunkVerdict(
                hunk_index=i, file_path=hunk.file_path,
                verdict=PatchVerdict.ANCILLARY, evidence_strength="weak",
                reasoning="No verdict returned by LLM for this hunk",
            ))
            continue

        try:
            verdict = PatchVerdict(verdict_item.verdict)
        except ValueError:
            verdict = PatchVerdict.ANCILLARY

        hunk_verdicts.append(HunkVerdict(
            hunk_index=i, file_path=hunk.file_path,
            verdict=verdict, evidence_strength=verdict_item.evidence_strength,
            reasoning=verdict_item.reasoning,
        ))

    required = sum(1 for v in hunk_verdicts if v.verdict == PatchVerdict.REQUIRED)
    ancillary = sum(1 for v in hunk_verdicts if v.verdict == PatchVerdict.ANCILLARY)
    unrelated = sum(1 for v in hunk_verdicts if v.verdict == PatchVerdict.UNRELATED)

    return PatchAnalysis(
        total_hunks=len(hunk_verdicts),
        required_count=required,
        ancillary_count=ancillary,
        unrelated_count=unrelated,
        hunk_verdicts=hunk_verdicts,
    )
