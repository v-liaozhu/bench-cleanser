"""Dual Taxonomy classifier: Axis 1 (task labels) + Axis 2 (agent labels).

Axis 1 assigns zero or more TaskContaminationLabel to each task.
Axis 2 assigns a single AgentTrajectoryLabel per agent-task pair.

7 binary labels, bucket-based severity, no ratio thresholds.

Taxonomy alignment with OpenAI's SWE-bench Verified audit (April 2026):
  - "Narrow test cases" (35.5% of audited failures) -> APPROACH_LOCK
  - "Wide test cases" (18.8% of audited failures)   -> OVER_TEST
  - Training contamination (gold patch memorization) -> Axis 2 agent_passed_leak

Uses structured output with strict JSON schema enforcement.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from bench_cleanser.analysis.cross_ref import CrossReferenceResult
from bench_cleanser.models import (
    DescriptionClarity,
    IntentStatement,
    PatchAnalysis,
    PatchVerdict,
    Severity,
    TaskContaminationLabel,
    TaskLabelAssignment,
    TaskRecord,
    TestAnalysis,
)
from bench_cleanser.prompts import load as _load_prompt
from bench_cleanser.schemas import TaskClassificationResponse

logger = logging.getLogger(__name__)


LABEL_DEFINITIONS: dict[str, dict[str, Any]] = {
    "approach_lock": {
        "display": "Approach Lock",
        "definition": (
            "The F2P tests require a specific implementation approach that "
            "the problem statement does not determine.  An agent that solves "
            "the described problem using a different valid approach will fail "
            "the tests."
        ),
        "openai_equivalent": "Narrow test cases",
        "prompt": (
            "Would a correct-but-different solution fail the F2P tests?  "
            "Do the tests assert on implementation details (specific class, "
            "method, or code structure) rather than on observable behavior?  "
            "Does the gold patch take a fundamentally different approach "
            "than the problem statement suggests?"
        ),
    },
    "over_test": {
        "display": "Over Test",
        "definition": (
            "F2P tests verify behavior or features that the problem "
            "statement does not describe.  The tests go beyond the stated "
            "acceptance criteria — they test additional functionality, "
            "edge cases, or code paths not mentioned.  Includes tests "
            "enforcing features the problem explicitly defers, and "
            "pre-existing tests modified to assert on behavior beyond "
            "what the problem description asks for."
        ),
        "openai_equivalent": "Wide test cases",
        "prompt": (
            "Do the F2P tests assert on behavior not described in the "
            "problem?  Is there at least one test or assertion that targets "
            "undescribed behavior?  Does the problem contain deferral "
            "language yet F2P tests exercise the deferred feature?  "
            "Were any pre-existing tests modified to introduce assertions "
            "beyond what the problem description requires?"
        ),
    },
    "over_patch": {
        "display": "Over Patch",
        "definition": (
            "The gold patch contains behavioral code changes beyond what "
            "the problem asks for — new features, unrelated refactoring, "
            "scope expansion.  Pure ancillary changes (imports, whitespace, "
            "docstrings) do NOT count; only behavioral excess matters."
        ),
        "prompt": (
            "Does the gold patch modify code that the problem doesn't ask "
            "for?  Are there hunks introducing NEW behavior beyond problem "
            "scope?  Ignore purely ancillary changes (imports, whitespace, "
            "docstrings).  Does the patch modify broader scope (base class, "
            "public API) than the problem describes?"
        ),
    },
    "unclear_description": {
        "display": "Unclear Description",
        "definition": (
            "The problem description is too ambiguous or actively misleading "
            "to determine the correct solution.  Either key information is "
            "missing (no repro steps, no affected component, multiple valid "
            "interpretations) or the description points toward the wrong fix."
        ),
        "prompt": (
            "Can a competent developer determine the correct fix from the "
            "problem description alone?  Is the problem ambiguous enough that "
            "multiple incompatible approaches are equally reasonable?  Does "
            "the problem suggest a specific fix strategy that is incorrect "
            "per the gold patch?"
        ),
    },
    "hidden_context": {
        "display": "Hidden Context",
        "definition": (
            "Essential information needed to solve the problem exists only "
            "in the hints text (code review comments, maintainer decisions) "
            "and not in the problem description.  The problem alone is "
            "insufficient; the hints contain the actual specification."
        ),
        "prompt": (
            "Does the hints text contain solution-critical information "
            "absent from the problem?  Function names, root cause, or "
            "design decisions not derivable from the problem alone?"
        ),
    },
    "weak_coverage": {
        "display": "Weak Coverage",
        "definition": (
            "The F2P tests or gold patch don't fully cover the stated "
            "acceptance criteria.  A partial or incorrect fix can pass.  "
            "This makes the task easier (not harder) — it's a benchmark "
            "quality issue, not a fairness issue."
        ),
        "prompt": (
            "Can an incomplete fix pass the F2P tests?  Are there "
            "acceptance criteria items that no F2P test verifies?  Does "
            "the gold patch leave some stated requirements unaddressed?"
        ),
    },
}



def compute_task_severity(
    labels: list[TaskLabelAssignment],
) -> Severity:
    """Compute task severity from label presence (pure set membership).

    No arithmetic, no thresholds. The label co-occurrence rules encode a
    core insight from the human audit of 107 SEVERE cases:

      In an authentic PR, the gold patch and the F2P tests are authored
      together. If the tests assert on behaviour NOT described in the
      problem (OVER_TEST), then either:
        (a) the patch also reaches beyond scope (OVER_TEST + OVER_PATCH),
            which is unambiguous contamination, OR
        (b) the patch appears in-scope while the tests silently widen
            expectations — an equally dangerous form of contamination
            because the agent has no signal to know the tests demand
            behaviour outside the problem.

      Both (a) and (b) are SEVERE. OVER_TEST alone is RARE and demands
      maximum attention; it is NOT a softer form of (a).

    Rules:
      SEVERE  = APPROACH_LOCK  ∈ labels  OR
                OVER_TEST      ∈ labels
      MODERATE = OVER_PATCH ∈ labels AND (HIDDEN_CONTEXT ∈ labels OR UNCLEAR_DESCRIPTION ∈ labels)
      MINOR   = any single one of {OVER_PATCH, UNCLEAR_DESCRIPTION,
                                   HIDDEN_CONTEXT, WEAK_COVERAGE}
      CLEAN   = no contamination labels
    """
    if not labels:
        return Severity.CLEAN

    label_set = {
        la.label for la in labels
        if la.label != TaskContaminationLabel.CLEAN
    }

    if not label_set:
        return Severity.CLEAN

    if (TaskContaminationLabel.APPROACH_LOCK in label_set
            or TaskContaminationLabel.OVER_TEST in label_set):
        return Severity.SEVERE

    if (TaskContaminationLabel.OVER_PATCH in label_set
            and (TaskContaminationLabel.HIDDEN_CONTEXT in label_set
                 or TaskContaminationLabel.UNCLEAR_DESCRIPTION in label_set)):
        return Severity.MODERATE

    return Severity.MINOR



def _heuristic_labels(
    intent: IntentStatement,
    patch_analysis: PatchAnalysis,
    test_analysis: TestAnalysis,
    description_clarity: DescriptionClarity,
    record: TaskRecord | None = None,
    cross_ref: CrossReferenceResult | None = None,
) -> list[TaskLabelAssignment]:
    """Fast heuristic pre-classification from pipeline signals.

    Uses binary signals only — no ratio thresholds or counting.
    These serve as initial candidates for the LLM to refine.
    """
    candidates: list[TaskLabelAssignment] = []

    # ── OVER_TEST candidate ─────────────────────────────────────────
    # All evidence rolls up into a single TaskLabelAssignment regardless of
    # which combination of signals fired. Emitting one candidate (not 2-3)
    # keeps the heuristic deterministic and the downstream LLM prompt
    # readable.
    over_test_evidence: list[str] = []
    if test_analysis.off_topic_assertions > 0:
        over_test_evidence.append(
            f"{test_analysis.off_topic_assertions} OFF_TOPIC assertions found"
        )
    if test_analysis.unrelated_count > 0:
        over_test_evidence.append(
            f"{test_analysis.unrelated_count} UNRELATED tests found"
        )
    for tv in test_analysis.test_verdicts:
        if tv.is_modified and tv.off_topic_count > 0:
            over_test_evidence.append(
                f"Modified test '{tv.test_name}' has {tv.off_topic_count} "
                f"OFF_TOPIC assertions — pre-existing test with added excess "
                f"that may assert on gold-patch implementation values not "
                f"derivable from the problem statement"
            )
        if tv.is_modified and not tv.modification_aligned:
            over_test_evidence.append(
                f"Modified test '{tv.test_name}' has misaligned changes — "
                f"pre-existing test modified beyond problem scope"
            )

    if over_test_evidence:
        candidates.append(TaskLabelAssignment(
            label=TaskContaminationLabel.OVER_TEST,
            evidence=over_test_evidence,
        ))

    # OVER_PATCH: any UNRELATED hunk with behavioral changes
    # (pure ancillary does NOT trigger this)
    if patch_analysis.unrelated_count > 0:
        candidates.append(TaskLabelAssignment(
            label=TaskContaminationLabel.OVER_PATCH,
            evidence=[
                f"{patch_analysis.unrelated_count} UNRELATED hunks with "
                f"behavioral changes beyond problem scope",
            ],
        ))

    # Task/Patch Mismatch (AUDIT Pattern 1)
    if patch_analysis.required_count == 0 and patch_analysis.unrelated_count >= 2:
        candidates.append(TaskLabelAssignment(
            label=TaskContaminationLabel.APPROACH_LOCK,
            evidence=[
                f"Task/Patch Mismatch: 0 REQUIRED hunks, {patch_analysis.unrelated_count} UNRELATED hunks",
                "Gold patch implements entirely different functionality than problem describes",
            ],
        ))

    # Compilation barrier (AUDIT Pattern 3)
    monolithic_extensions = {".go", ".ts", ".tsx", ".rs"}
    if patch_analysis.unrelated_count > 0:
        unrelated_files = [hv.file_path for hv in patch_analysis.hunk_verdicts
                           if hv.verdict == PatchVerdict.UNRELATED]
        has_monolithic = any(
            any(f.endswith(ext) for ext in monolithic_extensions)
            for f in unrelated_files
        )
        if has_monolithic:
            candidates.append(TaskLabelAssignment(
                label=TaskContaminationLabel.APPROACH_LOCK,
                evidence=[
                    f"Potential compilation barrier: UNRELATED hunks in monolithic project files",
                    f"Files: {', '.join(unrelated_files[:5])}",
                    "In Go/TypeScript/Rust, unrelated changes may be required for compilation",
                ],
            ))

    # UNCLEAR_DESCRIPTION is intentionally NOT triggered by a float threshold
    # on ambiguity_score.  The classifier LLM decides from the problem text
    # directly.  The only deterministic path to this label is the
    # self-referential HIDDEN_CONTEXT check below (which is a different
    # failure mode anyway).

    # HIDDEN_CONTEXT: check for self-referential problem description
    if record and record.problem_statement:
        ps_lower = record.problem_statement.lower()
        self_ref_phrases = [
            "see the patch", "test case of the patch",
            "attached pr", "see the pr", "in the attached",
        ]
        for phrase in self_ref_phrases:
            if phrase in ps_lower:
                candidates.append(TaskLabelAssignment(
                    label=TaskContaminationLabel.HIDDEN_CONTEXT,
                    evidence=[f'Problem contains "{phrase}"'],
                ))
                break

    # APPROACH_LOCK: reporter suggested a specific fix approach
    if intent.decomposition and intent.decomposition.suggested_fix:
        candidates.append(TaskLabelAssignment(
            label=TaskContaminationLabel.APPROACH_LOCK,
            evidence=[
                f"Reporter suggests fix approach: "
                f"{intent.decomposition.suggested_fix}",
            ],
        ))

    # APPROACH_LOCK: overpatch-overtest coupling (tests require out-of-scope hunks)
    if cross_ref and cross_ref.has_coupling:
        for cd in cross_ref.couplings:
            candidates.append(TaskLabelAssignment(
                label=TaskContaminationLabel.APPROACH_LOCK,
                evidence=[
                    cd.reasoning,
                    f"Linked UNRELATED hunks: {cd.linked_hunk_indices}",
                ],
            ))
            break

    # Pre-staged test detection (AUDIT_PROTOCOL Gap 1)
    if record:
        has_test_patch = bool(getattr(record, 'test_patch', '') and record.test_patch.strip())
        has_before_cmd = "git checkout" in getattr(record, 'before_repo_set_cmd', '')
        if (has_test_patch or has_before_cmd) and not test_analysis.has_modified_tests:
            candidates.append(TaskLabelAssignment(
                label=TaskContaminationLabel.APPROACH_LOCK,
                evidence=[
                    "Tests pre-staged via before_repo_set_cmd (git checkout from gold commit)",
                    "Pipeline is_modified=False but test_patch has content — gap in modification detection",
                    "Pre-staged tests may assert on exact implementation values from gold commit",
                ],
            ))

    return candidates



TASK_CLASSIFIER_SYSTEM_PROMPT = _load_prompt("task_classifier")


def _build_task_classifier_user_prompt(
    intent: IntentStatement,
    patch_analysis: PatchAnalysis,
    test_analysis: TestAnalysis,
    description_clarity: DescriptionClarity,
    record: TaskRecord | None = None,
    heuristic_candidates: list[TaskLabelAssignment] | None = None,
    cross_ref: CrossReferenceResult | None = None,
) -> str:
    """Build the user prompt for the LLM task classifier."""
    parts: list[str] = []

    parts.append(f"INSTANCE: {intent.instance_id}")
    parts.append("")

    # Problem statement (full, un-truncated)
    if record and record.problem_statement:
        parts.append("PROBLEM STATEMENT:")
        parts.append(record.problem_statement)
        parts.append("")

    # Requirements (SWE-bench Pro, full)
    if record and record.requirements:
        parts.append("REQUIREMENTS:")
        parts.append(record.requirements)
        parts.append("")

    # Interface (SWE-bench Pro, full)
    if record and record.interface:
        parts.append("INTERFACE:")
        parts.append(record.interface)
        parts.append("")

    # Hints (full)
    if record and record.hints_text:
        parts.append("HINTS TEXT:")
        parts.append(record.hints_text)
        parts.append("")

    # Intent extraction
    parts.append("INTENT EXTRACTION:")
    parts.append(f"- Core requirement: {intent.core_requirement}")
    parts.append(f"- Behavioral contract: {intent.behavioral_contract}")
    parts.append(f"- Acceptance criteria: {json.dumps(intent.acceptance_criteria)}")
    parts.append(f"- Out of scope: {intent.out_of_scope}")
    parts.append(f"- Ambiguity score (raw LLM output, 0–1): {intent.ambiguity_score}")

    if intent.decomposition:
        d = intent.decomposition
        parts.append("")
        parts.append("PROBLEM DECOMPOSITION:")
        parts.append(f"- Bug description: {d.bug_description}")
        if d.suggested_fix:
            parts.append(f"- Reporter's suggested fix: {d.suggested_fix}")
            parts.append("  (Compare this to the gold patch — divergence signals APPROACH_LOCK)")
        parts.append(f"- Legitimacy: {d.legitimacy}")
        entities = []
        if d.mentioned_files:
            entities.append(f"Files: {', '.join(d.mentioned_files)}")
        if d.mentioned_functions:
            entities.append(f"Functions: {', '.join(d.mentioned_functions)}")
        if d.mentioned_classes:
            entities.append(f"Classes: {', '.join(d.mentioned_classes)}")
        if d.mentioned_variables:
            entities.append(f"Variables: {', '.join(d.mentioned_variables)}")
        if d.mentioned_modules:
            entities.append(f"Modules: {', '.join(d.mentioned_modules)}")
        if entities:
            parts.append(f"- Code entities: {'; '.join(entities)}")
    parts.append("")

    # Patch analysis
    parts.append("GOLD PATCH ANALYSIS:")
    parts.append(f"Hunks: {patch_analysis.total_hunks} "
                 f"(REQUIRED={patch_analysis.required_count}, "
                 f"ANCILLARY={patch_analysis.ancillary_count}, "
                 f"UNRELATED={patch_analysis.unrelated_count})")
    for hv in patch_analysis.hunk_verdicts:
        heuristic_tag = " [heuristic]" if hv.is_heuristic else ""
        parts.append(f"  Hunk {hv.hunk_index} [{hv.file_path}]: "
                     f"{hv.verdict.value} [{hv.evidence_strength}]{heuristic_tag} — "
                     f"{hv.reasoning}")
    parts.append("")

    # Test analysis
    parts.append("F2P TEST ANALYSIS:")
    parts.append(f"Tests: {test_analysis.total_tests} "
                 f"(ALIGNED={test_analysis.aligned_count}, "
                 f"TANGENTIAL={test_analysis.tangential_count}, "
                 f"UNRELATED={test_analysis.unrelated_count})")
    parts.append(f"Assertions: {test_analysis.total_assertions} "
                 f"(ON_TOPIC={test_analysis.on_topic_assertions}, "
                 f"OFF_TOPIC={test_analysis.off_topic_assertions})")
    parts.append(f"Has modified tests: {test_analysis.has_modified_tests}")
    for tv in test_analysis.test_verdicts:
        mod_tag = ""
        if tv.is_modified:
            mod_tag = " [MODIFIED pre-existing test"
            if not tv.modification_aligned:
                mod_tag += ", MISALIGNED changes"
            if tv.off_topic_count > 0:
                mod_tag += f", {tv.off_topic_count} OFF_TOPIC assertions"
            mod_tag += "]"
        parts.append(f"  Test '{tv.test_name}': {tv.intent_match.value} "
                     f"[{tv.evidence_strength}]{mod_tag}")
        if tv.reasoning:
            parts.append(f"    Reasoning: {tv.reasoning}")
        for av in tv.assertion_verdicts:
            reason_tag = f" — {av.reason}" if av.reason else ""
            parts.append(f"    [{av.verdict.value}] {av.statement}{reason_tag}")
    parts.append("")

    # Description clarity
    parts.append(f"DESCRIPTION CLARITY: score={description_clarity.score:.4f}")
    if description_clarity.reasoning:
        parts.append(f"  Reasoning: {description_clarity.reasoning}")
    parts.append("")

    if cross_ref and cross_ref.has_coupling:
        parts.append("CROSS-REFERENCE ANALYSIS (OVERPATCH-OVERTEST COUPLING):")
        parts.append(f"Overpatch-overtest couplings detected: {len(cross_ref.couplings)}")
        for cd in cross_ref.couplings:
            parts.append(f"  Test '{cd.test_name}' → UNRELATED hunks {cd.linked_hunk_indices} "
                         f"[{cd.evidence_strength}]")
            if cd.linked_files:
                parts.append(f"    Files: {', '.join(cd.linked_files)}")
            parts.append(f"    {cd.reasoning}")
        parts.append("  This is a strong APPROACH_LOCK signal: tests require code "
                     "the problem doesn't ask for.")
        parts.append("")

    # Heuristic candidates (guidance for LLM)
    if heuristic_candidates:
        parts.append("HEURISTIC PRE-CLASSIFICATION (refine or override):")
        for hc in heuristic_candidates:
            parts.append(f"  {hc.label.value}: "
                         f"{'; '.join(hc.evidence)}")
        parts.append("")

    return "\n".join(parts)


async def classify_task_labels(
    intent: IntentStatement,
    patch_analysis: PatchAnalysis,
    test_analysis: TestAnalysis,
    description_clarity: DescriptionClarity,
    record: TaskRecord | None = None,
    llm: Any | None = None,
    cross_ref: CrossReferenceResult | None = None,
) -> list[TaskLabelAssignment]:
    """Classify task contamination labels (Axis 1).

    If *llm* is provided, uses LLM for nuanced classification with
    heuristic candidates as guidance.  Otherwise falls back to pure
    heuristic classification.
    """
    heuristic = _heuristic_labels(intent, patch_analysis, test_analysis,
                                  description_clarity, record, cross_ref)

    if llm is None:
        # Pure heuristic fallback
        if not heuristic:
            return [TaskLabelAssignment(
                label=TaskContaminationLabel.CLEAN,
                evidence=["No heuristic signals detected"],
            )]
        return heuristic

    # Build LLM prompt
    user_prompt = _build_task_classifier_user_prompt(
        intent, patch_analysis, test_analysis, description_clarity, record, heuristic,
        cross_ref=cross_ref,
    )

    try:
        result: TaskClassificationResponse = await llm.query_structured(
            system_prompt=TASK_CLASSIFIER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=TaskClassificationResponse,
        )
        labels: list[TaskLabelAssignment] = []
        for item in result.labels:
            try:
                label_enum = TaskContaminationLabel(item.label)
            except ValueError:
                logger.warning("Unknown label from LLM: %s", item.label)
                continue
            labels.append(TaskLabelAssignment(
                label=label_enum,
                evidence=item.evidence,
                reasoning=item.reasoning,
            ))

        # Protect a curated set of high-precision deterministic heuristics
        # from being silently overruled by the LLM. The LLM is the *primary*
        # classifier (it can refine evidence, judge nuance, and combine
        # signals) but for a small subset of heuristics the *signal itself
        # is the ground truth* — the LLM has no extra information that
        # would let it override them. Pre-staged tests via
        # before_repo_set_cmd, 0-REQUIRED task/patch mismatch, and self-
        # referential problem text fall into this category: the heuristic
        # already cites unambiguous evidence the LLM doesn't see.
        #
        # The union policy: any protected heuristic that fired AND is
        # absent from the LLM output is appended with its original
        # evidence. The LLM's labels still drive everything else.
        protected_evidence_markers = (
            "Pre-staged via before_repo_set_cmd",
            "Tests pre-staged via before_repo_set_cmd",
            "Task/Patch Mismatch:",
            "Problem contains \"",
        )
        llm_label_set = {la.label for la in labels}
        for hc in heuristic:
            is_protected = any(
                any(marker in ev for marker in protected_evidence_markers)
                for ev in hc.evidence
            )
            if is_protected and hc.label not in llm_label_set:
                logger.info(
                    "Protected heuristic %s survived: LLM omitted it but "
                    "evidence is deterministic (%s)",
                    hc.label.value,
                    "; ".join(hc.evidence)[:160],
                )
                labels.append(hc)
                llm_label_set.add(hc.label)

        # Enforce co-occurrence rules
        has_contamination = any(
            la.label != TaskContaminationLabel.CLEAN
            for la in labels
        )
        if has_contamination:
            labels = [
                la for la in labels
                if la.label != TaskContaminationLabel.CLEAN
            ]

        if not labels:
            labels = [TaskLabelAssignment(
                label=TaskContaminationLabel.CLEAN,
                evidence=["LLM found no contamination signals"],
            )]
        return labels

    except Exception:
        logger.exception("LLM classification failed; falling back to heuristics")
        if not heuristic:
            return [TaskLabelAssignment(
                label=TaskContaminationLabel.CLEAN,
                evidence=["Heuristic fallback — no signals"],
            )]
        return heuristic

