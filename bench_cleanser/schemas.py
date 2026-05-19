"""Pydantic models that define structured LLM output schemas.

Every LLM call in the pipeline returns one of these models via
OpenAI's ``response_format={"type": "json_schema", ...}`` parameter.
This enforces schema compliance at the API level — no regex extraction,
no fallback JSON parsing, no silent field omissions.

All schemas use ``strict=True`` so the API rejects non-conforming output.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ── Stage 2: Intent Extraction ──────────────────────────────────────


class IntentExtractionResponse(BaseModel):
    """Structured output for Stage 2: intent extraction from problem statement."""

    core_requirement: str = Field(
        ...,
        description=(
            "One-sentence description of the primary bug or feature being "
            "reported. Be precise — do not inflate scope."
        ),
    )
    behavioral_contract: str = Field(
        ...,
        description=(
            "Concrete BEFORE vs AFTER behavioral change. What observable "
            "behavior should change after the fix?"
        ),
    )
    acceptance_criteria: list[str] = Field(
        ...,
        description=(
            "Each specific, testable behavior that the problem description "
            "explicitly asks for. Only include behaviors DIRECTLY STATED or "
            "CLEARLY IMPLIED. Do NOT extrapolate."
        ),
    )
    out_of_scope: str = Field(
        ...,
        description=(
            "Behaviors, features, or refactors NOT asked for by the problem "
            "statement. Be explicit."
        ),
    )
    ambiguity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "How clear is the specification? 0.0 = perfectly clear, "
            "1.0 = extremely vague."
        ),
    )
    bug_description: str = Field(
        ...,
        description="What is actually broken — the observable defect or missing capability.",
    )
    suggested_fix: str = Field(
        ...,
        description=(
            "Reporter's suggested approach to fix. Empty string if none."
        ),
    )
    legitimacy: Literal[
        "bug",
        "feature_request",
        "enhancement",
        "question",
        "discussion",
        "unclear",
    ] = Field(
        ...,
        description=(
            "Classification of the problem type. One of: bug, feature_request, "
            "enhancement, question, discussion, unclear."
        ),
    )
    mentioned_files: list[str] = Field(
        default_factory=list,
        description="File paths mentioned literally in the problem statement.",
    )
    mentioned_functions: list[str] = Field(
        default_factory=list,
        description="Function/method names mentioned literally in the problem statement.",
    )
    mentioned_classes: list[str] = Field(
        default_factory=list,
        description="Class names mentioned literally in the problem statement.",
    )
    mentioned_variables: list[str] = Field(
        default_factory=list,
        description="Variable/attribute/setting names mentioned literally.",
    )
    mentioned_modules: list[str] = Field(
        default_factory=list,
        description="Module/package names mentioned literally.",
    )


# ── Stage 4A: Patch Hunk Verdict ────────────────────────────────────


class PatchHunkVerdictResponse(BaseModel):
    """Structured output for a single patch hunk intent-match verdict."""

    verdict: Literal["REQUIRED", "ANCILLARY", "UNRELATED"] = Field(
        ...,
        description=(
            "REQUIRED: directly implements acceptance criteria. "
            "ANCILLARY: supports the fix but not described in problem. "
            "UNRELATED: modifies behavior not described and not required."
        ),
    )
    evidence_strength: Literal["strong", "moderate", "weak"] = Field(
        ...,
        description="How strong is the evidence for this verdict: strong, moderate, or weak.",
    )
    reasoning: str = Field(
        ...,
        description="Explanation citing specific acceptance criteria or problem statement text.",
    )


class BatchPatchVerdictItem(BaseModel):
    """Verdict for one hunk in a batched patch analysis call."""

    hunk_index: int = Field(..., description="Zero-based index of the hunk in the list provided.")
    file_path: str = Field(..., description="File path of this hunk.")
    verdict: Literal["REQUIRED", "ANCILLARY", "UNRELATED"] = Field(...)
    evidence_strength: Literal["strong", "moderate", "weak"] = Field(...)
    reasoning: str = Field(
        ...,
        description="Detailed explanation citing specific acceptance criteria.",
    )


class BatchPatchVerdictsResponse(BaseModel):
    """Structured output for batched patch analysis — all hunks in one call."""

    verdicts: list[BatchPatchVerdictItem] = Field(
        ...,
        description="One verdict per patch hunk, in the same order as the input hunks.",
    )


# ── Stage 4B: Test Verdict ──────────────────────────────────────────


class AssertionVerdictItem(BaseModel):
    """Verdict for a single assertion within a test."""

    index: int = Field(..., description="Zero-based index of the assertion in the list provided.")
    verdict: Literal["ON_TOPIC", "OFF_TOPIC"] = Field(
        ...,
        description=(
            "ON_TOPIC: verifies behavior in acceptance criteria. "
            "OFF_TOPIC: checks behavior NOT described in the problem."
        ),
    )
    reason: str = Field(..., description="Brief explanation.")


class TestVerdictResponse(BaseModel):
    """Structured output for a single test function intent-match verdict."""

    test_verdict: Literal["ALIGNED", "TANGENTIAL", "UNRELATED"] = Field(
        ...,
        description=(
            "ALIGNED: test targets the described problem. "
            "TANGENTIAL: partially targets but includes significant excess. "
            "UNRELATED: does not target the described problem."
        ),
    )
    evidence_strength: Literal["strong", "moderate", "weak"] = Field(...)
    reasoning: str = Field(
        ..., description="Concise explanation citing acceptance criteria."
    )
    is_modification_aligned: bool = Field(
        ...,
        description=(
            "For MODIFIED tests: are the modifications aligned with "
            "acceptance criteria? True if test is NEW."
        ),
    )
    assertion_verdicts: list[AssertionVerdictItem] = Field(
        default_factory=list,
        description="Per-assertion verdicts, indexed to match input assertion list.",
    )


class BatchTestVerdictItem(BaseModel):
    """Verdict for one test in a batched test analysis call."""

    test_index: int = Field(..., description="Zero-based index of the test in the input list.")
    test_id: str = Field(..., description="Full test identifier.")
    test_verdict: Literal["ALIGNED", "TANGENTIAL", "UNRELATED"] = Field(...)
    evidence_strength: Literal["strong", "moderate", "weak"] = Field(...)
    reasoning: str = Field(...)
    is_modification_aligned: bool = Field(...)
    assertion_verdicts: list[AssertionVerdictItem] = Field(default_factory=list)


class BatchTestVerdictsResponse(BaseModel):
    """Structured output for batched test analysis — all tests in one call."""

    verdicts: list[BatchTestVerdictItem] = Field(
        ...,
        description="One verdict per F2P test, in the same order as the input tests.",
    )


# ── Stage 6: Task Classification ────────────────────────────────────


class TaskLabelItem(BaseModel):
    """A single contamination label assignment."""

    label: Literal[
        "approach_lock",
        "over_test",
        "over_patch",
        "unclear_description",
        "hidden_context",
        "weak_coverage",
        "clean",
    ] = Field(..., description="Taxonomy label from the contamination classification.")
    evidence: list[str] = Field(
        ...,
        description="Specific evidence items supporting this label (cite hunks, assertions, text).",
    )
    reasoning: str = Field(
        ...,
        description="Detailed explanation of why this label applies.",
    )


class TaskClassificationResponse(BaseModel):
    """Structured output for Stage 6: task-level contamination classification."""

    labels: list[TaskLabelItem] = Field(
        ...,
        description=(
            "All contamination labels that apply. If none apply, "
            "return a single 'clean' label. When any contamination label "
            "is present, do NOT include 'clean'."
        ),
    )


# ── Stage 7 (per-agent): Trajectory Classification ──────────────────


class TrajectoryClassificationResponse(BaseModel):
    """Structured output for per-agent trajectory classification."""

    pattern: Literal[
        "GENUINE_SOLUTION",
        "GOLD_PATCH_LEAK",
        "PACKAGE_LEAK",
        "TEST_AWARE",
        "PARTIAL_MATCH",
        "UNKNOWN",
    ] = Field(
        ...,
        description=(
            "Leakage pattern that best characterises how the agent arrived at "
            "its final patch. UNKNOWN is reserved for cases with insufficient "
            "trajectory data."
        ),
    )
    trajectory_label: Literal[
        "agent_passed_genuine",
        "agent_passed_leak",
        "agent_passed_package_leak",
        "agent_passed_test_aware",
        "agent_passed_trained_hack",
        "agent_failed_completed_intent",
        "agent_failed_no_intent",
        "agent_unknown",
    ] = Field(
        ...,
        description=(
            "Axis-2 fairness label. Must be consistent with `pattern` and with "
            "the agent's reported pass/fail outcome — passed-* labels only "
            "when the agent resolved the task; failed-* labels only when it "
            "did not."
        ),
    )
    evidence_strength: Literal["strong", "moderate", "weak"] = Field(
        ...,
        description="How well the available evidence supports the chosen labels.",
    )
    reasoning: str = Field(
        ...,
        description=(
            "Detailed explanation of the classification, citing concrete "
            "actions, files, or patch content from the trajectory."
        ),
    )
    causal_chain: str = Field(
        default="",
        description=(
            "Short description of what led the agent to its approach "
            "(e.g. infrastructure failure forced a workaround, agent "
            "anchored on stack-trace search). Empty string if not applicable."
        ),
    )
    key_evidence: list[str] = Field(
        default_factory=list,
        description="Bulleted concrete evidence items (steps, quotes, file refs).",
    )
    agent_behavior_summary: str = Field(
        default="",
        description="One- or two-sentence characterisation of the agent's behaviour.",
    )

