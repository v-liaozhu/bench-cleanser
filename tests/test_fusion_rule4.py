"""Regression tests for Fusion Rule 4 (AMBIGUOUS_PASS vs INCONCLUSIVE).

Rule 4 was previously misfiring: the discriminator between AMBIGUOUS_PASS
and INCONCLUSIVE was an analysis-identity check that was always true. The
v1.0.0 fix routes the decision through ``TrajectoryAnalysis.resolved`` —
the agent's reported pass/fail outcome on the F2P tests.

Matrix:

  ┌──────────────┬─────────────────┬──────────────────────────────────────┐
  │ traj.resolved│ task severity   │ expected verdict                     │
  ├──────────────┼─────────────────┼──────────────────────────────────────┤
  │ True         │ CLEAN           │ AMBIGUOUS_PASS                       │
  │ True         │ MINOR           │ INCONCLUSIVE  (can't rule out leak)  │
  │ True         │ MODERATE        │ INCONCLUSIVE                         │
  │ True         │ SEVERE          │ INCONCLUSIVE                         │
  │ False        │ CLEAN           │ INCONCLUSIVE                         │
  │ False        │ SEVERE          │ INCONCLUSIVE                         │
  └──────────────┴─────────────────┴──────────────────────────────────────┘

Why "MINOR" goes to INCONCLUSIVE rather than AMBIGUOUS_PASS: the rule
encodes a strict-purity check. If the task carries *any* label set above
CLEAN we cannot say the pass was legitimate even when the trajectory is
just unknown — manual review is required.
"""

from __future__ import annotations

import pytest

from bench_cleanser.fusion import FusionVerdict, fuse
from bench_cleanser.models import (
    AgentTrajectoryLabel,
    ContaminationReport,
    DescriptionClarity,
    IntentStatement,
    PatchAnalysis,
    Severity,
    TaskContaminationLabel,
    TaskLabelAssignment,
    TestAnalysis,
)
from bench_cleanser.trajectory.models import LeakagePattern, TrajectoryAnalysis


def _report(
    severity: Severity,
    labels: list[TaskContaminationLabel] | None = None,
) -> ContaminationReport:
    return ContaminationReport(
        instance_id="pkg/repo-rule4",
        severity=severity,
        intent=IntentStatement(
            instance_id="pkg/repo-rule4",
            core_requirement="r", behavioral_contract="c",
            acceptance_criteria=[], out_of_scope="",
            ambiguity_score=0.0,
        ),
        patch_analysis=PatchAnalysis(
            total_hunks=0, required_count=0,
            ancillary_count=0, unrelated_count=0,
        ),
        test_analysis=TestAnalysis(
            total_tests=0, aligned_count=0,
            tangential_count=0, unrelated_count=0,
            total_assertions=0, on_topic_assertions=0,
            off_topic_assertions=0, has_modified_tests=False,
        ),
        description_clarity=DescriptionClarity(score=0.0, reasoning=""),
        task_labels=[
            TaskLabelAssignment(label=label, evidence=["e"], reasoning="r")
            for label in (labels or [])
        ],
    )


def _unknown_traj(resolved: bool) -> TrajectoryAnalysis:
    return TrajectoryAnalysis(
        instance_id="pkg/repo-rule4",
        agent_name="test-agent",
        leakage_pattern=LeakagePattern.UNKNOWN,
        trajectory_label=AgentTrajectoryLabel.AGENT_UNKNOWN,
        resolved=resolved,
    )


@pytest.mark.parametrize("resolved, severity, expected", [
    (True,  Severity.CLEAN,    FusionVerdict.AMBIGUOUS_PASS),
    (True,  Severity.MINOR,    FusionVerdict.INCONCLUSIVE),
    (True,  Severity.MODERATE, FusionVerdict.INCONCLUSIVE),
    (True,  Severity.SEVERE,   FusionVerdict.INCONCLUSIVE),
    (False, Severity.CLEAN,    FusionVerdict.INCONCLUSIVE),
    (False, Severity.SEVERE,   FusionVerdict.INCONCLUSIVE),
])
def test_fusion_rule4_matrix(resolved: bool, severity: Severity,
                             expected: FusionVerdict):
    """The four-cell ``resolved × severity`` matrix on UNKNOWN trajectories."""
    labels = []
    if severity == Severity.SEVERE:
        labels = [TaskContaminationLabel.OVER_TEST]
    elif severity == Severity.MODERATE:
        labels = [TaskContaminationLabel.OVER_PATCH]
    elif severity == Severity.MINOR:
        labels = [TaskContaminationLabel.UNCLEAR_DESCRIPTION]

    fusion = fuse(_report(severity, labels), _unknown_traj(resolved))

    assert fusion.verdict == expected, (
        f"resolved={resolved} severity={severity.value} -> "
        f"expected {expected.value}, got {fusion.verdict.value}; "
        f"reasoning={fusion.reasoning}"
    )


def test_ambiguous_pass_does_not_invalidate_measurement():
    """AMBIGUOUS_PASS is a soft verdict — it does not flag the row as broken."""
    fusion = fuse(_report(Severity.CLEAN), _unknown_traj(resolved=True))
    assert fusion.verdict == FusionVerdict.AMBIGUOUS_PASS
    assert fusion.invalidates_measurement is False


def test_inconclusive_severe_does_not_invalidate_measurement():
    """INCONCLUSIVE on a SEVERE-with-UNKNOWN-trajectory still doesn't flip
    `invalidates_measurement` to True. The fusion engine reserves that flag
    for verdicts that *know* the row is bad (AGENT_CHEATED, CONTAMINATED_PASS,
    UNFAIR_FAILURE) — not for verdicts that simply lack evidence."""
    fusion = fuse(
        _report(Severity.SEVERE, [TaskContaminationLabel.OVER_TEST]),
        _unknown_traj(resolved=True),
    )
    assert fusion.verdict == FusionVerdict.INCONCLUSIVE
    assert fusion.invalidates_measurement is False
