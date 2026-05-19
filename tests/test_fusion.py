"""Tests for bench_cleanser.fusion."""

from __future__ import annotations

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


def _report(severity: Severity, labels: list[TaskContaminationLabel]) -> ContaminationReport:
    return ContaminationReport(
        instance_id="pkg/repo-42",
        severity=severity,
        intent=IntentStatement(
            instance_id="pkg/repo-42",
            core_requirement="r",
            behavioral_contract="c",
            acceptance_criteria=[],
            out_of_scope="",
            ambiguity_score=0.0,
        ),
        patch_analysis=PatchAnalysis(total_hunks=0, required_count=0,
                                     ancillary_count=0, unrelated_count=0),
        test_analysis=TestAnalysis(total_tests=0, aligned_count=0,
                                   tangential_count=0, unrelated_count=0,
                                   total_assertions=0, on_topic_assertions=0,
                                   off_topic_assertions=0,
                                   has_modified_tests=False),
        description_clarity=DescriptionClarity(score=0.0, reasoning=""),
        task_labels=[
            TaskLabelAssignment(label=label, evidence=["e"], reasoning="r")
            for label in labels
        ],
    )


def _traj(agent_label: AgentTrajectoryLabel, pattern: LeakagePattern) -> TrajectoryAnalysis:
    return TrajectoryAnalysis(
        instance_id="pkg/repo-42",
        agent_name="test-agent",
        leakage_pattern=pattern,
        trajectory_label=agent_label,
    )


def test_fair_pass_clean_genuine():
    r = _report(Severity.CLEAN, [TaskContaminationLabel.CLEAN])
    t = _traj(AgentTrajectoryLabel.AGENT_PASSED_GENUINE, LeakagePattern.GENUINE_SOLUTION)
    f = fuse(r, t)
    assert f.verdict == FusionVerdict.FAIR_PASS
    assert not f.invalidates_measurement


def test_agent_cheated_leak_on_clean_task():
    r = _report(Severity.CLEAN, [TaskContaminationLabel.CLEAN])
    t = _traj(AgentTrajectoryLabel.AGENT_PASSED_LEAK, LeakagePattern.GOLD_PATCH_LEAK)
    f = fuse(r, t)
    assert f.verdict == FusionVerdict.AGENT_CHEATED
    assert f.invalidates_measurement


def test_agent_cheated_test_aware_on_clean_task():
    r = _report(Severity.CLEAN, [TaskContaminationLabel.CLEAN])
    t = _traj(
        AgentTrajectoryLabel.AGENT_PASSED_TEST_AWARE,
        LeakagePattern.TEST_AWARE,
    )
    f = fuse(r, t)
    assert f.verdict == FusionVerdict.AGENT_CHEATED
    assert f.invalidates_measurement


def test_agent_cheated_package_leak_on_clean_task():
    r = _report(Severity.CLEAN, [TaskContaminationLabel.CLEAN])
    t = _traj(
        AgentTrajectoryLabel.AGENT_PASSED_PACKAGE_LEAK,
        LeakagePattern.PACKAGE_LEAK,
    )
    f = fuse(r, t)
    assert f.verdict == FusionVerdict.AGENT_CHEATED
    assert f.invalidates_measurement


def test_agent_cheated_trained_hack_on_clean_task():
    r = _report(Severity.CLEAN, [TaskContaminationLabel.CLEAN])
    t = _traj(
        AgentTrajectoryLabel.AGENT_PASSED_TRAINED_HACK,
        LeakagePattern.PARTIAL_MATCH,
    )
    f = fuse(r, t)
    assert f.verdict == FusionVerdict.AGENT_CHEATED
    assert f.invalidates_measurement


def test_contaminated_pass_severe_task_genuine_trajectory():
    r = _report(Severity.SEVERE, [TaskContaminationLabel.OVER_TEST])
    t = _traj(AgentTrajectoryLabel.AGENT_PASSED_GENUINE, LeakagePattern.GENUINE_SOLUTION)
    f = fuse(r, t)
    assert f.verdict == FusionVerdict.CONTAMINATED_PASS
    assert f.invalidates_measurement


def test_contaminated_pass_moderate_task_genuine_trajectory():
    r = _report(
        Severity.MODERATE,
        [TaskContaminationLabel.OVER_PATCH, TaskContaminationLabel.UNCLEAR_DESCRIPTION],
    )
    t = _traj(AgentTrajectoryLabel.AGENT_PASSED_GENUINE, LeakagePattern.GENUINE_SOLUTION)
    f = fuse(r, t)
    assert f.verdict == FusionVerdict.CONTAMINATED_PASS
    assert f.invalidates_measurement


def test_unfair_failure_approach_lock():
    r = _report(Severity.SEVERE, [TaskContaminationLabel.APPROACH_LOCK])
    t = _traj(AgentTrajectoryLabel.AGENT_FAILED_COMPLETED_INTENT, LeakagePattern.UNKNOWN)
    f = fuse(r, t)
    assert f.verdict == FusionVerdict.UNFAIR_FAILURE
    assert f.invalidates_measurement


def test_unfair_failure_over_test():
    r = _report(Severity.SEVERE, [TaskContaminationLabel.OVER_TEST])
    t = _traj(AgentTrajectoryLabel.AGENT_FAILED_COMPLETED_INTENT, LeakagePattern.UNKNOWN)
    f = fuse(r, t)
    assert f.verdict == FusionVerdict.UNFAIR_FAILURE


def test_fair_failure_clean_task():
    r = _report(Severity.CLEAN, [TaskContaminationLabel.CLEAN])
    t = _traj(AgentTrajectoryLabel.AGENT_FAILED_COMPLETED_INTENT, LeakagePattern.UNKNOWN)
    f = fuse(r, t)
    assert f.verdict == FusionVerdict.FAIR_FAILURE
    assert not f.invalidates_measurement


def test_agent_disengaged():
    r = _report(Severity.CLEAN, [TaskContaminationLabel.CLEAN])
    t = _traj(AgentTrajectoryLabel.AGENT_FAILED_NO_INTENT, LeakagePattern.UNKNOWN)
    f = fuse(r, t)
    assert f.verdict == FusionVerdict.AGENT_DISENGAGED


def test_minor_over_patch_doesnt_excuse_failure():
    """over_patch alone doesn't make a failure unfair — the test didn't reach it."""
    r = _report(Severity.MINOR, [TaskContaminationLabel.OVER_PATCH])
    t = _traj(AgentTrajectoryLabel.AGENT_FAILED_COMPLETED_INTENT, LeakagePattern.UNKNOWN)
    f = fuse(r, t)
    assert f.verdict == FusionVerdict.FAIR_FAILURE
