"""Stage 7: Task-Trajectory Fusion.

Combines the two independent axes produced by the pipeline into a single
fairness verdict per (task, agent) pair:

  - Axis 1: task-level contamination (Severity + TaskContaminationLabel set)
            — produced by bench_cleanser.classification.scorer
  - Axis 2: per-agent trajectory classification (AgentTrajectoryLabel)
            — produced by bench_cleanser.trajectory.classifier

The fusion verdict answers the question a human reviewer actually cares about:
"Given what the benchmark asked and what the agent did, was this a fair
measurement, a contaminated pass, a false failure, or something we should
discard?"

This stage is deliberately rule-based and deterministic — no LLM calls.
Every verdict is reproducible from the two axis labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from bench_cleanser.models import (
    AgentTrajectoryLabel,
    ContaminationReport,
    Severity,
    TaskContaminationLabel,
)
from bench_cleanser.trajectory.models import TrajectoryAnalysis


class FusionVerdict(str, Enum):
    """Combined fairness verdict per (task, agent)."""

    # Agent passed, task is clean => genuine measurement of capability.
    FAIR_PASS = "fair_pass"

    # Agent passed via a leak (gold patch, test-aware, package, trained hack).
    # The task may be clean; the measurement is still invalid for this agent.
    AGENT_CHEATED = "agent_cheated"

    # Agent passed a contaminated task.  Even if the trajectory looks
    # genuine, the pass is over a broken measurement.
    CONTAMINATED_PASS = "contaminated_pass"

    # Agent passed a clean task but via ambiguous means (UNKNOWN trajectory).
    AMBIGUOUS_PASS = "ambiguous_pass"

    # Agent failed; task is contaminated in a way that likely caused the
    # failure (approach_lock, over_test).  This is an UNFAIR failure.
    UNFAIR_FAILURE = "unfair_failure"

    # Agent failed; task is clean or only minor.  Genuine capability gap.
    FAIR_FAILURE = "fair_failure"

    # Agent failed without actually engaging the problem (no intent).
    # Not a benchmark quality signal either way.
    AGENT_DISENGAGED = "agent_disengaged"

    # Nothing is clear.
    INCONCLUSIVE = "inconclusive"


@dataclass
class TaskTrajectoryFusion:
    """Fusion result for a single (task, agent) pair."""

    instance_id: str
    agent_name: str
    task_severity: Severity
    task_labels: list[TaskContaminationLabel]
    trajectory_label: AgentTrajectoryLabel
    verdict: FusionVerdict
    reasoning: str
    invalidates_measurement: bool
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "agent_name": self.agent_name,
            "task_severity": self.task_severity.value,
            "task_labels": [l.value for l in self.task_labels],
            "trajectory_label": self.trajectory_label.value,
            "verdict": self.verdict.value,
            "reasoning": self.reasoning,
            "invalidates_measurement": self.invalidates_measurement,
            "evidence": list(self.evidence),
        }


# Groups of agent labels by outcome.
_AGENT_PASS_LEAK = {
    AgentTrajectoryLabel.AGENT_PASSED_LEAK,
    AgentTrajectoryLabel.AGENT_PASSED_TEST_AWARE,
    AgentTrajectoryLabel.AGENT_PASSED_PACKAGE_LEAK,
    AgentTrajectoryLabel.AGENT_PASSED_TRAINED_HACK,
}
_AGENT_PASS_GENUINE = {AgentTrajectoryLabel.AGENT_PASSED_GENUINE}
_AGENT_FAIL_TRIED = {AgentTrajectoryLabel.AGENT_FAILED_COMPLETED_INTENT}
_AGENT_FAIL_DISENGAGED = {AgentTrajectoryLabel.AGENT_FAILED_NO_INTENT}
_AGENT_UNKNOWN = {AgentTrajectoryLabel.AGENT_UNKNOWN}

# Contamination labels that can directly cause an UNFAIR_FAILURE.
_UNFAIR_FAILURE_DRIVERS = {
    TaskContaminationLabel.APPROACH_LOCK,
    TaskContaminationLabel.OVER_TEST,
}


def fuse(
    report: ContaminationReport,
    trajectory: TrajectoryAnalysis,
) -> TaskTrajectoryFusion:
    """Fuse one task report with one agent trajectory analysis.

    Rules (evaluated top-down; first match wins):

    1. Agent passed with leak   -> AGENT_CHEATED          (invalidates)
    2. Agent passed, task SEVERE/MODERATE -> CONTAMINATED_PASS (invalidates)
    3. Agent passed genuine, task clean/minor -> FAIR_PASS
    4. Agent passed unknown     -> AMBIGUOUS_PASS
    5. Agent failed completed intent, task has APPROACH_LOCK or OVER_TEST
                                -> UNFAIR_FAILURE         (invalidates)
    6. Agent failed completed intent, task clean/minor -> FAIR_FAILURE
    7. Agent failed no intent   -> AGENT_DISENGAGED
    8. Otherwise                -> INCONCLUSIVE
    """
    labels = [a.label for a in report.task_labels]
    label_set = set(labels)
    traj = trajectory.agent_trajectory_label
    sev = report.severity
    evidence: list[str] = []

    # Rule 1 — trajectory leak dominates everything else.
    if traj in _AGENT_PASS_LEAK:
        evidence.append(f"Agent trajectory classified as {traj.value}")
        if trajectory.evidence:
            evidence.extend(trajectory.evidence[:3])
        return _make(
            report, trajectory,
            verdict=FusionVerdict.AGENT_CHEATED,
            reasoning=(
                "Agent passed but the trajectory shows a leakage signal. "
                "The measurement does not reflect genuine capability."
            ),
            invalidates=True, evidence=evidence,
        )

    # Rule 2 — pass over contaminated task.
    if traj in _AGENT_PASS_GENUINE and sev in (Severity.SEVERE, Severity.MODERATE):
        evidence.append(f"Task severity {sev.value} with labels: "
                        f"{[l.value for l in labels]}")
        return _make(
            report, trajectory,
            verdict=FusionVerdict.CONTAMINATED_PASS,
            reasoning=(
                "Agent passed a contaminated task. Even a genuine-looking "
                "trajectory cannot produce a valid measurement when the "
                "underlying benchmark item is broken."
            ),
            invalidates=True, evidence=evidence,
        )

    # Rule 3 — the good case.
    if traj in _AGENT_PASS_GENUINE and sev in (Severity.CLEAN, Severity.MINOR):
        evidence.append("Clean/minor task; genuine trajectory")
        return _make(
            report, trajectory,
            verdict=FusionVerdict.FAIR_PASS,
            reasoning=(
                "Agent passed a clean (or only minor) task via a genuine "
                "trajectory.  Valid evidence of capability."
            ),
            invalidates=False, evidence=evidence,
        )

    # Rule 4 — passed but we cannot classify the trajectory.
    # Driven by the trajectory's reported outcome (`resolved`), not by analysis
    # identity. A passed-but-UNKNOWN trajectory on a clean task is genuinely
    # AMBIGUOUS_PASS; on a contaminated task it is INCONCLUSIVE because the
    # contamination labels would otherwise have produced CONTAMINATED_PASS
    # had we been able to confirm the pass was clean-trajectory.
    if traj in _AGENT_UNKNOWN:
        evidence.append("Trajectory classification inconclusive")
        if trajectory.resolved:
            evidence.append("Agent reported PASS on this task")
            if sev == Severity.CLEAN:
                return _make(
                    report, trajectory,
                    verdict=FusionVerdict.AMBIGUOUS_PASS,
                    reasoning=(
                        "Agent passed a clean task but the trajectory could "
                        "not be characterised. Manual review recommended."
                    ),
                    invalidates=False, evidence=evidence,
                )
            return _make(
                report, trajectory,
                verdict=FusionVerdict.INCONCLUSIVE,
                reasoning=(
                    "Agent passed but the trajectory could not be characterised "
                    f"and the task carries severity {sev.value}. Cannot decide "
                    "between FAIR_PASS, AGENT_CHEATED, and CONTAMINATED_PASS."
                ),
                invalidates=False, evidence=evidence,
            )
        # Did not resolve and trajectory is unknown — no usable signal.
        evidence.append("Agent reported FAIL on this task")
        return _make(
            report, trajectory,
            verdict=FusionVerdict.INCONCLUSIVE,
            reasoning=(
                "Agent failed and the trajectory could not be characterised. "
                "Manual review recommended."
            ),
            invalidates=False, evidence=evidence,
        )

    # Rule 5 — failed but task is unfair.
    if traj in _AGENT_FAIL_TRIED and (label_set & _UNFAIR_FAILURE_DRIVERS):
        drivers = sorted(l.value for l in label_set & _UNFAIR_FAILURE_DRIVERS)
        evidence.append(f"Agent attempted the problem; unfair labels: {drivers}")
        return _make(
            report, trajectory,
            verdict=FusionVerdict.UNFAIR_FAILURE,
            reasoning=(
                "Agent completed the intent described in the problem but "
                "failed the F2P tests.  The task carries contamination "
                f"labels ({', '.join(drivers)}) that reject valid "
                "alternative solutions or demand out-of-scope behaviour, "
                "so the failure does not reflect an actual capability gap."
            ),
            invalidates=True, evidence=evidence,
        )

    # Rule 6 — failed, task reasonable.
    if traj in _AGENT_FAIL_TRIED:
        evidence.append(f"Task severity {sev.value}; agent completed intent but failed")
        return _make(
            report, trajectory,
            verdict=FusionVerdict.FAIR_FAILURE,
            reasoning=(
                "Agent engaged the problem but failed.  The task does not "
                "carry the contamination signals that would excuse the "
                "failure, so this is a genuine capability gap."
            ),
            invalidates=False, evidence=evidence,
        )

    # Rule 7 — disengaged.
    if traj in _AGENT_FAIL_DISENGAGED:
        evidence.append("Agent never produced an intent to solve the described problem")
        return _make(
            report, trajectory,
            verdict=FusionVerdict.AGENT_DISENGAGED,
            reasoning=(
                "Agent never produced an intent matching the problem "
                "description.  This is an agent behaviour failure, not a "
                "benchmark quality signal."
            ),
            invalidates=False, evidence=evidence,
        )

    # Rule 8 — everything else.
    return _make(
        report, trajectory,
        verdict=FusionVerdict.INCONCLUSIVE,
        reasoning="Neither axis provides a decisive signal.",
        invalidates=False, evidence=evidence,
    )


def fuse_all(
    report: ContaminationReport,
    trajectories: list[TrajectoryAnalysis],
) -> list[TaskTrajectoryFusion]:
    return [fuse(report, t) for t in trajectories]


def _make(
    report: ContaminationReport,
    trajectory: TrajectoryAnalysis,
    *,
    verdict: FusionVerdict,
    reasoning: str,
    invalidates: bool,
    evidence: list[str],
) -> TaskTrajectoryFusion:
    return TaskTrajectoryFusion(
        instance_id=report.instance_id,
        agent_name=trajectory.agent_name,
        task_severity=report.severity,
        task_labels=[a.label for a in report.task_labels],
        trajectory_label=trajectory.agent_trajectory_label,
        verdict=verdict,
        reasoning=reasoning,
        invalidates_measurement=invalidates,
        evidence=evidence,
    )
