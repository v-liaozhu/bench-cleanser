"""Unit tests for :mod:`bench_cleanser.trajectory.classifier`.

Exercises every public function and verifies the LLM path against an
in-test ``FakeLLM`` so the suite stays offline.
"""

from __future__ import annotations

from typing import Any

import pytest

from bench_cleanser.models import AgentTrajectoryLabel
from bench_cleanser.schemas import TrajectoryClassificationResponse
from bench_cleanser.trajectory.classifier import (
    GOLD_PATCH_SIMILARITY_THRESHOLD,
    HIGH_SIMILARITY_THRESHOLD,
    classify_cross_agent,
    classify_heuristic_only,
    classify_with_llm,
    compute_patch_similarity,
    detect_pip_installs,
    detect_test_references,
    extract_heuristic_signals,
)
from bench_cleanser.trajectory.models import (
    ActionType,
    LeakagePattern,
    TrajectoryAction,
    TrajectoryRecord,
)

# Small but realistic patches that share most lines.
_GOLD_PATCH = """\
diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,15 @@
-def add(a, b):
-    return a + b
+def add(a, b):
+    if a is None or b is None:
+        raise ValueError("None operand")
+    if not isinstance(a, (int, float)):
+        raise TypeError("non-numeric a")
+    if not isinstance(b, (int, float)):
+        raise TypeError("non-numeric b")
+    if a > 1e300 or b > 1e300:
+        raise OverflowError("operand too large")
+    if a < -1e300 or b < -1e300:
+        raise OverflowError("operand too small")
+    return a + b
"""

_IDENTICAL_PATCH = _GOLD_PATCH
_TOTALLY_DIFFERENT_PATCH = """\
diff --git a/bar.py b/bar.py
--- a/bar.py
+++ b/bar.py
@@ -10,3 +10,3 @@
-x = compute_legacy()
+x = compute_modernised(rounding="banker")
"""


def _trajectory(
    *,
    instance_id: str = "pkg/repo-1",
    agent_name: str = "agent-A",
    actions: list[TrajectoryAction] | None = None,
    final_patch: str = "",
    resolved: bool = True,
) -> TrajectoryRecord:
    return TrajectoryRecord(
        instance_id=instance_id,
        agent_name=agent_name,
        actions=actions or [],
        final_patch=final_patch,
        resolved=resolved,
    )


# ── compute_patch_similarity ────────────────────────────────────────


class TestComputePatchSimilarity:
    def test_identical_patches(self):
        assert compute_patch_similarity(_GOLD_PATCH, _IDENTICAL_PATCH) == 1.0

    def test_empty_inputs_return_zero(self):
        assert compute_patch_similarity("", _GOLD_PATCH) == 0.0
        assert compute_patch_similarity(_GOLD_PATCH, "") == 0.0
        assert compute_patch_similarity("", "") == 0.0

    def test_normalizes_whitespace(self):
        spaced = _GOLD_PATCH.replace("    return", "        return")
        assert compute_patch_similarity(_GOLD_PATCH, spaced) == 1.0

    def test_strips_comments(self):
        with_comment = _GOLD_PATCH.replace(
            "+    return a + b",
            "+    return a + b  # sum",
        )
        assert compute_patch_similarity(_GOLD_PATCH, with_comment) >= 0.99

    def test_unrelated_patches_score_low(self):
        sim = compute_patch_similarity(_GOLD_PATCH, _TOTALLY_DIFFERENT_PATCH)
        assert sim < HIGH_SIMILARITY_THRESHOLD


# ── detect_pip_installs ─────────────────────────────────────────────


class TestDetectPipInstalls:
    def test_finds_simple_install(self):
        actions = [TrajectoryAction(
            action_type=ActionType.TERMINAL,
            content="pip install requests",
        )]
        installs = detect_pip_installs(_trajectory(actions=actions))
        assert any("requests" in cmd for cmd in installs)

    def test_finds_upgrade_install(self):
        actions = [TrajectoryAction(
            action_type=ActionType.TERMINAL,
            content="pip install --upgrade django",
        )]
        installs = detect_pip_installs(_trajectory(actions=actions))
        assert any("django" in cmd for cmd in installs)

    def test_ignores_non_terminal_actions(self):
        actions = [TrajectoryAction(
            action_type=ActionType.EDIT,
            content="pip install pwned",
        )]
        assert detect_pip_installs(_trajectory(actions=actions)) == []

    def test_empty_trajectory(self):
        assert detect_pip_installs(_trajectory(actions=[])) == []


# ── detect_test_references ──────────────────────────────────────────


class TestDetectTestReferences:
    def test_finds_short_test_name(self):
        actions = [TrajectoryAction(
            action_type=ActionType.THINK,
            content="I should make test_add_handles_none pass.",
        )]
        refs = detect_test_references(
            _trajectory(actions=actions),
            ["tests/test_foo.py::test_add_handles_none"],
        )
        assert len(refs) == 1
        assert "test_add_handles_none" in refs[0]

    def test_no_match(self):
        actions = [TrajectoryAction(
            action_type=ActionType.THINK,
            content="Looking at the code.",
        )]
        refs = detect_test_references(
            _trajectory(actions=actions),
            ["tests/test_foo.py::test_completely_unrelated"],
        )
        assert refs == []


# ── classify_heuristic_only ─────────────────────────────────────────


class TestClassifyHeuristicOnly:
    def test_gold_patch_match_returns_gold_leak(self):
        traj = _trajectory(final_patch=_GOLD_PATCH)
        analysis = classify_heuristic_only(traj, _GOLD_PATCH, [])
        assert analysis.leakage_pattern == LeakagePattern.GOLD_PATCH_LEAK
        assert analysis.evidence_strength == "strong"
        assert analysis.gold_patch_similarity >= GOLD_PATCH_SIMILARITY_THRESHOLD

    def test_pip_install_returns_package_leak(self):
        traj = _trajectory(
            actions=[TrajectoryAction(
                action_type=ActionType.TERMINAL,
                content="pip install librarywith-fix",
            )],
            final_patch=_TOTALLY_DIFFERENT_PATCH,
        )
        analysis = classify_heuristic_only(traj, _GOLD_PATCH, [])
        assert analysis.leakage_pattern == LeakagePattern.PACKAGE_LEAK

    def test_test_reference_returns_test_aware(self):
        traj = _trajectory(
            actions=[TrajectoryAction(
                action_type=ActionType.THINK,
                content="The test_my_f2p test must pass.",
            )],
            final_patch=_TOTALLY_DIFFERENT_PATCH,
        )
        analysis = classify_heuristic_only(
            traj, _GOLD_PATCH, ["test_my_f2p"],
        )
        assert analysis.leakage_pattern == LeakagePattern.TEST_AWARE

    def test_no_signals_returns_genuine_solution(self):
        traj = _trajectory(final_patch=_TOTALLY_DIFFERENT_PATCH)
        analysis = classify_heuristic_only(traj, _GOLD_PATCH, [])
        assert analysis.leakage_pattern == LeakagePattern.GENUINE_SOLUTION

    def test_resolved_propagates(self):
        traj = _trajectory(final_patch=_TOTALLY_DIFFERENT_PATCH, resolved=True)
        assert classify_heuristic_only(traj, _GOLD_PATCH, []).resolved is True
        traj2 = _trajectory(final_patch=_TOTALLY_DIFFERENT_PATCH, resolved=False)
        assert classify_heuristic_only(traj2, _GOLD_PATCH, []).resolved is False


# ── extract_heuristic_signals ───────────────────────────────────────


def test_extract_heuristic_signals_shape():
    traj = _trajectory(final_patch=_GOLD_PATCH)
    sig = extract_heuristic_signals(traj, _GOLD_PATCH, ["test_a"])
    assert sig["gold_patch_similarity"] == 1.0
    assert sig["has_gold_patch_match"] is True
    assert sig["has_high_similarity"] is True
    assert sig["pip_install_commands"] == []
    assert sig["test_references"] == []


# ── classify_cross_agent ────────────────────────────────────────────


def test_cross_agent_upgrades_genuine_to_leak_when_all_similar():
    """Cross-agent: if all final patches match, every GENUINE analysis is
    upgraded to GOLD_PATCH_LEAK with strong evidence."""
    from bench_cleanser.trajectory.models import TrajectoryAnalysis

    analyses = [
        TrajectoryAnalysis(
            instance_id="pkg/repo-1", agent_name="a1",
            leakage_pattern=LeakagePattern.GENUINE_SOLUTION,
        ),
        TrajectoryAnalysis(
            instance_id="pkg/repo-1", agent_name="a2",
            leakage_pattern=LeakagePattern.PARTIAL_MATCH,
        ),
    ]
    trajectories = [
        _trajectory(agent_name="a1", final_patch=_GOLD_PATCH),
        _trajectory(agent_name="a2", final_patch=_GOLD_PATCH),
    ]
    updated = classify_cross_agent(analyses, trajectories)
    assert all(a.leakage_pattern == LeakagePattern.GOLD_PATCH_LEAK for a in updated)
    assert all(a.evidence_strength == "strong" for a in updated)


def test_cross_agent_no_change_when_divergent():
    """If patches diverge, GENUINE analyses stay GENUINE."""
    from bench_cleanser.trajectory.models import TrajectoryAnalysis

    analyses = [
        TrajectoryAnalysis(
            instance_id="pkg/repo-1", agent_name="a1",
            leakage_pattern=LeakagePattern.GENUINE_SOLUTION,
        ),
        TrajectoryAnalysis(
            instance_id="pkg/repo-1", agent_name="a2",
            leakage_pattern=LeakagePattern.GENUINE_SOLUTION,
        ),
    ]
    trajectories = [
        _trajectory(agent_name="a1", final_patch=_GOLD_PATCH),
        _trajectory(agent_name="a2", final_patch=_TOTALLY_DIFFERENT_PATCH),
    ]
    updated = classify_cross_agent(analyses, trajectories)
    assert all(a.leakage_pattern == LeakagePattern.GENUINE_SOLUTION for a in updated)


def test_cross_agent_singleton_returns_unchanged():
    """Less than two analyses → no cross-agent inference is possible."""
    from bench_cleanser.trajectory.models import TrajectoryAnalysis

    analyses = [TrajectoryAnalysis(
        instance_id="pkg/repo-1", agent_name="a1",
        leakage_pattern=LeakagePattern.GENUINE_SOLUTION,
    )]
    trajectories = [_trajectory(agent_name="a1", final_patch=_GOLD_PATCH)]
    assert classify_cross_agent(analyses, trajectories) == analyses


# ── classify_with_llm (against fake structured client) ──────────────


class _FakeLLM:
    """Minimal LLMClient stub for trajectory tests."""

    def __init__(self, response: TrajectoryClassificationResponse):
        self.response = response
        self.calls: list[tuple[str, str, type]] = []

    async def query_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type,
        *,
        skip_cache: bool = False,
    ) -> Any:
        self.calls.append((system_prompt, user_prompt, response_model))
        return self.response


@pytest.mark.asyncio
async def test_classify_with_llm_happy_path():
    """LLM returns a passed_leak verdict → analysis carries that label and
    the heuristic signals stay attached."""
    response = TrajectoryClassificationResponse(
        pattern="GOLD_PATCH_LEAK",
        trajectory_label="agent_passed_leak",
        evidence_strength="strong",
        reasoning="Final patch matches gold patch closely.",
        causal_chain="Agent searched for file then jumped to fix.",
        key_evidence=["jump-to-file step 3", "no debug steps"],
        agent_behavior_summary="Direct application of pre-known fix.",
    )
    llm = _FakeLLM(response)

    traj = _trajectory(final_patch=_GOLD_PATCH, resolved=True)
    analysis = await classify_with_llm(
        trajectory=traj,
        gold_patch=_GOLD_PATCH,
        problem_statement="Handle None.",
        f2p_test_names=["test_none"],
        llm=llm,
    )

    assert analysis.leakage_pattern == LeakagePattern.GOLD_PATCH_LEAK
    assert analysis.trajectory_label == AgentTrajectoryLabel.AGENT_PASSED_LEAK
    assert analysis.evidence_strength == "strong"
    assert analysis.llm_reasoning.startswith("Final patch matches")
    assert analysis.causal_chain.startswith("Agent searched")
    assert analysis.agent_behavior_summary.startswith("Direct application")
    assert analysis.gold_patch_similarity == 1.0
    assert analysis.resolved is True
    # Schema is what the LLM was asked to validate against.
    assert llm.calls[0][2] is TrajectoryClassificationResponse


@pytest.mark.asyncio
async def test_classify_with_llm_falls_back_on_error():
    """If the LLM raises, fall back to the heuristic classifier — and we
    still get a structurally valid TrajectoryAnalysis."""

    class _BrokenLLM:
        async def query_structured(self, *args, **kwargs):
            raise RuntimeError("LLM is down")

    traj = _trajectory(final_patch=_GOLD_PATCH, resolved=True)
    analysis = await classify_with_llm(
        trajectory=traj,
        gold_patch=_GOLD_PATCH,
        problem_statement="x",
        f2p_test_names=[],
        llm=_BrokenLLM(),
    )
    # heuristic classifier sees gold-patch match
    assert analysis.leakage_pattern == LeakagePattern.GOLD_PATCH_LEAK
    assert analysis.resolved is True
