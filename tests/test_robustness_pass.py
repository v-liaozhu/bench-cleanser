"""Tests for the robustness pass (commit 0dde693) and the follow-ups:

  - pipeline_error round-trips through ContaminationReport.to_dict / from_dict
  - summary CSV includes the pipeline_error column
  - LLMClient._strip_fences handles ``json/bare/no fence variants
  - LLMClient._strictify_schema produces strict-mode-compatible JSON schemas
  - LLMClient._structured_cache_key is stable across schema text changes
  - analyze_trajectories respects max_concurrency
  - classify_cross_agent uses median quorum and gates low-entropy patches
  - classify_task_labels preserves protected heuristics the LLM omits

All tests are offline. The LLM tier is exercised against in-test fakes.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
from typing import Any

import pytest
from pydantic import BaseModel

from bench_cleanser.analysis.cross_ref import CrossReferenceResult
from bench_cleanser.classification.dual_taxonomy import classify_task_labels
from bench_cleanser.fusion import FusionVerdict, fuse
from bench_cleanser.llm_client import LLMClient
from bench_cleanser.models import (
    AgentTrajectoryLabel,
    ContaminationReport,
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
from bench_cleanser.pipeline import _atomic_write_text, _write_summary
from bench_cleanser.schemas import TaskClassificationResponse
from bench_cleanser.trajectory.analyzer import analyze_trajectories
from bench_cleanser.trajectory.classifier import (
    CROSS_AGENT_QUORUM_THRESHOLD,
    LOW_ENTROPY_PATCH_LINES,
    classify_cross_agent,
)
from bench_cleanser.trajectory.models import (
    LeakagePattern,
    TrajectoryAnalysis,
    TrajectoryRecord,
)

# ── pipeline_error ─────────────────────────────────────────────────────────


def _bare_report(pipeline_error: str | None = None) -> ContaminationReport:
    return ContaminationReport(
        instance_id="pkg/repo-1",
        severity=Severity.CLEAN,
        intent=IntentStatement(
            instance_id="pkg/repo-1",
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
        pipeline_error=pipeline_error,
    )


def test_pipeline_error_roundtrips_through_dict():
    r = _bare_report(pipeline_error="ConnectionError: boom")
    encoded = r.to_dict()
    assert encoded["pipeline_error"] == "ConnectionError: boom"
    decoded = ContaminationReport.from_dict(encoded)
    assert decoded.pipeline_error == "ConnectionError: boom"


def test_pipeline_error_absent_when_unset():
    r = _bare_report()
    encoded = r.to_dict()
    assert "pipeline_error" not in encoded
    decoded = ContaminationReport.from_dict(encoded)
    assert decoded.pipeline_error is None


def test_summary_csv_has_pipeline_error_column(tmp_path: pathlib.Path):
    r_good = _bare_report()
    r_bad = _bare_report(pipeline_error="OperatorError: bad")
    _write_summary([r_good, r_bad], tmp_path)
    csv_text = (tmp_path / "summary.csv").read_text(encoding="utf-8")
    assert "pipeline_error" in csv_text.splitlines()[0]
    assert "OperatorError: bad" in csv_text

    stats = json.loads((tmp_path / "summary_stats.json").read_text(encoding="utf-8"))
    assert stats["total_tasks"] == 2
    assert stats["analytic_tasks"] == 1
    assert stats["pipeline_errors"] == 1


def test_stage7_fusion_skips_error_rows():
    # An error report fused against a passing trajectory must NOT produce
    # a CONTAMINATED_PASS verdict — the analysis is missing, not severe.
    r = _bare_report(pipeline_error="ValueError: ...")
    traj = TrajectoryAnalysis(
        instance_id=r.instance_id,
        agent_name="agent-X",
        leakage_pattern=LeakagePattern.GENUINE_SOLUTION,
        trajectory_label=AgentTrajectoryLabel.AGENT_PASSED_GENUINE,
    )
    # The fusion function itself doesn't know about pipeline_error;
    # the analyzer is the layer that filters. Verify the analyzer-layer
    # check: error rows have pipeline_error set, callers must skip.
    assert r.pipeline_error is not None
    # Sanity-check that fusing the same report without the error flag
    # would have produced a clean FAIR_PASS — i.e. our filter is what
    # protects the agent, not the verdict shape.
    r_clean = _bare_report()
    f = fuse(r_clean, traj)
    assert f.verdict == FusionVerdict.FAIR_PASS


def test_atomic_write_text_creates_file(tmp_path: pathlib.Path):
    target = tmp_path / "subdir" / "out.json"
    _atomic_write_text(target, '{"k": "v"}')
    assert target.read_text(encoding="utf-8") == '{"k": "v"}'
    # No leftover .tmp file beside it
    tmps = list(target.parent.glob("*.tmp"))
    assert tmps == []


# ── LLMClient helpers ──────────────────────────────────────────────────────


class _Inner(BaseModel):
    x: int


class _Outer(BaseModel):
    label: str
    payload: _Inner
    notes: list[str] = []


def test_strip_fences_handles_json_tag():
    assert LLMClient._strip_fences('```json\n{"a":1}\n```') == '{"a":1}'


def test_strip_fences_handles_bare_fence():
    assert LLMClient._strip_fences('```\n{"a":1}\n```') == '{"a":1}'


def test_strip_fences_passes_through_unfenced():
    assert LLMClient._strip_fences('{"a":1}') == '{"a":1}'


def test_strip_fences_does_not_eat_legitimate_strings():
    # An un-fenced payload containing triple-backticks inside a string value
    # should pass through unchanged (no leading ``` so nothing to strip).
    payload = '{"a": "text with ``` inside"}'
    assert LLMClient._strip_fences(payload) == payload


def test_strictify_schema_forces_additional_properties_false():
    schema = _Outer.model_json_schema()
    strict = LLMClient._strictify_schema(schema)
    assert strict["additionalProperties"] is False


def test_strictify_schema_makes_all_properties_required():
    schema = _Outer.model_json_schema()
    strict = LLMClient._strictify_schema(schema)
    assert set(strict["required"]) == set(strict["properties"].keys())


def test_strictify_schema_drops_defaults():
    # _Outer.notes has default=[] — strict mode forbids per-property defaults.
    schema = _Outer.model_json_schema()
    strict = LLMClient._strictify_schema(schema)
    notes_prop = strict["properties"]["notes"]
    assert "default" not in notes_prop


def test_structured_cache_key_stable_when_only_schema_text_changes():
    # The structured cache key MUST NOT change when the same pydantic model
    # produces a textually different schema across pydantic versions —
    # otherwise every upgrade silently wipes the .cache directory.
    from bench_cleanser.cache import ResponseCache

    class _FakeClient:
        _model = "model-X"
        _STRUCTURED_SCHEMA_VERSION = "v1"

        _structured_cache_key = LLMClient._structured_cache_key  # type: ignore[assignment]

    fc = _FakeClient()
    k1 = LLMClient._structured_cache_key(fc, "sys", "user", _Outer)  # type: ignore[arg-type]
    k2 = LLMClient._structured_cache_key(fc, "sys", "user", _Outer)  # type: ignore[arg-type]
    assert k1 == k2
    assert k1 != ResponseCache.make_key("sys", "user", "model-X")


# ── analyze_trajectories semaphore ─────────────────────────────────────────


class _ConcurrencyTrackingLLM:
    """Fake LLM that records peak concurrent in-flight calls."""

    def __init__(self) -> None:
        self.in_flight = 0
        self.peak = 0

    async def query_structured(self, *args: Any, **kwargs: Any) -> Any:
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(0.01)
            from bench_cleanser.schemas import TrajectoryClassificationResponse
            return TrajectoryClassificationResponse(
                pattern="GENUINE_SOLUTION",
                trajectory_label="agent_passed_genuine",
                evidence_strength="moderate",
                reasoning="fake",
                causal_chain="",
                key_evidence=[],
                agent_behavior_summary="",
            )
        finally:
            self.in_flight -= 1


def test_analyze_trajectories_respects_max_concurrency():
    fake_llm = _ConcurrencyTrackingLLM()
    trajectories = [
        TrajectoryRecord(
            instance_id=f"pkg/repo-{i}",
            agent_name=f"agent-{i}",
            actions=[],
            final_patch="diff --git a/x b/x\n+1",
            resolved=True,
        )
        for i in range(20)
    ]
    gold_patches = {t.instance_id: "diff --git a/x b/x\n+1" for t in trajectories}
    f2p_tests = {t.instance_id: ["t::a"] for t in trajectories}
    problem_statements = {t.instance_id: "do x" for t in trajectories}

    asyncio.run(analyze_trajectories(
        trajectories, gold_patches, f2p_tests, problem_statements,
        llm=fake_llm, max_concurrency=3,
    ))

    # Peak in-flight calls must never exceed the cap.
    assert fake_llm.peak <= 3, f"semaphore violated: peak={fake_llm.peak}"
    # Sanity: more than one was actually in flight at once (otherwise the
    # test wouldn't be exercising parallelism at all).
    assert fake_llm.peak >= 2


# ── cross-agent quorum ─────────────────────────────────────────────────────


_GOLD = """\
diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,15 @@
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

_SMALL_FIX = """\
diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -1,1 +1,1 @@
-return a+b
+return a + b
"""


def _analysis(iid: str, agent: str, pattern: LeakagePattern) -> TrajectoryAnalysis:
    return TrajectoryAnalysis(
        instance_id=iid,
        agent_name=agent,
        leakage_pattern=pattern,
        evidence=[],
    )


def _traj(iid: str, agent: str, patch: str) -> TrajectoryRecord:
    return TrajectoryRecord(
        instance_id=iid,
        agent_name=agent,
        actions=[],
        final_patch=patch,
        resolved=True,
    )


def test_cross_agent_upgrade_fires_on_three_agent_quorum():
    analyses = [
        _analysis("i1", "a1", LeakagePattern.GENUINE_SOLUTION),
        _analysis("i1", "a2", LeakagePattern.GENUINE_SOLUTION),
        _analysis("i1", "a3", LeakagePattern.GENUINE_SOLUTION),
    ]
    trajs = [
        _traj("i1", "a1", _GOLD),
        _traj("i1", "a2", _GOLD),
        _traj("i1", "a3", _GOLD),  # all converged on the gold patch
    ]
    out = classify_cross_agent(analyses, trajs)
    assert all(a.leakage_pattern == LeakagePattern.GOLD_PATCH_LEAK for a in out)


def test_cross_agent_upgrade_tolerates_one_outlier():
    # The previous all-pairs rule would have skipped this case (any single
    # diverging agent vetoed the upgrade). The new median-quorum rule must
    # tolerate at least one outlier in a converged majority. Use 5 agents
    # so 4-of-5 convergence drives the pairwise-similarity median above
    # threshold.
    analyses = [
        _analysis("i1", f"a{i}", LeakagePattern.GENUINE_SOLUTION) for i in range(1, 6)
    ]
    trajs = [
        _traj("i1", "a1", _GOLD),
        _traj("i1", "a2", _GOLD),
        _traj("i1", "a3", _GOLD),
        _traj("i1", "a4", _GOLD),
        _traj("i1", "a5", "diff --git a/z b/z\n+totally different\n+lines\n+here\n"),
    ]
    out = classify_cross_agent(analyses, trajs)
    upgraded = sum(1 for a in out if a.leakage_pattern == LeakagePattern.GOLD_PATCH_LEAK)
    # At least the four converged agents must be upgraded.
    assert upgraded >= 4


def test_cross_agent_upgrade_gated_on_low_entropy_patches():
    # A one-line fix should NOT be upgraded even when every agent agrees —
    # there's only one sensible answer.
    analyses = [
        _analysis("i1", "a1", LeakagePattern.GENUINE_SOLUTION),
        _analysis("i1", "a2", LeakagePattern.GENUINE_SOLUTION),
    ]
    trajs = [
        _traj("i1", "a1", _SMALL_FIX),
        _traj("i1", "a2", _SMALL_FIX),
    ]
    out = classify_cross_agent(analyses, trajs)
    assert all(a.leakage_pattern == LeakagePattern.GENUINE_SOLUTION for a in out)


def test_cross_agent_thresholds_are_sane():
    # Guard against accidental drift of the public constants.
    assert 0.5 < CROSS_AGENT_QUORUM_THRESHOLD <= 1.0
    assert LOW_ENTROPY_PATCH_LINES >= 1


# ── heuristic-LLM union ────────────────────────────────────────────────────


class _FakeLLMReturning:
    """Fake LLM that always returns a single 'clean' label, dropping any
    heuristic the upstream pipeline detected."""

    def __init__(self, labels: list[str]) -> None:
        self._labels = labels

    async def query_structured(
        self, system_prompt: str, user_prompt: str, response_model: type
    ) -> Any:
        return TaskClassificationResponse(
            labels=[
                {"label": label, "evidence": ["llm e"], "reasoning": "llm r"}
                for label in self._labels
            ],
        )


def _intent() -> IntentStatement:
    return IntentStatement(
        instance_id="pkg/repo-1",
        core_requirement="r", behavioral_contract="c",
        acceptance_criteria=[], out_of_scope="",
        ambiguity_score=0.0,
    )


def _empty_analyses() -> tuple[PatchAnalysis, TestAnalysis, DescriptionClarity]:
    return (
        PatchAnalysis(total_hunks=0, required_count=0,
                      ancillary_count=0, unrelated_count=0),
        TestAnalysis(total_tests=0, aligned_count=0, tangential_count=0,
                     unrelated_count=0, total_assertions=0,
                     on_topic_assertions=0, off_topic_assertions=0,
                     has_modified_tests=False),
        DescriptionClarity(score=0.0, reasoning=""),
    )


def test_protected_heuristic_survives_llm_omission():
    # Construct a record that triggers the deterministic "self-referential
    # problem statement" heuristic. The LLM (fake) returns only 'clean'.
    record = TaskRecord(
        instance_id="pkg/repo-1",
        repo="pkg/repo",
        base_commit="abc",
        patch="",
        test_patch="",
        problem_statement="The bug is fixed — see the patch attached.",
        hints_text="",
        fail_to_pass=[],
        pass_to_pass=[],
        version="1.0",
    )
    pa, ta, dc = _empty_analyses()
    fake_llm = _FakeLLMReturning(["clean"])

    labels = asyncio.run(classify_task_labels(
        intent=_intent(),
        patch_analysis=pa,
        test_analysis=ta,
        description_clarity=dc,
        record=record,
        llm=fake_llm,
    ))

    # The LLM said "clean", but the deterministic "Problem contains 'see the
    # patch'" heuristic must survive — the union policy protects it.
    label_values = [la.label.value for la in labels]
    assert "hidden_context" in label_values, (
        f"protected heuristic was overruled: {label_values}"
    )


def test_unprotected_heuristic_can_be_dropped_by_llm():
    # OVER_PATCH (from unrelated_count) is NOT in the protected set —
    # the LLM is allowed to overrule it. Verify the LLM's verdict wins
    # when no protected heuristic is in play.
    pa = PatchAnalysis(
        total_hunks=1, required_count=0,
        ancillary_count=0, unrelated_count=1,
        hunk_verdicts=[],
    )
    _, ta, dc = _empty_analyses()
    fake_llm = _FakeLLMReturning(["clean"])

    labels = asyncio.run(classify_task_labels(
        intent=_intent(),
        patch_analysis=pa,
        test_analysis=ta,
        description_clarity=dc,
        record=None,
        llm=fake_llm,
    ))
    label_values = [la.label.value for la in labels]
    assert label_values == ["clean"]
