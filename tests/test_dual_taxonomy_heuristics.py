"""Regression tests for the heuristic pre-classifier in :mod:`bench_cleanser.classification.dual_taxonomy`.

The pre-classifier emits *candidates* (label + evidence) that the LLM
classifier may refine. The contract we lock in here is:

  * One ``TaskLabelAssignment`` per axis-1 label per task — never duplicates.
  * All accumulated evidence rolls into that single assignment.

The "no duplicate OVER_TEST" rule was a real bug fixed during the v1.0.0
release; this test prevents it from regressing.
"""

from __future__ import annotations

from bench_cleanser.classification.dual_taxonomy import _heuristic_labels
from bench_cleanser.models import (
    AssertionVerdict,
    AssertionVerdictReport,
    DescriptionClarity,
    IntentStatement,
    PatchAnalysis,
    TaskContaminationLabel,
    TestAnalysis,
    TestVerdict,
    TestVerdictReport,
)


def _make_clarity() -> DescriptionClarity:
    return DescriptionClarity(score=0.0, reasoning="")


def _make_intent() -> IntentStatement:
    return IntentStatement(
        instance_id="pkg/repo-1",
        core_requirement="r",
        behavioral_contract="c",
        acceptance_criteria=[],
        out_of_scope="",
        ambiguity_score=0.0,
    )


def _make_test_verdict(
    *,
    test_id: str,
    is_modified: bool,
    off_topic_count: int = 0,
    modification_aligned: bool = True,
) -> TestVerdictReport:
    """Build a TestVerdictReport with N OFF_TOPIC assertions."""
    return TestVerdictReport(
        test_id=test_id,
        test_name=test_id,
        intent_match=TestVerdict.ALIGNED,
        is_modified=is_modified,
        modification_aligned=modification_aligned,
        assertion_verdicts=[
            AssertionVerdictReport(
                statement=f"a{i}",
                verdict=AssertionVerdict.OFF_TOPIC,
                reason="excess",
            )
            for i in range(off_topic_count)
        ],
    )


def test_over_test_emits_one_candidate_even_with_overlapping_signals():
    """Regression: collapsed builder must not append OVER_TEST more than once.

    Scenario:
      * 2 off-topic assertions at the analysis level.
      * has_modified_tests=True.
      * One modified test with 1 OFF_TOPIC assertion AND misaligned modification.

    Expected: exactly ONE OVER_TEST candidate carrying all four evidence lines.
    """
    test_verdicts = [
        _make_test_verdict(
            test_id="tests/test_a.py::test_one",
            is_modified=True,
            off_topic_count=1,
            modification_aligned=False,
        ),
    ]
    test_analysis = TestAnalysis(
        total_tests=1,
        aligned_count=0,
        tangential_count=1,
        unrelated_count=0,
        total_assertions=3,
        on_topic_assertions=1,
        off_topic_assertions=2,
        has_modified_tests=True,
        test_verdicts=test_verdicts,
    )
    patch_analysis = PatchAnalysis(
        total_hunks=1, required_count=1,
        ancillary_count=0, unrelated_count=0,
    )

    candidates = _heuristic_labels(_make_intent(), patch_analysis, test_analysis, _make_clarity())

    over_test = [c for c in candidates if c.label == TaskContaminationLabel.OVER_TEST]
    assert len(over_test) == 1, (
        f"expected exactly one OVER_TEST candidate, got {len(over_test)}: "
        f"{[c.evidence for c in over_test]}"
    )

    # All four signals should have been preserved as evidence on the single
    # candidate — collapsing the candidates must not lose any evidence lines.
    evidence = over_test[0].evidence
    assert len(evidence) >= 3, (
        "expected ≥3 evidence lines (off-topic count, modified-with-off-topic, "
        f"misaligned-modification); got {evidence}"
    )
    joined = " | ".join(evidence)
    assert "OFF_TOPIC assertions" in joined
    assert "OFF_TOPIC" in joined and "Modified test" in joined
    assert "misaligned" in joined


def test_no_signals_emits_no_over_test():
    """Sanity: with all signals clean the pre-classifier does not emit OVER_TEST."""
    test_analysis = TestAnalysis(
        total_tests=2, aligned_count=2, tangential_count=0, unrelated_count=0,
        total_assertions=4, on_topic_assertions=4, off_topic_assertions=0,
        has_modified_tests=False,
    )
    patch_analysis = PatchAnalysis(
        total_hunks=1, required_count=1, ancillary_count=0, unrelated_count=0,
    )
    candidates = _heuristic_labels(_make_intent(), patch_analysis, test_analysis, _make_clarity())
    assert not any(c.label == TaskContaminationLabel.OVER_TEST for c in candidates)


def test_off_topic_assertions_alone_emits_one_candidate():
    """One firing signal still rolls up into a single candidate."""
    test_analysis = TestAnalysis(
        total_tests=1, aligned_count=0, tangential_count=1, unrelated_count=0,
        total_assertions=2, on_topic_assertions=0, off_topic_assertions=2,
        has_modified_tests=False,
    )
    patch_analysis = PatchAnalysis(
        total_hunks=0, required_count=0, ancillary_count=0, unrelated_count=0,
    )
    candidates = _heuristic_labels(_make_intent(), patch_analysis, test_analysis, _make_clarity())
    over_test = [c for c in candidates if c.label == TaskContaminationLabel.OVER_TEST]
    assert len(over_test) == 1
    assert any("OFF_TOPIC assertions" in e for e in over_test[0].evidence)
