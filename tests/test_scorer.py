"""Unit tests for the scorer module.

Tests focus on contamination patterns identified during manual audit:
- TEST_DESC_MISALIGN: django__django-15916 (mixed aligned/misaligned assertions)
- SNEAKY_TEST_MOD false positives: astropy-7671/8872 (aligned modifications)
- SNEAKY_TEST_MOD true positives: astropy-14365 (misaligned modifications)

v2 tests validate the 4-category intent-matching scorer:
- EXCESS_PATCH scoring from hunk verdicts
- EXCESS_TEST scoring from assertion verdicts
- VAGUE_SPEC passthrough
- Combined score formula
- End-to-end report building
"""

import pytest

from bench_cleanser.classification.scorer import (
    build_report,
    build_report_v2,
    compute_category_scores,
    compute_combined_score,
    compute_total_confidence,
    compute_verdict_scores,
)
from bench_cleanser.models import (
    AssertionVerdict,
    AssertionVerdictReport,
    CategoryScore,
    ContaminationCategory,
    CrossReferenceAnalysis,
    ExcessPatchDetail,
    ExcessTestDetail,
    HunkClassification,
    HunkReport,
    HunkVerdict,
    IntentStatement,
    PatchAnalysis,
    PatchVerdict,
    PipelineConfig,
    ScopeAnalysis,
    Severity,
    TestAnalysis,
    TestClassification,
    TestModificationType,
    TestReport,
    TestVerdict,
    TestVerdictReport,
    VagueSpecDetail,
    VerdictCategory,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _make_scope(instance_id: str, ambiguity: float = 0.1) -> ScopeAnalysis:
    return ScopeAnalysis(
        instance_id=instance_id,
        core_requirement="Fix the reported bug",
        affected_components=["module.py"],
        behavioral_contract="Function should work correctly",
        out_of_scope="Anything not described",
        ambiguity_score=ambiguity,
    )


def _make_patch(
    instance_id: str,
    hunks: list[tuple[HunkClassification, float]] | None = None,
) -> PatchAnalysis:
    if hunks is None:
        hunks = [(HunkClassification.IN_SCOPE, 0.9)]
    reports = [
        HunkReport(
            hunk_index=i,
            file_path=f"module{i}.py",
            classification=cls,
            confidence=conf,
            reasoning="test",
            is_heuristic=False,
        )
        for i, (cls, conf) in enumerate(hunks)
    ]
    oos = sum(1 for cls, _ in hunks if cls == HunkClassification.OUT_OF_SCOPE)
    bl = sum(1 for cls, _ in hunks if cls == HunkClassification.BORDERLINE)
    ins = sum(1 for cls, _ in hunks if cls == HunkClassification.IN_SCOPE)
    infra = sum(1 for cls, _ in hunks if cls == HunkClassification.INFRASTRUCTURE)
    total = len(hunks) or 1
    return PatchAnalysis(
        instance_id=instance_id,
        hunk_reports=reports,
        total_hunks=total,
        in_scope_count=ins,
        out_of_scope_count=oos,
        borderline_count=bl,
        infrastructure_count=infra,
        overpatch_score=oos / total,
    )


def _make_test_analysis(
    instance_id: str,
    test_reports: list[TestReport],
) -> TestAnalysis:
    total = len(test_reports) or 1
    misaligned = sum(
        1 for r in test_reports
        if r.classification in (TestClassification.MISALIGNED, TestClassification.PARTIALLY_ALIGNED)
    )
    sneaky = sum(
        1 for r in test_reports
        if r.classification == TestClassification.SNEAKY_MODIFICATION
    )
    return TestAnalysis(
        instance_id=instance_id,
        test_reports=test_reports,
        total_f2p_tests=len(test_reports),
        aligned_count=len(test_reports) - misaligned - sneaky,
        misaligned_count=misaligned,
        sneaky_mod_count=sneaky,
        overtest_score=(misaligned + sneaky) / total,
        sneaky_test_mod_score=sneaky / total,
    )


def _make_cross_ref(instance_id: str) -> CrossReferenceAnalysis:
    return CrossReferenceAnalysis(
        instance_id=instance_id,
        circular_dependencies=[],
        compound_patterns=[],
        circular_dependency_score=0.0,
    )


def _default_config() -> PipelineConfig:
    return PipelineConfig()


# ── Tests: TEST_DESC_MISALIGN (django__django-15916 pattern) ───────────


class TestDescMisalignScoring:
    """Validate TEST_DESC_MISALIGN detection for django__django-15916.

    The test_custom_callback_from_base_form_meta test has:
    - Assertion 1: checks modelform_factory preserves callback (ALIGNED)
    - Assertion 2: checks InheritedForm inherits widgets (NOT targeting #31721)

    Expected: TEST_DESC_MISALIGN confidence = 1/2 = 0.5
    """

    def test_mixed_assertions_detected(self):
        """1 of 2 assertions misaligned should produce 0.5 confidence."""
        instance_id = "django__django-15916"
        tests = _make_test_analysis(instance_id, [
            TestReport(
                test_id="test_custom_callback_in_meta",
                test_name="test_custom_callback_in_meta",
                modification_type=TestModificationType.NEW,
                classification=TestClassification.ALIGNED,
                confidence=0.9,
                reasoning="Directly tests formfield_callback in Meta",
                is_modified_existing=False,
                assertion_count=1,
                misaligned_assertion_count=0,
            ),
            TestReport(
                test_id="test_custom_callback_from_base_form_meta",
                test_name="test_custom_callback_from_base_form_meta",
                modification_type=TestModificationType.NEW,
                classification=TestClassification.PARTIALLY_ALIGNED,
                confidence=0.7,
                reasoning="Tests both factory callback and inheritance",
                is_modified_existing=False,
                assertion_count=2,
                misaligned_assertion_count=1,
            ),
        ])

        scores = compute_category_scores(
            _make_scope(instance_id),
            _make_patch(instance_id, [
                (HunkClassification.IN_SCOPE, 0.9),
                (HunkClassification.IN_SCOPE, 0.9),
                (HunkClassification.IN_SCOPE, 0.9),
            ]),
            tests,
            _make_cross_ref(instance_id),
        )

        misalign = scores[ContaminationCategory.TEST_DESC_MISALIGN.value]
        # 1 misaligned / 3 total assertions = 0.333
        assert misalign.confidence == pytest.approx(1 / 3, abs=0.01)
        assert len(misalign.evidence) == 1
        assert "1/3" in misalign.evidence[0]

    def test_all_assertions_aligned_no_signal(self):
        """When no assertions are misaligned, confidence should be 0."""
        instance_id = "django__django-15916-clean"
        tests = _make_test_analysis(instance_id, [
            TestReport(
                test_id="test_clean",
                test_name="test_clean",
                modification_type=TestModificationType.NEW,
                classification=TestClassification.ALIGNED,
                confidence=0.9,
                reasoning="Clean test",
                is_modified_existing=False,
                assertion_count=3,
                misaligned_assertion_count=0,
            ),
        ])

        scores = compute_category_scores(
            _make_scope(instance_id),
            _make_patch(instance_id),
            tests,
            _make_cross_ref(instance_id),
        )

        misalign = scores[ContaminationCategory.TEST_DESC_MISALIGN.value]
        assert misalign.confidence == 0.0
        assert len(misalign.evidence) == 0


# ── Tests: SNEAKY_TEST_MOD false positive suppression ──────────────────


class TestSneakyTestModScoring:
    """Validate SNEAKY_TEST_MOD scoring after the false-positive fix.

    The fix should:
    - Cap confidence at 0.30 when modified tests have 0 misaligned assertions
    - Keep confidence >= 0.90 when modified tests have misaligned assertions
    """

    def test_aligned_modification_capped(self):
        """astropy-7671/8872 pattern: modification is aligned, should be low confidence."""
        instance_id = "astropy__astropy-7671"
        tests = _make_test_analysis(instance_id, [
            TestReport(
                test_id="test_minversion",
                test_name="test_minversion",
                modification_type=TestModificationType.MODIFIED,
                classification=TestClassification.ALIGNED,
                confidence=0.9,
                reasoning="Added dev-tag version to good_versions list",
                is_modified_existing=True,
                assertion_count=1,
                misaligned_assertion_count=0,
            ),
        ])

        scores = compute_category_scores(
            _make_scope(instance_id),
            _make_patch(instance_id),
            tests,
            _make_cross_ref(instance_id),
        )

        sneaky = scores[ContaminationCategory.SNEAKY_TEST_MOD.value]
        # Should be capped at 0.30, not boosted to 0.90
        assert sneaky.confidence <= 0.30
        # Evidence should still be present (test was modified)
        assert len(sneaky.evidence) == 1

    def test_misaligned_modification_boosted(self):
        """astropy-14365 pattern: modification adds out-of-scope assertions."""
        instance_id = "astropy__astropy-14365"
        tests = _make_test_analysis(instance_id, [
            TestReport(
                test_id="test_roundtrip",
                test_name="test_roundtrip",
                modification_type=TestModificationType.MODIFIED,
                classification=TestClassification.SNEAKY_MODIFICATION,
                confidence=0.9,
                reasoning="Modified test uses uppercase-only commands",
                is_modified_existing=True,
                assertion_count=7,
                misaligned_assertion_count=7,
            ),
        ])

        scores = compute_category_scores(
            _make_scope(instance_id),
            _make_patch(instance_id),
            tests,
            _make_cross_ref(instance_id),
        )

        sneaky = scores[ContaminationCategory.SNEAKY_TEST_MOD.value]
        # Should be boosted to >= 0.90
        assert sneaky.confidence >= 0.90

    def test_no_modification_no_signal(self):
        """Entirely new tests should produce no SNEAKY_TEST_MOD signal."""
        instance_id = "clean-task"
        tests = _make_test_analysis(instance_id, [
            TestReport(
                test_id="test_new",
                test_name="test_new",
                modification_type=TestModificationType.NEW,
                classification=TestClassification.ALIGNED,
                confidence=0.9,
                reasoning="Brand new test",
                is_modified_existing=False,
                assertion_count=3,
                misaligned_assertion_count=0,
            ),
        ])

        scores = compute_category_scores(
            _make_scope(instance_id),
            _make_patch(instance_id),
            tests,
            _make_cross_ref(instance_id),
        )

        sneaky = scores[ContaminationCategory.SNEAKY_TEST_MOD.value]
        assert sneaky.confidence == 0.0
        assert len(sneaky.evidence) == 0


# ── Tests: End-to-end report building ──────────────────────────────────


class TestBuildReport:
    """Test the full build_report pipeline for django-15916 pattern."""

    def test_django_15916_with_correct_analysis(self):
        """If the LLM correctly identifies the misaligned assertion,
        the report should show MODERATE or higher severity."""
        instance_id = "django__django-15916"
        scope = _make_scope(instance_id, ambiguity=0.35)
        patch = _make_patch(instance_id, [
            (HunkClassification.IN_SCOPE, 0.9),
            (HunkClassification.IN_SCOPE, 0.9),
            (HunkClassification.IN_SCOPE, 0.9),
        ])
        tests = _make_test_analysis(instance_id, [
            TestReport(
                test_id="test_custom_callback_in_meta",
                test_name="test_custom_callback_in_meta",
                modification_type=TestModificationType.NEW,
                classification=TestClassification.ALIGNED,
                confidence=0.9,
                reasoning="Directly tests formfield_callback in Meta",
                is_modified_existing=False,
                assertion_count=1,
                misaligned_assertion_count=0,
            ),
            TestReport(
                test_id="test_custom_callback_from_base_form_meta",
                test_name="test_custom_callback_from_base_form_meta",
                modification_type=TestModificationType.NEW,
                classification=TestClassification.PARTIALLY_ALIGNED,
                confidence=0.7,
                reasoning="Second assertion checks inheritance, not the reported bug",
                is_modified_existing=False,
                assertion_count=2,
                misaligned_assertion_count=1,
            ),
        ])
        cross_ref = _make_cross_ref(instance_id)
        config = _default_config()

        report = build_report(scope, patch, tests, cross_ref, config)

        assert report.instance_id == instance_id
        # With 1/3 misaligned assertions + 0.35 ambiguity, total should be > MINOR
        assert report.total_confidence > 0.2
        # TEST_DESC_MISALIGN should be non-zero
        misalign = report.categories[ContaminationCategory.TEST_DESC_MISALIGN.value]
        assert misalign.confidence > 0.0

    def test_clean_task_produces_clean_report(self):
        """A task with all in-scope hunks and aligned tests should be CLEAN."""
        instance_id = "clean-task"
        scope = _make_scope(instance_id, ambiguity=0.1)
        patch = _make_patch(instance_id, [
            (HunkClassification.IN_SCOPE, 0.95),
        ])
        tests = _make_test_analysis(instance_id, [
            TestReport(
                test_id="test_good",
                test_name="test_good",
                modification_type=TestModificationType.NEW,
                classification=TestClassification.ALIGNED,
                confidence=0.95,
                reasoning="Perfectly aligned new test",
                is_modified_existing=False,
                assertion_count=3,
                misaligned_assertion_count=0,
            ),
        ])
        cross_ref = _make_cross_ref(instance_id)
        config = _default_config()

        report = build_report(scope, patch, tests, cross_ref, config)

        assert report.severity == Severity.CLEAN
        assert report.total_confidence < 0.2


# ═══════════════════════════════════════════════════════════════════════
# v2 Tests: 4-category intent-matching scorer
# ═══════════════════════════════════════════════════════════════════════


# ── v2 Helpers ────────────────────────────────────────────────────────


def _make_intent(instance_id: str, ambiguity: float = 0.1) -> IntentStatement:
    return IntentStatement(
        instance_id=instance_id,
        core_requirement="Fix the reported bug",
        behavioral_contract="Function should work correctly",
        acceptance_criteria=["The function handles edge case X"],
        out_of_scope="Anything not described",
        ambiguity_score=ambiguity,
        raw_llm_response="test",
    )


def _make_excess_patch(
    hunks: list[tuple[PatchVerdict, float]] | None = None,
) -> ExcessPatchDetail:
    if hunks is None:
        hunks = [(PatchVerdict.REQUIRED, 0.9)]
    hunk_verdicts = [
        HunkVerdict(
            hunk_index=i,
            file_path=f"module{i}.py",
            verdict=v,
            confidence=c,
            reasoning="test",
        )
        for i, (v, c) in enumerate(hunks)
    ]
    required = sum(1 for v, _ in hunks if v == PatchVerdict.REQUIRED)
    ancillary = sum(1 for v, _ in hunks if v == PatchVerdict.ANCILLARY)
    unrelated = sum(1 for v, _ in hunks if v == PatchVerdict.UNRELATED)
    total = len(hunks) or 1
    score = (unrelated + 0.5 * ancillary) / total
    return ExcessPatchDetail(
        score=score,
        total_hunks=len(hunks),
        required_count=required,
        ancillary_count=ancillary,
        unrelated_count=unrelated,
        hunk_verdicts=hunk_verdicts,
    )


def _make_excess_test(
    tests: list[tuple[TestVerdict, int, int, bool]] | None = None,
) -> ExcessTestDetail:
    """Create ExcessTestDetail.

    Each test tuple: (verdict, on_topic_count, off_topic_count, is_modified)
    """
    if tests is None:
        tests = [(TestVerdict.ALIGNED, 3, 0, False)]

    test_verdicts: list[TestVerdictReport] = []
    total_on = 0
    total_off = 0
    aligned = 0
    tangential = 0
    unrelated = 0
    has_modified = False

    for i, (verdict, on_topic, off_topic, is_mod) in enumerate(tests):
        assertions = (
            [AssertionVerdictReport(statement=f"assert_{j}", verdict=AssertionVerdict.ON_TOPIC) for j in range(on_topic)]
            + [AssertionVerdictReport(statement=f"assert_off_{j}", verdict=AssertionVerdict.OFF_TOPIC) for j in range(off_topic)]
        )
        test_verdicts.append(TestVerdictReport(
            test_id=f"test_{i}",
            test_name=f"test_{i}",
            intent_match=verdict,
            confidence=0.9,
            reasoning="test",
            is_modified=is_mod,
            assertion_verdicts=assertions,
        ))
        total_on += on_topic
        total_off += off_topic
        if verdict == TestVerdict.ALIGNED:
            aligned += 1
        elif verdict == TestVerdict.TANGENTIAL:
            tangential += 1
        else:
            unrelated += 1
        if is_mod:
            has_modified = True

    total_assertions = total_on + total_off
    total_tests = len(tests)
    avg = total_assertions / total_tests if total_tests else 1
    score = min(1.0, (total_off + unrelated * avg) / total_assertions) if total_assertions else 0.0

    return ExcessTestDetail(
        score=score,
        total_tests=total_tests,
        aligned_count=aligned,
        tangential_count=tangential,
        unrelated_count=unrelated,
        total_assertions=total_assertions,
        on_topic_assertions=total_on,
        off_topic_assertions=total_off,
        has_modified_tests=has_modified,
        test_verdicts=test_verdicts,
    )


# ── Tests: EXCESS_PATCH scoring ──────────────────────────────────────


class TestExcessPatchScoring:
    """Validate EXCESS_PATCH verdict scoring."""

    def test_all_required_is_zero(self):
        """All REQUIRED hunks should produce score = 0."""
        ep = _make_excess_patch([
            (PatchVerdict.REQUIRED, 0.9),
            (PatchVerdict.REQUIRED, 0.9),
            (PatchVerdict.REQUIRED, 0.9),
        ])
        scores = compute_verdict_scores(ep, _make_excess_test(), VagueSpecDetail(score=0.1))
        assert scores[VerdictCategory.EXCESS_PATCH.value].confidence == 0.0
        assert len(scores[VerdictCategory.EXCESS_PATCH.value].evidence) == 0

    def test_unrelated_hunks_score(self):
        """2/4 UNRELATED hunks should produce score = 0.5."""
        ep = _make_excess_patch([
            (PatchVerdict.REQUIRED, 0.9),
            (PatchVerdict.REQUIRED, 0.9),
            (PatchVerdict.UNRELATED, 0.8),
            (PatchVerdict.UNRELATED, 0.8),
        ])
        scores = compute_verdict_scores(ep, _make_excess_test(), VagueSpecDetail(score=0.1))
        assert scores[VerdictCategory.EXCESS_PATCH.value].confidence == pytest.approx(0.5)

    def test_ancillary_counts_half(self):
        """2 ANCILLARY out of 4 hunks should produce score = 0.25."""
        ep = _make_excess_patch([
            (PatchVerdict.REQUIRED, 0.9),
            (PatchVerdict.REQUIRED, 0.9),
            (PatchVerdict.ANCILLARY, 0.8),
            (PatchVerdict.ANCILLARY, 0.8),
        ])
        scores = compute_verdict_scores(ep, _make_excess_test(), VagueSpecDetail(score=0.1))
        assert scores[VerdictCategory.EXCESS_PATCH.value].confidence == pytest.approx(0.25)

    def test_mixed_unrelated_ancillary(self):
        """1 UNRELATED + 1 ANCILLARY out of 4 = (1 + 0.5) / 4 = 0.375."""
        ep = _make_excess_patch([
            (PatchVerdict.REQUIRED, 0.9),
            (PatchVerdict.REQUIRED, 0.9),
            (PatchVerdict.UNRELATED, 0.8),
            (PatchVerdict.ANCILLARY, 0.8),
        ])
        scores = compute_verdict_scores(ep, _make_excess_test(), VagueSpecDetail(score=0.1))
        assert scores[VerdictCategory.EXCESS_PATCH.value].confidence == pytest.approx(0.375)


# ── Tests: EXCESS_TEST scoring ───────────────────────────────────────


class TestExcessTestScoring:
    """Validate EXCESS_TEST verdict scoring."""

    def test_all_aligned_on_topic_is_zero(self):
        """All ALIGNED tests with ON_TOPIC assertions should produce score = 0."""
        et = _make_excess_test([
            (TestVerdict.ALIGNED, 3, 0, False),
            (TestVerdict.ALIGNED, 2, 0, False),
        ])
        scores = compute_verdict_scores(_make_excess_patch(), et, VagueSpecDetail(score=0.1))
        assert scores[VerdictCategory.EXCESS_TEST.value].confidence == 0.0

    def test_off_topic_assertions_detected(self):
        """1/3 OFF_TOPIC assertions should produce non-zero score."""
        et = _make_excess_test([
            (TestVerdict.ALIGNED, 2, 1, False),
        ])
        scores = compute_verdict_scores(_make_excess_patch(), et, VagueSpecDetail(score=0.1))
        ep_score = scores[VerdictCategory.EXCESS_TEST.value]
        assert ep_score.confidence == pytest.approx(1 / 3, abs=0.01)
        assert any("OFF_TOPIC" in e for e in ep_score.evidence)

    def test_modified_test_flagged(self):
        """Modified pre-existing test should appear in evidence."""
        et = _make_excess_test([
            (TestVerdict.ALIGNED, 2, 0, True),
        ])
        scores = compute_verdict_scores(_make_excess_patch(), et, VagueSpecDetail(score=0.1))
        ep_score = scores[VerdictCategory.EXCESS_TEST.value]
        assert any("modified" in e.lower() for e in ep_score.evidence)

    def test_django_15916_pattern(self):
        """Django-15916: 1 test aligned, 1 tangential with 1/2 off-topic assertion."""
        et = _make_excess_test([
            (TestVerdict.ALIGNED, 1, 0, False),
            (TestVerdict.TANGENTIAL, 1, 1, False),
        ])
        scores = compute_verdict_scores(_make_excess_patch(), et, VagueSpecDetail(score=0.35))
        et_score = scores[VerdictCategory.EXCESS_TEST.value]
        # 1 OFF_TOPIC / 3 total assertions = 0.333
        assert et_score.confidence == pytest.approx(1 / 3, abs=0.01)


# ── Tests: Combined score formula ────────────────────────────────────


class TestCombinedScore:
    """Validate the combined contamination score formula."""

    def test_all_zero_is_zero(self):
        """When all scores are 0, combined should be 0."""
        scores = compute_verdict_scores(
            _make_excess_patch(),
            _make_excess_test(),
            VagueSpecDetail(score=0.0),
        )
        combined = compute_combined_score(scores)
        assert combined == 0.0

    def test_single_category_passes_through(self):
        """With only one non-zero category, combined equals that category."""
        ep = _make_excess_patch([
            (PatchVerdict.REQUIRED, 0.9),
            (PatchVerdict.UNRELATED, 0.8),
        ])
        scores = compute_verdict_scores(ep, _make_excess_test(), VagueSpecDetail(score=0.0))
        combined = compute_combined_score(scores)
        # 1/2 = 0.5
        assert combined == pytest.approx(0.5)

    def test_two_categories_combine(self):
        """Combined = 1 - (1-0.5)*(1-0.333) = 1 - 0.333 = 0.667."""
        ep = _make_excess_patch([
            (PatchVerdict.REQUIRED, 0.9),
            (PatchVerdict.UNRELATED, 0.8),
        ])
        et = _make_excess_test([
            (TestVerdict.ALIGNED, 2, 1, False),
        ])
        scores = compute_verdict_scores(ep, et, VagueSpecDetail(score=0.0))
        combined = compute_combined_score(scores)
        expected = 1 - (1 - 0.5) * (1 - 1/3)
        assert combined == pytest.approx(expected, abs=0.01)

    def test_all_max_is_one(self):
        """If all categories are 1.0, combined should be 1.0."""
        ep = _make_excess_patch([
            (PatchVerdict.UNRELATED, 0.9),
        ])
        et = _make_excess_test([
            (TestVerdict.UNRELATED, 0, 0, False),
        ])
        # unrelated test with 0 assertions — score computed with avg
        # Score may not be exactly 1.0 but let's test with vague_spec=1.0
        scores = compute_verdict_scores(ep, et, VagueSpecDetail(score=1.0))
        combined = compute_combined_score(scores)
        assert combined == 1.0


# ── Tests: v2 End-to-end report building ─────────────────────────────


class TestBuildReportV2:
    """Test the full build_report_v2 pipeline."""

    def test_clean_task_produces_clean(self):
        """All REQUIRED hunks + ALIGNED tests + low ambiguity = CLEAN."""
        intent = _make_intent("clean-task", ambiguity=0.1)
        ep = _make_excess_patch([
            (PatchVerdict.REQUIRED, 0.95),
        ])
        et = _make_excess_test([
            (TestVerdict.ALIGNED, 3, 0, False),
        ])
        vs = VagueSpecDetail(score=0.1)
        config = _default_config()

        report = build_report_v2(intent, ep, et, vs, config)
        assert report.severity == Severity.CLEAN
        assert report.combined_score < 0.2
        assert report.instance_id == "clean-task"
        assert any("No contamination" in r or "fair" in r.lower() for r in report.recommendations)

    def test_excess_patch_triggers_moderate(self):
        """Significant unrelated hunks should produce MODERATE+ severity."""
        intent = _make_intent("patch-heavy")
        ep = _make_excess_patch([
            (PatchVerdict.REQUIRED, 0.9),
            (PatchVerdict.UNRELATED, 0.8),
            (PatchVerdict.UNRELATED, 0.8),
            (PatchVerdict.UNRELATED, 0.8),
        ])
        et = _make_excess_test()
        vs = VagueSpecDetail(score=0.1)
        config = _default_config()

        report = build_report_v2(intent, ep, et, vs, config)
        assert report.combined_score >= 0.5
        assert report.severity in (Severity.MODERATE, Severity.SEVERE)
        assert any("EXCESS_PATCH" in r for r in report.recommendations)

    def test_excess_test_triggers_signal(self):
        """Off-topic assertions should produce non-zero EXCESS_TEST."""
        intent = _make_intent("django__django-15916")
        ep = _make_excess_patch()
        et = _make_excess_test([
            (TestVerdict.ALIGNED, 1, 0, False),
            (TestVerdict.TANGENTIAL, 1, 1, False),
        ])
        vs = VagueSpecDetail(score=0.35)
        config = _default_config()

        report = build_report_v2(intent, ep, et, vs, config)
        assert report.excess_test.off_topic_assertions == 1
        assert report.excess_test.score > 0.0
        assert report.combined_score > 0.2

    def test_report_serialization(self):
        """to_dict() should produce valid JSON-compatible structure."""
        intent = _make_intent("serialize-test", ambiguity=0.5)
        ep = _make_excess_patch([
            (PatchVerdict.REQUIRED, 0.9),
            (PatchVerdict.ANCILLARY, 0.8),
        ])
        et = _make_excess_test([
            (TestVerdict.ALIGNED, 2, 1, True),
        ])
        vs = VagueSpecDetail(score=0.5, reasoning="Moderate ambiguity")
        config = _default_config()

        report = build_report_v2(intent, ep, et, vs, config)
        d = report.to_dict()

        assert d["instance_id"] == "serialize-test"
        assert d["severity"] in ("CLEAN", "MINOR", "MODERATE", "SEVERE")
        assert "excess_patch" in d
        assert "excess_test" in d
        assert "vague_spec" in d
        assert "recommendations" in d
        assert isinstance(d["excess_patch"]["hunks"], list)
        assert isinstance(d["excess_test"]["tests"], list)
        assert len(d["excess_test"]["tests"][0]["assertions"]) == 3
