"""Core data models for the bench-cleanser pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ContaminationCategory(str, Enum):
    """Taxonomy of benchmark contamination types (v1 — 7 categories)."""
    OVERTEST = "OVERTEST"
    OVERPATCH = "OVERPATCH"
    SNEAKY_TEST_MOD = "SNEAKY_TEST_MOD"
    SCOPE_CREEP = "SCOPE_CREEP"
    TEST_DESC_MISALIGN = "TEST_DESC_MISALIGN"
    CIRCULAR_DEPENDENCY = "CIRCULAR_DEPENDENCY"
    AMBIGUOUS_SPEC = "AMBIGUOUS_SPEC"


# ── v2 Taxonomy: 4 Verdict Categories ────────────────────────────────


class VerdictCategory(str, Enum):
    """v2 taxonomy: non-overlapping, actionable verdict categories."""
    EXCESS_PATCH = "EXCESS_PATCH"    # Gold patch includes changes beyond task scope
    EXCESS_TEST = "EXCESS_TEST"      # F2P tests verify behavior beyond task scope
    VAGUE_SPEC = "VAGUE_SPEC"        # Problem statement is ambiguous
    CLEAN = "CLEAN"                  # No contamination detected


class PatchVerdict(str, Enum):
    """Per-hunk verdict for gold patch intent matching."""
    REQUIRED = "REQUIRED"        # Directly implements the described fix
    ANCILLARY = "ANCILLARY"      # Supports the fix but isn't described (imports, infra)
    UNRELATED = "UNRELATED"      # Changes behavior not described in the problem


class TestVerdict(str, Enum):
    """Per-test verdict for F2P test intent matching."""
    ALIGNED = "ALIGNED"          # Test targets the described problem
    TANGENTIAL = "TANGENTIAL"    # Test partially targets the problem
    UNRELATED = "UNRELATED"      # Test doesn't target the described problem


class AssertionVerdict(str, Enum):
    """Per-assertion verdict for F2P test intent matching."""
    ON_TOPIC = "ON_TOPIC"        # Assertion checks behavior described in the problem
    OFF_TOPIC = "OFF_TOPIC"      # Assertion checks behavior NOT described in the problem


class Severity(str, Enum):
    """Severity classification for contaminated tasks."""
    CLEAN = "CLEAN"
    MINOR = "MINOR"
    MODERATE = "MODERATE"
    SEVERE = "SEVERE"


class HunkClassification(str, Enum):
    """Classification of a gold patch hunk relative to task scope."""
    IN_SCOPE = "IN_SCOPE"
    BORDERLINE = "BORDERLINE"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    INFRASTRUCTURE = "INFRASTRUCTURE"


class TestClassification(str, Enum):
    """Classification of an F2P test relative to task scope."""
    ALIGNED = "ALIGNED"
    PARTIALLY_ALIGNED = "PARTIALLY_ALIGNED"
    MISALIGNED = "MISALIGNED"
    SNEAKY_MODIFICATION = "SNEAKY_MODIFICATION"


class TestModificationType(str, Enum):
    """Whether a test in the test_patch is new or modified."""
    NEW = "NEW"
    MODIFIED = "MODIFIED"
    UNKNOWN = "UNKNOWN"


# --- Stage 1: Parsing ---

@dataclass
class TaskRecord:
    """Raw SWE-bench task record."""
    instance_id: str
    repo: str
    base_commit: str
    patch: str  # gold patch (unified diff)
    test_patch: str  # test modifications (unified diff)
    problem_statement: str
    hints_text: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    version: str
    environment_setup_commit: str = ""
    created_at: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskRecord:
        """Create from a SWE-bench dataset row."""
        f2p = data.get("FAIL_TO_PASS", "[]")
        p2p = data.get("PASS_TO_PASS", "[]")
        if isinstance(f2p, str):
            f2p = json.loads(f2p)
        if isinstance(p2p, str):
            p2p = json.loads(p2p)

        return cls(
            instance_id=data["instance_id"],
            repo=data.get("repo", ""),
            base_commit=data.get("base_commit", ""),
            patch=data.get("patch", ""),
            test_patch=data.get("test_patch", ""),
            problem_statement=data.get("problem_statement", ""),
            hints_text=data.get("hints_text", ""),
            fail_to_pass=f2p,
            pass_to_pass=p2p,
            version=data.get("version", ""),
            environment_setup_commit=data.get("environment_setup_commit", ""),
            created_at=data.get("created_at", ""),
        )


@dataclass
class PatchHunk:
    """A single hunk from a unified diff."""
    file_path: str
    hunk_index: int  # Index within the file's hunks
    header: str  # @@ line
    added_lines: list[str]
    removed_lines: list[str]
    context_lines: list[str]
    function_context: str  # Function name from @@ header if available
    raw_diff: str  # The raw hunk text

    @property
    def is_test_file(self) -> bool:
        return "test" in self.file_path.lower()

    @property
    def is_init_file(self) -> bool:
        return self.file_path.endswith("__init__.py")

    @property
    def is_doc_file(self) -> bool:
        lower = self.file_path.lower()
        # Only match standalone docs/ directories, not substrings like "admindocs/"
        parts = lower.replace("\\", "/").split("/")
        if any(p == "docs" for p in parts):
            return True
        return any(
            pat in lower
            for pat in ["readme", "changelog", "contributing", ".md", ".rst"]
        ) and not lower.endswith(".py")

    @property
    def net_lines_changed(self) -> int:
        return len(self.added_lines) + len(self.removed_lines)


@dataclass
class CallTarget:
    """A function/method call found in test source via AST."""
    name: str                    # e.g., "Run", "_check_regexp_csv"
    module: str                  # resolved module path (or "" if unresolved)
    file_path: str               # resolved file in repo (or "")
    line_number: int             # line in test source
    is_in_patch: bool            # True if this target is in a gold-patch file


@dataclass
class Assertion:
    """A structured assertion extracted from a test via AST."""
    statement: str               # full assertion line
    assertion_type: str          # "assert", "assertEqual", "assertRaises", etc.
    target_expression: str       # what's being asserted on
    expected_value: str          # expected result (if extractable)


@dataclass
class TestedFunction:
    """A source function that a test exercises."""
    name: str
    file_path: str
    source: str                  # full function source from repo
    is_modified_by_patch: bool   # True if gold patch modifies this function


@dataclass
class CodeContext:
    """Full code context retrieved via code visitation (repo clone)."""
    pre_patch_test_source: str       # full test function BEFORE patch
    post_patch_test_source: str      # full test function AFTER patch
    test_file_imports: str           # import block from test file
    test_file_fixtures: str          # fixtures/setup used by test
    tested_functions: list[TestedFunction]  # source code being tested
    call_targets: list[CallTarget]   # all calls from test body
    assertions: list[Assertion]      # structured assertions
    test_file_path: str
    repo_path: str                   # local clone path


@dataclass
class TestHunk:
    """A test function diff extracted from the test_patch."""
    file_path: str
    test_name: str  # e.g., "test_csv_regex_error"
    full_test_id: str  # e.g., "tests/config/test_config.py::test_csv_regex_error"
    modification_type: TestModificationType
    added_lines: list[str]
    removed_lines: list[str]
    full_source: str  # Reconstructed test function source (from + lines)
    raw_diff: str
    code_context: CodeContext | None = None  # populated by code visitation


@dataclass
class ParsedTask:
    """Fully parsed SWE-bench task, ready for analysis."""
    record: TaskRecord
    patch_hunks: list[PatchHunk]
    test_hunks: list[TestHunk]
    f2p_test_hunks: list[TestHunk]  # Test hunks matching F2P test IDs
    f2p_tests_with_no_hunk: list[str]  # F2P test IDs with no matching hunk
    files_in_gold_patch: list[str]
    files_in_test_patch: list[str]


# --- Stage 2: Scope Analysis ---

@dataclass
class ScopeAnalysis:
    """LLM-derived analysis of what the task actually asks for."""
    instance_id: str
    core_requirement: str
    affected_components: list[str]
    behavioral_contract: str
    out_of_scope: str
    ambiguity_score: float  # 0.0 = perfectly clear, 1.0 = very ambiguous
    raw_llm_response: str = ""


# --- Stage 3: Patch Analysis ---

@dataclass
class HunkReport:
    """Analysis result for a single gold patch hunk."""
    hunk_index: int
    file_path: str
    classification: HunkClassification
    confidence: float
    reasoning: str
    is_heuristic: bool  # True if classified by heuristic, False if by LLM


@dataclass
class PatchAnalysis:
    """Complete analysis of the gold patch."""
    instance_id: str
    hunk_reports: list[HunkReport]
    total_hunks: int
    in_scope_count: int
    out_of_scope_count: int
    borderline_count: int
    infrastructure_count: int
    overpatch_score: float  # Fraction of out-of-scope hunks


# --- Stage 4: Test Analysis ---

@dataclass
class TestReport:
    """Analysis result for a single F2P test."""
    test_id: str
    test_name: str
    modification_type: TestModificationType
    classification: TestClassification
    confidence: float
    reasoning: str
    is_modified_existing: bool  # Deterministic signal: test existed before
    assertion_count: int
    misaligned_assertion_count: int


@dataclass
class TestAnalysis:
    """Complete analysis of the test patch and F2P tests."""
    instance_id: str
    test_reports: list[TestReport]
    total_f2p_tests: int
    aligned_count: int
    misaligned_count: int
    sneaky_mod_count: int
    overtest_score: float
    sneaky_test_mod_score: float


# --- Stage 5: Cross-Reference ---

@dataclass
class CircularDependency:
    """A detected circular dependency between an F2P test and out-of-scope hunks."""
    test_id: str
    out_of_scope_hunks: list[int]  # Indices of OOS hunks the test exercises
    confidence: float
    reasoning: str


@dataclass
class CrossReferenceAnalysis:
    """Cross-reference analysis results."""
    instance_id: str
    circular_dependencies: list[CircularDependency]
    compound_patterns: list[str]  # e.g., ["SNEAKY+CIRCULAR", "OVERPATCH+OVERTEST"]
    circular_dependency_score: float


# --- Stage 6: Classification ---

@dataclass
class CategoryScore:
    """Confidence score for a single contamination category."""
    category: ContaminationCategory
    confidence: float
    evidence: list[str]


@dataclass
class ContaminationReport:
    """Final contamination report for a single task."""
    instance_id: str
    severity: Severity
    total_confidence: float
    categories: dict[str, CategoryScore]
    f2p_test_reports: list[TestReport]
    patch_hunk_reports: list[HunkReport]
    compound_patterns: list[str]
    evidence_summary: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "instance_id": self.instance_id,
            "severity": self.severity.value,
            "total_confidence": round(self.total_confidence, 4),
            "categories": {
                name: {
                    "category": score.category.value,
                    "confidence": round(score.confidence, 4),
                    "evidence": score.evidence,
                }
                for name, score in self.categories.items()
            },
            "f2p_test_reports": [
                {
                    "test_id": tr.test_id,
                    "test_name": tr.test_name,
                    "modification_type": tr.modification_type.value,
                    "classification": tr.classification.value,
                    "confidence": round(tr.confidence, 4),
                    "reasoning": tr.reasoning,
                    "is_modified_existing": tr.is_modified_existing,
                    "assertion_count": tr.assertion_count,
                    "misaligned_assertion_count": tr.misaligned_assertion_count,
                }
                for tr in self.f2p_test_reports
            ],
            "patch_hunk_reports": [
                {
                    "hunk_index": hr.hunk_index,
                    "file_path": hr.file_path,
                    "classification": hr.classification.value,
                    "confidence": round(hr.confidence, 4),
                    "reasoning": hr.reasoning,
                    "is_heuristic": hr.is_heuristic,
                }
                for hr in self.patch_hunk_reports
            ],
            "compound_patterns": self.compound_patterns,
            "evidence_summary": self.evidence_summary,
        }


# ── v2 Models ─────────────────────────────────────────────────────────


@dataclass
class IntentStatement:
    """Ground truth intent extracted from the problem statement (Stage 2 v2).

    The acceptance_criteria list is the key addition: explicit testable behaviors
    that the problem description asks for.  This becomes the reference for
    matching patches and tests against the described intent.
    """
    instance_id: str
    core_requirement: str              # What must change
    behavioral_contract: str           # How behavior should differ after fix
    acceptance_criteria: list[str]     # Specific verifiable claims from description
    out_of_scope: str                  # What is NOT asked for
    ambiguity_score: float             # 0-1
    raw_llm_response: str = ""


@dataclass
class ChangedBlock:
    """A source code block changed by the gold patch (Stage 3 v2)."""
    file_path: str
    block_name: str            # function/class name
    block_type: str            # "function", "class", "method", "statement"
    edit_status: str           # "INSERT", "DELETE", "UPDATE" (from astred_core)
    pre_source: str = ""       # source before patch
    post_source: str = ""      # source after patch


@dataclass
class AssertionDetail:
    """A single assertion extracted from a test function."""
    statement: str             # full assertion source line
    verdict: AssertionVerdict = AssertionVerdict.ON_TOPIC
    reason: str = ""


@dataclass
class TestBlock:
    """An F2P test function with extracted assertions (Stage 3 v2)."""
    test_id: str
    test_name: str
    file_path: str
    full_source: str
    assertions: list[AssertionDetail] = field(default_factory=list)
    called_functions: list[str] = field(default_factory=list)  # names of functions called


@dataclass
class StructuralDiff:
    """Structural analysis output from astred_core (Stage 3 v2)."""
    instance_id: str
    changed_blocks: list[ChangedBlock]       # Functions/classes changed by gold patch
    test_blocks: list[TestBlock]             # F2P test functions with assertions
    call_edges: list[tuple[str, str]]        # (test_function, changed_function) pairs
    astred_available: bool = True            # False if fell back to Python ast


# ── v2 Verdict Reports ───────────────────────────────────────────────


@dataclass
class HunkVerdict:
    """Intent-matching verdict for a single gold patch hunk (Stage 4A v2)."""
    hunk_index: int
    file_path: str
    verdict: PatchVerdict
    confidence: float
    reasoning: str
    is_heuristic: bool = False


@dataclass
class AssertionVerdictReport:
    """Intent-matching verdict for a single assertion within an F2P test."""
    statement: str
    verdict: AssertionVerdict
    reason: str = ""


@dataclass
class TestVerdictReport:
    """Intent-matching verdict for a single F2P test (Stage 4B v2)."""
    test_id: str
    test_name: str
    intent_match: TestVerdict
    confidence: float
    reasoning: str
    is_modified: bool                        # Was test pre-existing and modified?
    modification_aligned: bool = True        # If modified, is the modification aligned?
    assertion_verdicts: list[AssertionVerdictReport] = field(default_factory=list)

    @property
    def on_topic_count(self) -> int:
        return sum(1 for a in self.assertion_verdicts if a.verdict == AssertionVerdict.ON_TOPIC)

    @property
    def off_topic_count(self) -> int:
        return sum(1 for a in self.assertion_verdicts if a.verdict == AssertionVerdict.OFF_TOPIC)


@dataclass
class ExcessPatchDetail:
    """Detailed EXCESS_PATCH scoring breakdown."""
    score: float
    total_hunks: int
    required_count: int
    ancillary_count: int
    unrelated_count: int
    hunk_verdicts: list[HunkVerdict] = field(default_factory=list)


@dataclass
class ExcessTestDetail:
    """Detailed EXCESS_TEST scoring breakdown."""
    score: float
    total_tests: int
    aligned_count: int
    tangential_count: int
    unrelated_count: int
    total_assertions: int
    on_topic_assertions: int
    off_topic_assertions: int
    has_modified_tests: bool
    test_verdicts: list[TestVerdictReport] = field(default_factory=list)


@dataclass
class VagueSpecDetail:
    """Detailed VAGUE_SPEC scoring breakdown."""
    score: float
    reasoning: str = ""


@dataclass
class VerdictScore:
    """Confidence score for a single v2 verdict category."""
    category: VerdictCategory
    confidence: float
    evidence: list[str] = field(default_factory=list)


@dataclass
class ContaminationReportV2:
    """v2 contamination report with intent-matching verdicts."""
    instance_id: str
    severity: Severity
    combined_score: float
    intent: IntentStatement
    excess_patch: ExcessPatchDetail
    excess_test: ExcessTestDetail
    vague_spec: VagueSpecDetail
    categories: dict[str, VerdictScore] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "instance_id": self.instance_id,
            "severity": self.severity.value,
            "combined_score": round(self.combined_score, 4),
            "intent": {
                "core_requirement": self.intent.core_requirement,
                "behavioral_contract": self.intent.behavioral_contract,
                "acceptance_criteria": self.intent.acceptance_criteria,
                "out_of_scope": self.intent.out_of_scope,
                "ambiguity_score": round(self.intent.ambiguity_score, 4),
            },
            "excess_patch": {
                "score": round(self.excess_patch.score, 4),
                "total_hunks": self.excess_patch.total_hunks,
                "required": self.excess_patch.required_count,
                "ancillary": self.excess_patch.ancillary_count,
                "unrelated": self.excess_patch.unrelated_count,
                "hunks": [
                    {
                        "hunk_index": h.hunk_index,
                        "file": h.file_path,
                        "verdict": h.verdict.value,
                        "confidence": round(h.confidence, 4),
                        "reason": h.reasoning,
                    }
                    for h in self.excess_patch.hunk_verdicts
                ],
            },
            "excess_test": {
                "score": round(self.excess_test.score, 4),
                "total_tests": self.excess_test.total_tests,
                "aligned": self.excess_test.aligned_count,
                "tangential": self.excess_test.tangential_count,
                "unrelated": self.excess_test.unrelated_count,
                "total_assertions": self.excess_test.total_assertions,
                "on_topic": self.excess_test.on_topic_assertions,
                "off_topic": self.excess_test.off_topic_assertions,
                "has_modified_tests": self.excess_test.has_modified_tests,
                "tests": [
                    {
                        "test_id": t.test_id,
                        "test_name": t.test_name,
                        "intent_match": t.intent_match.value,
                        "is_modified": t.is_modified,
                        "modification_aligned": t.modification_aligned,
                        "assertions": [
                            {
                                "statement": a.statement,
                                "verdict": a.verdict.value,
                                "reason": a.reason,
                            }
                            for a in t.assertion_verdicts
                        ],
                    }
                    for t in self.excess_test.test_verdicts
                ],
            },
            "vague_spec": {
                "score": round(self.vague_spec.score, 4),
                "reasoning": self.vague_spec.reasoning,
            },
            "recommendations": self.recommendations,
        }


@dataclass
class PipelineConfig:
    """Configuration for the pipeline."""
    llm_base_url: str = "https://cloudgpt-openai.azure-api.net/"
    llm_api_version: str = "2025-04-01-preview"
    llm_model: str = "gpt-5.2-20251211"
    llm_max_tokens: int = 4096
    llm_reasoning_effort: str = "high"
    max_concurrent_requests: int = 10
    retry_attempts: int = 7
    retry_delay_seconds: float = 5.0
    concurrency: int = 5
    cache_dir: str = ".cache/llm_responses"
    output_dir: str = "output"
    clean_max: float = 0.2
    minor_max: float = 0.5
    moderate_max: float = 0.8
    astred_enabled: bool = False
    astred_binary_path: str = ""
    code_visitation_enabled: bool = True
    repo_cache_dir: str = ".cache/repos"
    clone_timeout_seconds: int = 120
    max_source_context_lines: int = 200
