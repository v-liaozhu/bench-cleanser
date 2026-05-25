"""Core data models for the bench-cleanser pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PatchVerdict(str, Enum):
    REQUIRED = "REQUIRED"
    ANCILLARY = "ANCILLARY"
    UNRELATED = "UNRELATED"


class TestVerdict(str, Enum):
    ALIGNED = "ALIGNED"
    TANGENTIAL = "TANGENTIAL"
    UNRELATED = "UNRELATED"


class AssertionVerdict(str, Enum):
    ON_TOPIC = "ON_TOPIC"
    OFF_TOPIC = "OFF_TOPIC"


class Severity(str, Enum):
    CLEAN = "CLEAN"
    MINOR = "MINOR"
    MODERATE = "MODERATE"
    SEVERE = "SEVERE"


class TaskContaminationLabel(str, Enum):
    """Axis 1: task-level contamination labels (7 binary labels).

    Labels 1-6 are contamination signals (multi-label, co-occur freely).
    CLEAN is exclusive — cannot co-occur with any other label.

    Terminology aligned with OpenAI's SWE-bench Verified audit (2026):
      - APPROACH_LOCK = "Narrow test cases"
      - OVER_TEST     = "Wide test cases"
    """
    APPROACH_LOCK = "approach_lock"
    OVER_TEST = "over_test"
    OVER_PATCH = "over_patch"
    UNCLEAR_DESCRIPTION = "unclear_description"
    HIDDEN_CONTEXT = "hidden_context"
    WEAK_COVERAGE = "weak_coverage"
    CLEAN = "clean"


class AgentTrajectoryLabel(str, Enum):
    """Axis 2: per-agent-task trajectory classification."""
    AGENT_PASSED_GENUINE = "agent_passed_genuine"
    AGENT_PASSED_LEAK = "agent_passed_leak"
    AGENT_PASSED_PACKAGE_LEAK = "agent_passed_package_leak"
    AGENT_PASSED_TEST_AWARE = "agent_passed_test_aware"
    AGENT_PASSED_TRAINED_HACK = "agent_passed_trained_hack"
    AGENT_FAILED_COMPLETED_INTENT = "agent_failed_completed_intent"
    AGENT_FAILED_NO_INTENT = "agent_failed_no_intent"
    AGENT_UNKNOWN = "agent_unknown"


class TestModificationType(str, Enum):
    NEW = "NEW"
    MODIFIED = "MODIFIED"
    UNKNOWN = "UNKNOWN"


@dataclass
class TaskRecord:
    """Raw SWE-bench task record."""
    instance_id: str
    repo: str
    base_commit: str
    patch: str
    test_patch: str
    problem_statement: str
    hints_text: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    version: str
    environment_setup_commit: str = ""
    created_at: str = ""
    requirements: str = ""
    interface: str = ""
    before_repo_set_cmd: str = ""

    @property
    def full_problem_context(self) -> str:
        """Return the complete task specification including requirements and interface.

        For SWE-bench Pro, the problem_statement is narrow; the full context
        includes separate requirements and interface fields. For SWE-bench
        Verified, requirements and interface are empty so this returns just
        the problem_statement.
        """
        parts = [self.problem_statement]
        if self.requirements:
            parts.append(f"\nRequirements:\n{self.requirements}")
        if self.interface:
            parts.append(f"\nInterface:\n{self.interface}")
        return "\n".join(parts)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskRecord:
        f2p = data.get("FAIL_TO_PASS") or data.get("fail_to_pass", "[]")
        p2p = data.get("PASS_TO_PASS") or data.get("pass_to_pass", "[]")
        if isinstance(f2p, str):
            try:
                f2p = json.loads(f2p)
            except json.JSONDecodeError:
                import ast as _ast
                f2p = _ast.literal_eval(f2p)
        if isinstance(p2p, str):
            try:
                p2p = json.loads(p2p)
            except json.JSONDecodeError:
                import ast as _ast
                p2p = _ast.literal_eval(p2p)
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
            requirements=data.get("requirements", ""),
            interface=data.get("interface", ""),
            before_repo_set_cmd=data.get("before_repo_set_cmd", ""),
        )


@dataclass
class PatchHunk:
    """A single hunk from a unified diff."""
    file_path: str
    hunk_index: int
    header: str
    added_lines: list[str]
    removed_lines: list[str]
    context_lines: list[str]
    function_context: str
    raw_diff: str

    @property
    def is_test_file(self) -> bool:
        return "test" in self.file_path.lower()

    @property
    def is_init_file(self) -> bool:
        return self.file_path.endswith("__init__.py")

    @property
    def is_doc_file(self) -> bool:
        lower = self.file_path.lower()
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
    name: str
    module: str
    file_path: str
    line_number: int
    is_in_patch: bool


@dataclass
class Assertion:
    """A structured assertion extracted from a test via AST."""
    statement: str
    assertion_type: str
    target_expression: str
    expected_value: str


@dataclass
class TestedFunction:
    """A source function that a test exercises."""
    name: str
    file_path: str
    source: str
    is_modified_by_patch: bool


@dataclass
class CodeContext:
    """Full code context retrieved via code visitation (repo clone)."""
    pre_patch_test_source: str
    post_patch_test_source: str
    test_file_imports: str
    test_file_fixtures: str
    tested_functions: list[TestedFunction]
    call_targets: list[CallTarget]
    assertions: list[Assertion]
    test_file_path: str
    repo_path: str


@dataclass
class TestHunk:
    """A test function diff extracted from the test_patch."""
    file_path: str
    test_name: str
    full_test_id: str
    modification_type: TestModificationType
    added_lines: list[str]
    removed_lines: list[str]
    full_source: str
    raw_diff: str
    code_context: CodeContext | None = None


@dataclass
class ProblemCodeContext:
    """Pre-patch source code context for grounding Stage 2 intent extraction.

    Contains source code from BEFORE the patch, so the LLM understands
    what code the problem references without seeing the fix.
    """
    mentioned_file_contents: dict[str, str] = field(default_factory=dict)
    relevant_directory_tree: str = ""
    mentioned_entity_sources: dict[str, str] = field(default_factory=dict)


@dataclass
class ParsedTask:
    """Fully parsed SWE-bench task, ready for analysis."""
    record: TaskRecord
    patch_hunks: list[PatchHunk]
    test_hunks: list[TestHunk]
    f2p_test_hunks: list[TestHunk]
    f2p_tests_with_no_hunk: list[str]
    files_in_gold_patch: list[str]
    files_in_test_patch: list[str]
    problem_code_context: ProblemCodeContext | None = None


@dataclass
class ProblemDecomposition:
    """Structured decomposition of the problem statement.

    Separates the problem into its component parts so downstream stages
    can distinguish what the reporter actually asked for vs. what they
    suggested as a fix approach.
    """
    bug_description: str
    suggested_fix: str
    legitimacy: str
    mentioned_files: list[str] = field(default_factory=list)
    mentioned_functions: list[str] = field(default_factory=list)
    mentioned_classes: list[str] = field(default_factory=list)
    mentioned_variables: list[str] = field(default_factory=list)
    mentioned_modules: list[str] = field(default_factory=list)


@dataclass
class IntentStatement:
    """Intent extracted from the problem statement (blind to gold patch).

    The acceptance_criteria list is key: explicit testable behaviors the
    problem asks for — the reference for matching patches and tests.
    """
    instance_id: str
    core_requirement: str
    behavioral_contract: str
    acceptance_criteria: list[str]
    out_of_scope: str
    ambiguity_score: float
    raw_llm_response: str = ""
    decomposition: ProblemDecomposition | None = None


@dataclass
class ChangedBlock:
    """A source code block changed by the gold patch."""
    file_path: str
    block_name: str
    block_type: str
    edit_status: str
    pre_source: str = ""
    post_source: str = ""


@dataclass
class AssertionDetail:
    """A single assertion extracted from a test function."""
    statement: str
    verdict: AssertionVerdict = AssertionVerdict.ON_TOPIC
    reason: str = ""


@dataclass
class TestBlock:
    """An F2P test function with extracted assertions."""
    test_id: str
    test_name: str
    file_path: str
    full_source: str
    assertions: list[AssertionDetail] = field(default_factory=list)
    called_functions: list[str] = field(default_factory=list)


@dataclass
class StructuralDiff:
    """Structural analysis output."""
    instance_id: str
    changed_blocks: list[ChangedBlock]
    test_blocks: list[TestBlock]
    call_edges: list[tuple[str, str]]
    astred_available: bool = True


@dataclass
class HunkVerdict:
    """Intent-matching verdict for a single gold patch hunk."""
    hunk_index: int
    file_path: str
    verdict: PatchVerdict
    evidence_strength: str = "moderate"
    reasoning: str = ""
    is_heuristic: bool = False


@dataclass
class AssertionVerdictReport:
    """Intent-matching verdict for a single assertion."""
    statement: str
    verdict: AssertionVerdict
    reason: str = ""


@dataclass
class TestVerdictReport:
    """Intent-matching verdict for a single F2P test."""
    test_id: str
    test_name: str
    intent_match: TestVerdict
    is_modified: bool = False
    evidence_strength: str = "moderate"
    reasoning: str = ""
    modification_aligned: bool = True
    assertion_verdicts: list[AssertionVerdictReport] = field(default_factory=list)

    @property
    def on_topic_count(self) -> int:
        return sum(1 for a in self.assertion_verdicts if a.verdict == AssertionVerdict.ON_TOPIC)

    @property
    def off_topic_count(self) -> int:
        return sum(1 for a in self.assertion_verdicts if a.verdict == AssertionVerdict.OFF_TOPIC)


@dataclass
class PatchAnalysis:
    """Patch analysis results: per-hunk verdicts."""
    total_hunks: int
    required_count: int
    ancillary_count: int
    unrelated_count: int
    hunk_verdicts: list[HunkVerdict] = field(default_factory=list)


@dataclass
class TestAnalysis:
    """Test analysis results: per-test verdicts."""
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
class DescriptionClarity:
    """Spec ambiguity analysis result."""
    score: float
    reasoning: str = ""


@dataclass
class TaskLabelAssignment:
    """A single Axis 1 label assigned to a task with evidence."""
    label: TaskContaminationLabel
    evidence: list[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class AgentLabelAssignment:
    """A single Axis 2 label assigned to an agent-task pair with evidence."""
    label: AgentTrajectoryLabel
    evidence: list[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class ContaminationReport:
    """Final contamination report for a single task."""
    instance_id: str
    severity: Severity
    intent: IntentStatement
    patch_analysis: PatchAnalysis
    test_analysis: TestAnalysis
    description_clarity: DescriptionClarity
    task_labels: list[TaskLabelAssignment] = field(default_factory=list)
    agent_labels: dict[str, AgentLabelAssignment] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    # Non-empty when this report represents a pipeline failure (LLM hang,
    # repo clone failure, schema-validation exhaustion, etc.) rather than
    # a real classification. Downstream consumers MUST exclude these rows
    # from aggregate statistics and from Stage-7 fusion — they carry no
    # analytic signal. Severity is forced to CLEAN in this case so the
    # documented "severity is set membership over task_labels" invariant
    # continues to hold for any row a reviewer recomputes from disk.
    pipeline_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "severity": self.severity.value,
            "intent": {
                "core_requirement": self.intent.core_requirement,
                "behavioral_contract": self.intent.behavioral_contract,
                "acceptance_criteria": self.intent.acceptance_criteria,
                "out_of_scope": self.intent.out_of_scope,
                "ambiguity_score": round(self.intent.ambiguity_score, 4),
                **({"decomposition": {
                    "bug_description": self.intent.decomposition.bug_description,
                    "suggested_fix": self.intent.decomposition.suggested_fix,
                    "legitimacy": self.intent.decomposition.legitimacy,
                    "mentioned_files": self.intent.decomposition.mentioned_files,
                    "mentioned_functions": self.intent.decomposition.mentioned_functions,
                    "mentioned_classes": self.intent.decomposition.mentioned_classes,
                    "mentioned_variables": self.intent.decomposition.mentioned_variables,
                    "mentioned_modules": self.intent.decomposition.mentioned_modules,
                }} if self.intent.decomposition else {}),
            },
            "patch_analysis": {
                "total_hunks": self.patch_analysis.total_hunks,
                "required_count": self.patch_analysis.required_count,
                "ancillary_count": self.patch_analysis.ancillary_count,
                "unrelated_count": self.patch_analysis.unrelated_count,
                "hunks": [
                    {
                        "hunk_index": h.hunk_index,
                        "file_path": h.file_path,
                        "verdict": h.verdict.value,
                        "evidence_strength": h.evidence_strength,
                        "reasoning": h.reasoning,
                    }
                    for h in self.patch_analysis.hunk_verdicts
                ],
            },
            "test_analysis": {
                "total_tests": self.test_analysis.total_tests,
                "aligned_count": self.test_analysis.aligned_count,
                "tangential_count": self.test_analysis.tangential_count,
                "unrelated_count": self.test_analysis.unrelated_count,
                "total_assertions": self.test_analysis.total_assertions,
                "on_topic_assertions": self.test_analysis.on_topic_assertions,
                "off_topic_assertions": self.test_analysis.off_topic_assertions,
                "has_modified_tests": self.test_analysis.has_modified_tests,
                "tests": [
                    {
                        "test_id": t.test_id,
                        "test_name": t.test_name,
                        "intent_match": t.intent_match.value,
                        "evidence_strength": t.evidence_strength,
                        "reasoning": t.reasoning,
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
                    for t in self.test_analysis.test_verdicts
                ],
            },
            "description_clarity": {
                "score": round(self.description_clarity.score, 4),
                "reasoning": self.description_clarity.reasoning,
            },
            "task_labels": [
                {
                    "label": tl.label.value,
                    "evidence": tl.evidence,
                    "reasoning": tl.reasoning,
                }
                for tl in self.task_labels
            ],
            "agent_labels": {
                agent: {
                    "label": al.label.value,
                    "evidence": al.evidence,
                    "reasoning": al.reasoning,
                }
                for agent, al in self.agent_labels.items()
            },
            "recommendations": self.recommendations,
            **({"pipeline_error": self.pipeline_error} if self.pipeline_error else {}),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContaminationReport:
        intent_d = data.get("intent", {})
        decomp_d = intent_d.get("decomposition")
        decomposition = None
        if decomp_d:
            decomposition = ProblemDecomposition(
                bug_description=decomp_d.get("bug_description", ""),
                suggested_fix=decomp_d.get("suggested_fix", ""),
                legitimacy=decomp_d.get("legitimacy", "unclear"),
                mentioned_files=decomp_d.get("mentioned_files", []),
                mentioned_functions=decomp_d.get("mentioned_functions", []),
                mentioned_classes=decomp_d.get("mentioned_classes", []),
                mentioned_variables=decomp_d.get("mentioned_variables", []),
                mentioned_modules=decomp_d.get("mentioned_modules", []),
            )
        intent = IntentStatement(
            instance_id=data["instance_id"],
            core_requirement=intent_d.get("core_requirement", ""),
            behavioral_contract=intent_d.get("behavioral_contract", ""),
            acceptance_criteria=intent_d.get("acceptance_criteria", []),
            out_of_scope=intent_d.get("out_of_scope", ""),
            ambiguity_score=intent_d.get("ambiguity_score", 0.0),
            decomposition=decomposition,
        )

        ep_d = data.get("patch_analysis", {})
        hunk_verdicts = [
            HunkVerdict(
                hunk_index=h.get("hunk_index", 0),
                file_path=h.get("file_path", ""),
                verdict=PatchVerdict(h.get("verdict", "REQUIRED")),
                evidence_strength=h.get("evidence_strength", "moderate"),
                reasoning=h.get("reasoning", ""),
            )
            for h in ep_d.get("hunks", [])
        ]
        patch_analysis = PatchAnalysis(
            total_hunks=ep_d.get("total_hunks", 0),
            required_count=ep_d.get("required_count", 0),
            ancillary_count=ep_d.get("ancillary_count", 0),
            unrelated_count=ep_d.get("unrelated_count", 0),
            hunk_verdicts=hunk_verdicts,
        )

        et_d = data.get("test_analysis", {})
        test_verdicts = []
        for t in et_d.get("tests", []):
            assertion_verdicts = [
                AssertionVerdictReport(
                    statement=a.get("statement", ""),
                    verdict=AssertionVerdict(a.get("verdict", "ON_TOPIC")),
                    reason=a.get("reason", ""),
                )
                for a in t.get("assertions", [])
            ]
            test_verdicts.append(TestVerdictReport(
                test_id=t.get("test_id", ""),
                test_name=t.get("test_name", ""),
                intent_match=TestVerdict(t.get("intent_match", "ALIGNED")),
                evidence_strength=t.get("evidence_strength", "moderate"),
                reasoning=t.get("reasoning", ""),
                is_modified=t.get("is_modified", False),
                modification_aligned=t.get("modification_aligned", True),
                assertion_verdicts=assertion_verdicts,
            ))
        test_analysis = TestAnalysis(
            total_tests=et_d.get("total_tests", 0),
            aligned_count=et_d.get("aligned_count", 0),
            tangential_count=et_d.get("tangential_count", 0),
            unrelated_count=et_d.get("unrelated_count", 0),
            total_assertions=et_d.get("total_assertions", 0),
            on_topic_assertions=et_d.get("on_topic_assertions", 0),
            off_topic_assertions=et_d.get("off_topic_assertions", 0),
            has_modified_tests=et_d.get("has_modified_tests", False),
            test_verdicts=test_verdicts,
        )

        dc_d = data.get("description_clarity", {})
        description_clarity = DescriptionClarity(
            score=dc_d.get("score", 0.0),
            reasoning=dc_d.get("reasoning", ""),
        )

        task_labels = []
        for tl_d in data.get("task_labels", []):
            try:
                task_labels.append(TaskLabelAssignment(
                    label=TaskContaminationLabel(tl_d.get("label", "clean")),
                    evidence=tl_d.get("evidence", []),
                    reasoning=tl_d.get("reasoning", ""),
                ))
            except ValueError:
                pass

        return cls(
            instance_id=data["instance_id"],
            severity=Severity(data.get("severity", "CLEAN")),
            intent=intent,
            patch_analysis=patch_analysis,
            test_analysis=test_analysis,
            description_clarity=description_clarity,
            task_labels=task_labels,
            recommendations=data.get("recommendations", []),
            pipeline_error=data.get("pipeline_error"),
        )


@dataclass
class PipelineConfig:
    llm_base_url: str = "https://cloudgpt-openai.azure-api.net/"
    llm_api_version: str = "2025-04-01-preview"
    llm_model: str = "gpt-5.4-20260305"
    llm_max_tokens: int = 65536
    llm_reasoning_effort: str = "high"
    max_concurrent_requests: int = 10
    retry_attempts: int = 7
    retry_delay_seconds: float = 5.0
    concurrency: int = 5
    cache_dir: str = ".cache/llm_responses"
    output_dir: str = "output"
    repo_cache_dir: str = ".cache/repos"
    clone_timeout_seconds: int = 120
    max_source_context_lines: int = 200
