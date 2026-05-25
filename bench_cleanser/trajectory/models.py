"""Data models for trajectory validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from bench_cleanser.models import AgentTrajectoryLabel


class LeakagePattern(str, Enum):
    """Classification of how an agent arrived at its solution."""
    GENUINE_SOLUTION = "GENUINE_SOLUTION"      # Derived from problem statement
    GOLD_PATCH_LEAK = "GOLD_PATCH_LEAK"        # Patch matches gold patch too closely
    PACKAGE_LEAK = "PACKAGE_LEAK"              # Solution installed from PyPI/package
    TEST_AWARE = "TEST_AWARE"                  # References F2P test names/values
    PARTIAL_MATCH = "PARTIAL_MATCH"            # Some leakage signals, inconclusive
    UNKNOWN = "UNKNOWN"                        # Not enough data to classify


class ActionType(str, Enum):
    """Types of actions in an agent trajectory."""
    EDIT = "EDIT"
    TERMINAL = "TERMINAL"
    BROWSE = "BROWSE"
    THINK = "THINK"
    SEARCH = "SEARCH"
    READ = "READ"
    WRITE = "WRITE"
    OTHER = "OTHER"


@dataclass
class TrajectoryAction:
    """A single action taken by an agent during its trajectory."""
    action_type: ActionType
    content: str
    file_path: str = ""
    timestamp: str = ""
    observation: str = ""
    role: str = ""
    tool_name: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrajectoryAction:
        action_type_str = data.get("action_type", data.get("type", "OTHER"))
        try:
            action_type = ActionType(action_type_str.upper())
        except ValueError:
            action_type = ActionType.OTHER
        return cls(
            action_type=action_type,
            content=data.get("content", data.get("command", "")),
            file_path=data.get("file_path", data.get("path", "")),
            timestamp=data.get("timestamp", ""),
            observation=data.get("observation", data.get("output", "")),
            role=data.get("role", ""),
            tool_name=data.get("tool_name", data.get("function", "")),
        )


@dataclass
class TrajectoryRecord:
    """Complete trajectory for a single agent on a single task."""
    instance_id: str
    agent_name: str
    actions: list[TrajectoryAction]
    final_patch: str = ""
    passed_tests: bool = False
    resolved: bool = False
    model_name: str = ""
    total_tokens: int = 0
    turn_count: int = 0
    raw_messages: list[dict] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrajectoryRecord:
        actions = [
            TrajectoryAction.from_dict(a)
            for a in data.get("actions", data.get("trajectory", []))
        ]
        return cls(
            instance_id=data.get("instance_id", ""),
            agent_name=data.get("agent_name", data.get("model_name_or_path", "")),
            actions=actions,
            final_patch=data.get("final_patch", data.get("model_patch", "")),
            passed_tests=data.get("passed_tests", data.get("resolved", False)),
            resolved=data.get("resolved", False),
            model_name=data.get("model_name", data.get("model_name_or_path", "")),
            total_tokens=data.get("total_tokens", data.get("token_count", 0)),
            turn_count=data.get("turn_count", data.get("num_turns", 0)),
            raw_messages=data.get("raw_messages", data.get("messages", [])),
        )


@dataclass
class TrajectoryAnalysis:
    """Analysis result for a single trajectory."""
    instance_id: str
    agent_name: str
    leakage_pattern: LeakagePattern
    evidence_strength: str = "moderate"
    evidence: list[str] = field(default_factory=list)
    gold_patch_similarity: float = 0.0          # 0-1, difflib ratio
    pip_install_commands: list[str] = field(default_factory=list)
    test_references: list[str] = field(default_factory=list)
    llm_reasoning: str = ""                     # LLM's detailed reasoning
    causal_chain: str = ""                      # What led the agent to its approach
    agent_behavior_summary: str = ""            # Brief characterization of agent behavior
    trajectory_label: AgentTrajectoryLabel | None = None
    # The agent's reported outcome on the F2P tests — propagated from the
    # source TrajectoryRecord. Stage 7 fusion needs it to distinguish
    # AMBIGUOUS_PASS (passed with UNKNOWN trajectory) from a failed-and-
    # uncharacterised attempt.
    resolved: bool = False

    @property
    def agent_trajectory_label(self) -> AgentTrajectoryLabel:
        """Return the trajectory label, mapping from LeakagePattern if needed."""
        if self.trajectory_label is not None:
            return self.trajectory_label
        _map = {
            LeakagePattern.GENUINE_SOLUTION: AgentTrajectoryLabel.AGENT_PASSED_GENUINE,
            LeakagePattern.GOLD_PATCH_LEAK: AgentTrajectoryLabel.AGENT_PASSED_LEAK,
            LeakagePattern.PACKAGE_LEAK: AgentTrajectoryLabel.AGENT_PASSED_PACKAGE_LEAK,
            LeakagePattern.TEST_AWARE: AgentTrajectoryLabel.AGENT_PASSED_TEST_AWARE,
            LeakagePattern.PARTIAL_MATCH: AgentTrajectoryLabel.AGENT_UNKNOWN,
            LeakagePattern.UNKNOWN: AgentTrajectoryLabel.AGENT_UNKNOWN,
        }
        return _map.get(self.leakage_pattern, AgentTrajectoryLabel.AGENT_UNKNOWN)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "agent_name": self.agent_name,
            "leakage_pattern": self.leakage_pattern.value,
            "trajectory_label": self.agent_trajectory_label.value,
            "evidence_strength": self.evidence_strength,
            "evidence": self.evidence,
            "gold_patch_similarity": round(self.gold_patch_similarity, 4),
            "pip_install_commands": self.pip_install_commands,
            "test_references": self.test_references,
            "llm_reasoning": self.llm_reasoning,
            "causal_chain": self.causal_chain,
            "agent_behavior_summary": self.agent_behavior_summary,
            "resolved": self.resolved,
        }
