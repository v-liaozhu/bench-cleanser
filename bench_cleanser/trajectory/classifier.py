"""Trajectory classifier for detecting benchmark leakage.

Primary approach: LLM analysis of the full trajectory to understand
agent behavior and identify leakage patterns.

Heuristic signals (patch similarity, pip installs, test references)
are computed first and fed to the LLM as supporting evidence, but
the LLM makes the final classification decision.

Tier 1: Heuristic signal extraction (fast, no LLM)
  - Gold patch similarity
  - pip install commands
  - F2P test name/value references

Tier 2: LLM analysis (primary classifier) — structured Pydantic output
  via :meth:`LLMClient.query_structured`. No ad-hoc JSON parsing.

Tier 3: Cross-agent comparison
  - If all agents produce identical patches, likely leakage
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from typing import Any

from bench_cleanser.models import AgentTrajectoryLabel
from bench_cleanser.prompts import load as _load_prompt
from bench_cleanser.schemas import TrajectoryClassificationResponse
from bench_cleanser.trajectory.models import (
    ActionType,
    LeakagePattern,
    TrajectoryAction,
    TrajectoryAnalysis,
    TrajectoryRecord,
)

logger = logging.getLogger(__name__)

GOLD_PATCH_SIMILARITY_THRESHOLD = 0.90
HIGH_SIMILARITY_THRESHOLD = 0.80
PIP_INSTALL_RE = re.compile(
    r"pip\s+install\s+(?:--upgrade\s+)?([a-zA-Z0-9_.-]+)",
    re.IGNORECASE,
)



def compute_patch_similarity(patch_a: str, patch_b: str) -> float:
    """Compute similarity ratio between two patches using difflib.

    Returns a value between 0.0 (completely different) and 1.0 (identical).

    Improved normalization:
    - Strips comments and blank lines
    - Normalizes whitespace
    - Focuses on added/removed lines (skips diff context)
    """
    if not patch_a or not patch_b:
        return 0.0

    def _normalize_patch_lines(patch: str) -> list[str]:
        """Extract and normalize only the changed lines from a patch."""
        result = []
        for raw_line in patch.strip().splitlines():
            line = raw_line.rstrip()
            if line.startswith(("diff ", "index ", "---", "+++", "@@")):
                continue
            if line.startswith("+") or line.startswith("-"):
                content = line[1:]
                # Strip trailing `#`/`//` comments before comparing so a
                # commented-out tweak doesn't perturb similarity. Skip the
                # strip when an unbalanced quote count signals the `#` is
                # likely inside a string literal — otherwise we'd mangle
                # `pattern = "match #1"` into `pattern = "match`.
                if content.count('"') % 2 == 0 and content.count("'") % 2 == 0:
                    content = re.sub(r'\s*#\s.*$', '', content)
                    content = re.sub(r'\s*//\s.*$', '', content)
                content = ' '.join(content.split())
                if content:
                    result.append(content)
        return result

    lines_a = _normalize_patch_lines(patch_a)
    lines_b = _normalize_patch_lines(patch_b)

    if not lines_a or not lines_b:
        return 0.0

    matcher = difflib.SequenceMatcher(None, lines_a, lines_b)
    return matcher.ratio()


def detect_pip_installs(trajectory: TrajectoryRecord) -> list[str]:
    """Find pip install commands in the trajectory."""
    installs = []
    for action in trajectory.actions:
        if action.action_type in (ActionType.TERMINAL, ActionType.OTHER):
            for match in PIP_INSTALL_RE.finditer(action.content):
                installs.append(match.group(0))
    return installs


def detect_test_references(
    trajectory: TrajectoryRecord,
    f2p_test_names: list[str],
) -> list[str]:
    """Check if trajectory actions reference F2P test names or values."""
    references = []
    for action in trajectory.actions:
        content = action.content
        for test_name in f2p_test_names:
            # Check for test function name (strip module path)
            short_name = test_name.rsplit("::", 1)[-1] if "::" in test_name else test_name
            if short_name in content:
                ref = f"References test '{short_name}' in {action.action_type.value} action"
                if ref not in references:
                    references.append(ref)
    return references


def extract_heuristic_signals(
    trajectory: TrajectoryRecord,
    gold_patch: str,
    f2p_test_names: list[str],
) -> dict[str, Any]:
    """Extract all heuristic signals for a trajectory.

    Returns a dict of signals to be fed to the LLM for analysis.
    """
    similarity = compute_patch_similarity(trajectory.final_patch, gold_patch)
    pip_installs = detect_pip_installs(trajectory)
    test_refs = detect_test_references(trajectory, f2p_test_names)

    return {
        "gold_patch_similarity": similarity,
        "pip_install_commands": pip_installs,
        "test_references": test_refs,
        "has_high_similarity": similarity >= HIGH_SIMILARITY_THRESHOLD,
        "has_gold_patch_match": similarity >= GOLD_PATCH_SIMILARITY_THRESHOLD,
        "has_pip_installs": len(pip_installs) > 0,
        "has_test_refs": len(test_refs) > 0,
    }


def classify_heuristic_only(
    trajectory: TrajectoryRecord,
    gold_patch: str,
    f2p_test_names: list[str],
) -> TrajectoryAnalysis:
    """Fallback: heuristic-only classification when LLM is unavailable.

    Used as a fallback when LLM analysis fails or is disabled.
    """
    signals = extract_heuristic_signals(trajectory, gold_patch, f2p_test_names)
    evidence = []
    similarity = signals["gold_patch_similarity"]
    pip_installs = signals["pip_install_commands"]
    test_refs = signals["test_references"]

    if signals["has_gold_patch_match"]:
        evidence.append(
            f"Final patch is {similarity:.1%} similar to gold patch "
            f"(threshold: {GOLD_PATCH_SIMILARITY_THRESHOLD:.0%})"
        )
        return TrajectoryAnalysis(
            instance_id=trajectory.instance_id,
            agent_name=trajectory.agent_name,
            leakage_pattern=LeakagePattern.GOLD_PATCH_LEAK,
            evidence_strength="strong",
            evidence=evidence,
            gold_patch_similarity=similarity,
            pip_install_commands=pip_installs,
            test_references=test_refs,
            resolved=trajectory.resolved,
        )

    if pip_installs:
        evidence.append(
            f"Found {len(pip_installs)} pip install command(s): "
            + "; ".join(pip_installs)
        )
        return TrajectoryAnalysis(
            instance_id=trajectory.instance_id,
            agent_name=trajectory.agent_name,
            leakage_pattern=LeakagePattern.PACKAGE_LEAK,
            evidence_strength="moderate",
            evidence=evidence,
            gold_patch_similarity=similarity,
            pip_install_commands=pip_installs,
            test_references=test_refs,
            resolved=trajectory.resolved,
        )

    if test_refs:
        evidence.extend(test_refs)
        return TrajectoryAnalysis(
            instance_id=trajectory.instance_id,
            agent_name=trajectory.agent_name,
            leakage_pattern=LeakagePattern.TEST_AWARE,
            evidence_strength="moderate",
            evidence=evidence,
            gold_patch_similarity=similarity,
            pip_install_commands=pip_installs,
            test_references=test_refs,
            resolved=trajectory.resolved,
        )

    if signals["has_high_similarity"]:
        evidence.append(
            f"Final patch has high similarity to gold patch ({similarity:.1%}) "
            f"but below leak threshold"
        )
        return TrajectoryAnalysis(
            instance_id=trajectory.instance_id,
            agent_name=trajectory.agent_name,
            leakage_pattern=LeakagePattern.PARTIAL_MATCH,
            evidence_strength="moderate",
            evidence=evidence,
            gold_patch_similarity=similarity,
            pip_install_commands=pip_installs,
            test_references=test_refs,
            resolved=trajectory.resolved,
        )

    return TrajectoryAnalysis(
        instance_id=trajectory.instance_id,
        agent_name=trajectory.agent_name,
        leakage_pattern=LeakagePattern.GENUINE_SOLUTION,
        evidence_strength="moderate",
        evidence=["No deterministic leakage signals detected"],
        gold_patch_similarity=similarity,
        pip_install_commands=pip_installs,
        test_references=test_refs,
        resolved=trajectory.resolved,
    )



TRAJECTORY_ANALYSIS_SYSTEM_PROMPT = _load_prompt("trajectory_analysis")


def _build_user_prompt(
    trajectory: TrajectoryRecord,
    gold_patch: str,
    problem_statement: str,
    f2p_test_names: list[str],
    heuristic_signals: dict[str, Any],
    contamination_context: str = "",
) -> str:
    """Render the per-call user message for trajectory analysis.

    The system prompt (`trajectory_analysis.md`) defines the role, the
    taxonomy, and the schema. This function only assembles task-specific
    data — problem statement, gold patch, heuristic signals, action trace.
    """
    action_summary = _summarize_actions(trajectory.actions)
    signals_section = json.dumps(heuristic_signals, indent=2)
    contamination_section = (
        f"\nCONTAMINATION CONTEXT:\n{contamination_context}\n"
        if contamination_context else ""
    )

    return f"""Analyze this AI agent's trajectory on a software engineering task.

PROBLEM STATEMENT:
{problem_statement[:30000]}

GOLD PATCH (the correct solution — the agent should NOT have access to this):
{gold_patch[:50000]}

FAIL-TO-PASS TEST NAMES (used for evaluation):
{json.dumps(f2p_test_names[:100])}
{contamination_section}
HEURISTIC SIGNALS (pre-computed):
{signals_section}

AGENT: {trajectory.agent_name}
RESOLVED: {trajectory.resolved}

AGENT'S TRAJECTORY:
{action_summary}

AGENT'S FINAL PATCH:
{trajectory.final_patch[:50000]}
"""


async def classify_with_llm(
    trajectory: TrajectoryRecord,
    gold_patch: str,
    problem_statement: str,
    f2p_test_names: list[str],
    llm: Any,
    heuristic_signals: dict[str, Any] | None = None,
    contamination_context: str = "",
) -> TrajectoryAnalysis:
    """Tier 2: LLM-based trajectory classification (primary approach).

    Sends the full trajectory context to the LLM and validates the response
    against :class:`TrajectoryClassificationResponse`. Falls back to the
    heuristic classifier on any error (network, validation, etc.).
    """
    if heuristic_signals is None:
        heuristic_signals = extract_heuristic_signals(
            trajectory, gold_patch, f2p_test_names
        )

    user_prompt = _build_user_prompt(
        trajectory, gold_patch, problem_statement, f2p_test_names,
        heuristic_signals, contamination_context,
    )

    try:
        result: TrajectoryClassificationResponse = await llm.query_structured(
            TRAJECTORY_ANALYSIS_SYSTEM_PROMPT,
            user_prompt,
            TrajectoryClassificationResponse,
        )

        try:
            pattern = LeakagePattern(result.pattern)
        except ValueError:
            pattern = LeakagePattern.UNKNOWN

        try:
            trajectory_label = AgentTrajectoryLabel(result.trajectory_label)
        except ValueError:
            trajectory_label = None

        evidence: list[str] = []
        if result.reasoning:
            evidence.append(f"LLM analysis: {result.reasoning}")
        if result.causal_chain:
            evidence.append(f"Causal chain: {result.causal_chain}")
        evidence.extend(result.key_evidence)

        return TrajectoryAnalysis(
            instance_id=trajectory.instance_id,
            agent_name=trajectory.agent_name,
            leakage_pattern=pattern,
            evidence_strength=result.evidence_strength,
            evidence=evidence,
            gold_patch_similarity=heuristic_signals["gold_patch_similarity"],
            pip_install_commands=heuristic_signals["pip_install_commands"],
            test_references=heuristic_signals["test_references"],
            llm_reasoning=result.reasoning,
            causal_chain=result.causal_chain,
            agent_behavior_summary=result.agent_behavior_summary,
            trajectory_label=trajectory_label,
            resolved=trajectory.resolved,
        )
    except Exception as exc:
        logger.warning(
            "LLM trajectory analysis failed for %s/%s: %s — falling back to heuristics",
            trajectory.instance_id, trajectory.agent_name, exc,
        )
        return classify_heuristic_only(
            trajectory, gold_patch, f2p_test_names,
        )



# Cross-agent upgrade rule constants.
#
# CROSS_AGENT_QUORUM_THRESHOLD: median pairwise patch similarity above which
# we consider the cluster "converged" enough to suspect gold-patch leakage.
# Using the median (not all-pairs) means one outlier agent — common in
# multi-model comparisons — doesn't suppress an otherwise clear signal.
#
# LOW_ENTROPY_PATCH_LINES: a gate that prevents the upgrade firing on
# trivial problems where independent agents legitimately converge. A
# one-line bug fix has effectively one correct answer; agreeing on it
# is not evidence of leakage. The threshold counts non-empty added
# lines in the gold patch (proxy for problem entropy).
CROSS_AGENT_QUORUM_THRESHOLD = 0.85
LOW_ENTROPY_PATCH_LINES = 10


def _count_gold_patch_added_lines(patches: list[str]) -> int:
    """Estimate problem entropy as the median count of non-empty added lines.

    Uses the median (not min) so a single outlier agent with a trivial or
    failed patch doesn't drag the entropy estimate to zero and disable the
    cross-agent upgrade entirely. The median across the agent cluster
    reflects what the *problem itself* admits as a typical answer.
    """
    if not patches:
        return 0
    counts: list[int] = []
    for patch in patches:
        added = 0
        for raw in patch.splitlines():
            line = raw.rstrip()
            if line.startswith("+") and not line.startswith("+++"):
                if line[1:].strip():
                    added += 1
        counts.append(added)
    counts.sort()
    mid = len(counts) // 2
    if len(counts) % 2 == 1:
        return counts[mid]
    return (counts[mid - 1] + counts[mid]) // 2


def classify_cross_agent(
    analyses: list[TrajectoryAnalysis],
    trajectories: list[TrajectoryRecord],
) -> list[TrajectoryAnalysis]:
    """Tier 3: Cross-agent comparison.

    Upgrades GENUINE_SOLUTION / PARTIAL_MATCH to GOLD_PATCH_LEAK when
    multiple agents produced patches that cluster around the gold answer.

    Rule (replaces the previous all-pairs short-circuit):

      1. Compute pairwise similarity for every (i, j) with i < j.
      2. If the *median* pairwise similarity ≥ CROSS_AGENT_QUORUM_THRESHOLD,
         the cluster has quorum. Median (not min) means one diverging
         agent doesn't veto the signal.
      3. Gate the upgrade on patch entropy: skip if the smallest patch
         is < LOW_ENTROPY_PATCH_LINES added lines. Low-entropy problems
         admit one obvious answer; convergence there is not leakage.
    """
    if len(analyses) < 2:
        return analyses

    patches = [t.final_patch for t in trajectories if t.final_patch]
    if len(patches) < 2:
        return analyses

    sims: list[float] = []
    for i in range(len(patches)):
        for j in range(i + 1, len(patches)):
            sims.append(compute_patch_similarity(patches[i], patches[j]))

    if not sims:
        return analyses

    sims.sort()
    mid = len(sims) // 2
    median_sim = sims[mid] if len(sims) % 2 == 1 else (sims[mid - 1] + sims[mid]) / 2

    if median_sim < CROSS_AGENT_QUORUM_THRESHOLD:
        return analyses

    min_added = _count_gold_patch_added_lines(patches)
    if min_added < LOW_ENTROPY_PATCH_LINES:
        logger.debug(
            "Cross-agent quorum met (median sim %.2f) but skipping upgrade: "
            "median patch has only %d added lines (low-entropy convergence)",
            median_sim, min_added,
        )
        return analyses

    for analysis in analyses:
        if analysis.leakage_pattern in (
            LeakagePattern.GENUINE_SOLUTION,
            LeakagePattern.PARTIAL_MATCH,
        ):
            analysis.evidence.append(
                f"Cross-agent: {len(patches)} agents converged "
                f"(median pairwise similarity {median_sim:.2f}, "
                f"median patch {min_added} added lines), "
                "suggesting gold-patch leakage rather than independent derivation"
            )
            analysis.leakage_pattern = LeakagePattern.GOLD_PATCH_LEAK
            analysis.evidence_strength = "strong"

    return analyses



def _summarize_actions(
    actions: list[TrajectoryAction], max_chars: int = 500000
) -> str:
    """Summarize trajectory actions, using large context window."""
    parts = []
    total = 0
    for i, action in enumerate(actions):
        # Use generous per-action limit for LLM analysis
        content_limit = min(len(action.content), 50000)
        line = f"[Step {i}] {action.action_type.value}: {action.content[:content_limit]}"
        if action.file_path:
            line += f" (file: {action.file_path})"
        if total + len(line) > max_chars:
            parts.append(f"... ({len(actions) - i} more actions truncated)")
            break
        parts.append(line)
        total += len(line)
    return "\n".join(parts)
