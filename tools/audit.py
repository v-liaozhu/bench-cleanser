#!/usr/bin/env python3
"""Unified audit CLI for SWE-bench Pro contamination analysis.

Replaces 6 separate scripts (3,172 lines) with one cohesive toolkit (~900 lines).
All data loading, trajectory fetching, LLM analysis, human verdicting, and
report generation in a single entry point.

Usage:
    python audit.py fetch trajectories       Fetch trajectory summary from Docent
    python audit.py fetch swebench           Cache SWE-bench Pro dataset locally

    python audit.py analyze [CASES...] [--force] [--blind]
                                             LLM trajectory audit
    python audit.py analyze all --blind      Blind analysis (no pipeline context)

    python audit.py status                   Show audit progress dashboard
    python audit.py show <case_num>          Show detailed case summary
    python audit.py record <case_num> <verdict> "<reason>"
                                             Record a human verdict

    python audit.py export tracker           Regenerate tracker CSV from reports
    python audit.py export triage            Generate reviewer-friendly triage CSV

    python audit.py report case-studies [N]  Generate case study markdown files
    python audit.py report forensic          Run forensic analysis on all reports
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import textwrap
import time
from collections import Counter, defaultdict
from datetime import date
from itertools import combinations
from pathlib import Path
from typing import Any

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class AuditConfig:
    """Paths, credentials, and constants for the audit workflow."""

    PROJECT_ROOT = Path(__file__).resolve().parent
    AUDIT_DIR = PROJECT_ROOT / "audits" / "severe"
    REPORTS_DIR = PROJECT_ROOT / "output_pro_v6" / "reports"

    # Tracker and notes
    TRACKER_CSV = AUDIT_DIR / "audit_tracker_v2.csv"
    NOTES_DIR = AUDIT_DIR / "notes_v2"
    LLM_ANALYSIS_DIR = AUDIT_DIR / "llm_analysis"

    # Data caches
    SWEBENCH_CACHE = AUDIT_DIR / "swebench_pro_cache.json"
    TRAJECTORY_CACHE = AUDIT_DIR / "trajectory_cache"
    TRAJECTORY_SUMMARY = AUDIT_DIR / "trajectory_summary_v2.json"

    # Docent API (env-based, never hardcoded)
    DOCENT_API_URL = "https://api.docent.transluce.org"
    DOCENT_COLLECTION_ID = "196681cc-76fc-44f2-b3ce-d55eba81c0c6"

    # Report output
    CASE_STUDIES_DIR = PROJECT_ROOT / "case_studies" / "pro_severe"
    FORENSIC_DIR = PROJECT_ROOT / "analysis_v3"

    # Model priority for trajectory selection (strongest first)
    MODEL_PRIORITY = [
        "Claude Opus 4.1 - paper",
        "GPT-5 High - paper",
        "Gemini 2.5 Pro Preview - paper",
        "Claude 4.5 Sonnet - 10132025",
        "GPT-5 - 10132025",
        "GPT-5 Codex -- debug-oct22",
        "Claude 4 Sonnet - 10132025",
        "Kimi - paper",
        "GLM-4.5 -- 10222025",
        "Claude 4.5 Haiku -- 10222025",
        "GPT OSS - paper",
    ]

    VALID_VERDICTS = [
        "CONFIRMED_SEVERE",
        "DOWNGRADE_MODERATE",
        "DOWNGRADE_MINOR",
        "DOWNGRADE_CLEAN",
        "ESCALATE",
    ]

    @staticmethod
    def docent_api_key() -> str:
        key = os.environ.get("DOCENT_API_KEY", "")
        if not key:
            print("ERROR: DOCENT_API_KEY environment variable not set.")
            print("  export DOCENT_API_KEY='dk_...'")
            sys.exit(1)
        return key


CFG = AuditConfig()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DataManager — single source of truth for all I/O
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class DataManager:
    """Loads and saves all audit data: SWE-bench, tracker CSV, trajectories,
    pipeline reports, and LLM analysis results."""

    # ── SWE-bench Pro ─────────────────────────────────────────────

    @staticmethod
    def load_swebench() -> dict[str, dict]:
        """Load SWE-bench Pro, caching locally as JSON."""
        if CFG.SWEBENCH_CACHE.exists():
            with open(CFG.SWEBENCH_CACHE, encoding="utf-8") as f:
                return {row["instance_id"]: row for row in json.load(f)}

        from datasets import load_dataset

        ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
        rows = [dict(row) for row in ds]

        CFG.SWEBENCH_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(CFG.SWEBENCH_CACHE, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)
        print(f"Cached {len(rows)} SWE-bench Pro tasks to {CFG.SWEBENCH_CACHE}")
        return {row["instance_id"]: row for row in rows}

    # ── Tracker CSV ───────────────────────────────────────────────

    @staticmethod
    def load_tracker() -> list[dict]:
        """Load the audit tracker CSV."""
        with open(CFG.TRACKER_CSV, encoding="utf-8") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def save_tracker(rows: list[dict]) -> None:
        """Write the audit tracker CSV (single implementation, no duplicates)."""
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        with open(CFG.TRACKER_CSV, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # ── Trajectory Summary ────────────────────────────────────────

    @staticmethod
    def load_trajectory_summary() -> dict:
        """Load cached trajectory summary, fetching from Docent if missing."""
        if CFG.TRAJECTORY_SUMMARY.exists():
            with open(CFG.TRAJECTORY_SUMMARY, encoding="utf-8") as f:
                return json.load(f)
        return TrajectoryFetcher.fetch_summary()

    # ── Per-case trajectory cache ─────────────────────────────────

    @staticmethod
    def load_trajectory(instance_id: str) -> dict | None:
        path = CFG.TRAJECTORY_CACHE / f"{instance_id}.json"
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_trajectory(instance_id: str, data: dict) -> None:
        CFG.TRAJECTORY_CACHE.mkdir(parents=True, exist_ok=True)
        path = CFG.TRAJECTORY_CACHE / f"{instance_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    # ── LLM analysis results ─────────────────────────────────────

    @staticmethod
    def load_analysis(case_num: int) -> dict | None:
        path = CFG.LLM_ANALYSIS_DIR / f"case_{case_num:03d}.json"
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_analysis(case_num: int, result: dict) -> None:
        CFG.LLM_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        path = CFG.LLM_ANALYSIS_DIR / f"case_{case_num:03d}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    # ── Pipeline reports ──────────────────────────────────────────

    @staticmethod
    def load_reports() -> list[dict]:
        """Load all pipeline JSON reports."""
        reports = []
        if not CFG.REPORTS_DIR.exists():
            return reports
        for f in sorted(CFG.REPORTS_DIR.iterdir()):
            if f.suffix == ".json":
                with open(f, encoding="utf-8") as fh:
                    reports.append(json.load(fh))
        return reports

    @staticmethod
    def load_report(instance_id: str) -> dict | None:
        """Load a single pipeline report by instance_id."""
        path = CFG.REPORTS_DIR / f"{instance_id}.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        # Fuzzy match
        if CFG.REPORTS_DIR.exists():
            for fn in CFG.REPORTS_DIR.iterdir():
                if instance_id in fn.name:
                    with open(fn, encoding="utf-8") as f:
                        return json.load(f)
        return None

    # ── Resolution rate query ─────────────────────────────────────

    @staticmethod
    def count_all_fail_tasks() -> tuple[int, int]:
        """Return (all_fail_count, total_instances) from trajectory summary."""
        summary = DataManager.load_trajectory_summary()
        total = len(summary)
        all_fail = sum(1 for v in summary.values() if v.get("all_fail"))
        return all_fail, total


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TrajectoryFetcher — Docent API integration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TrajectoryFetcher:
    """Fetches trajectory data from the Docent API with local caching."""

    _client = None

    @classmethod
    def _get_client(cls):
        if cls._client is None:
            from docent import Docent

            cls._client = Docent(
                api_key=CFG.docent_api_key(),
                api_url=CFG.DOCENT_API_URL,
            )
        return cls._client

    @classmethod
    def fetch_summary(cls) -> dict:
        """Fetch resolution rates for all instances via DQL batched queries."""
        client = cls._get_client()
        cid = CFG.DOCENT_COLLECTION_ID

        all_dicts = []
        offset = 0
        batch_size = 5000
        while True:
            batch = client.execute_dql(
                cid,
                dql=f"SELECT agent_runs.id, agent_runs.metadata_json, "
                    f"agent_runs.created_at FROM agent_runs "
                    f"ORDER BY agent_runs.created_at DESC "
                    f"LIMIT {batch_size} OFFSET {offset}",
            )
            batch_dicts = client.dql_result_to_dicts(batch)
            all_dicts.extend(batch_dicts)
            if len(batch_dicts) < batch_size:
                break
            offset += batch_size

        instance_runs: dict[str, list] = defaultdict(list)
        for r in all_dicts:
            meta = r.get("metadata_json", {})
            if isinstance(meta, dict):
                iid = meta.get("instance_id", "")
                instance_runs[iid].append({
                    "run_id": r.get("id", ""),
                    "model": meta.get("model_name", "?"),
                    "resolved": meta.get("resolved"),
                    "turns": meta.get("turns"),
                    "created_at": r.get("created_at", ""),
                })

        summary = {}
        for iid, runs in instance_runs.items():
            passes = sum(1 for r in runs if r["resolved"] is True)
            fails = sum(1 for r in runs if r["resolved"] is False)
            summary[iid] = {
                "total_runs": len(runs),
                "passes": passes,
                "fails": fails,
                "resolve_rate": passes / len(runs) if runs else None,
                "all_fail": len(runs) > 0 and passes == 0,
                "runs": runs,
            }

        CFG.TRAJECTORY_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
        with open(CFG.TRAJECTORY_SUMMARY, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        print(f"Saved trajectory summary: {len(summary)} instances, {len(all_dicts)} total runs")
        return summary

    @classmethod
    def fetch_trajectory(cls, instance_id: str) -> dict:
        """Fetch a representative trajectory for an instance.

        Strategy: prefer a FAILED run from the strongest model available.
        Truncates oversized trajectories to head(55%) + tail(40%).
        Results are cached to trajectory_cache/.
        """
        cached = DataManager.load_trajectory(instance_id)
        if cached is not None:
            return cached

        client = cls._get_client()
        cid = CFG.DOCENT_COLLECTION_ID
        summary = DataManager.load_trajectory_summary()
        instance_data = summary.get(instance_id, {})
        runs = instance_data.get("runs", [])

        if not runs:
            return {"error": "No runs found", "model": None, "resolved": None}

        # Select best run: prefer failed from strong models
        failed_runs = [r for r in runs if r["resolved"] is False]
        passed_runs = [r for r in runs if r["resolved"] is True]

        def model_rank(run):
            try:
                return CFG.MODEL_PRIORITY.index(run.get("model", ""))
            except ValueError:
                return 999

        target_runs = failed_runs if failed_runs else passed_runs
        target_runs.sort(key=model_rank)
        selected = target_runs[0]

        run_id = selected["run_id"]
        model = selected.get("model", "?")
        resolved = selected.get("resolved")
        print(f"  Fetching trajectory: {model} (resolved={resolved}, run_id={run_id[:8]}...)")

        try:
            run = client.get_agent_run(collection_id=cid, agent_run_id=run_id)
            if not run.transcripts:
                return {"error": "No transcripts", "model": model, "resolved": resolved}

            msgs = run.transcripts[0].messages
            total_msgs = len(msgs)

            full_parts = []
            for i, m in enumerate(msgs):
                role = m.role if hasattr(m, "role") else "?"
                text = m.text if hasattr(m, "text") else str(m.content)
                full_parts.append(f"[MSG {i}] [{role}]\n{text}")

            full_text = "\n\n".join(full_parts)
            total_chars = len(full_text)
            agent_patch = _extract_agent_patch(msgs)

            MAX_CHARS = 400_000  # ~100K tokens

            if total_chars <= MAX_CHARS:
                result = {
                    "model": model, "resolved": resolved, "turns": total_msgs,
                    "total_chars": total_chars,
                    "total_tokens_approx": total_chars // 4,
                    "full_text": full_text, "head_text": None, "tail_text": None,
                    "agent_patch": agent_patch, "truncated": False,
                    "run_id": run_id,
                }
            else:
                head_count = max(3, int(total_msgs * 0.30))
                tail_count = max(3, int(total_msgs * 0.30))

                head_parts = []
                for i, m in enumerate(msgs[:head_count]):
                    role = m.role if hasattr(m, "role") else "?"
                    text = m.text if hasattr(m, "text") else str(m.content)
                    head_parts.append(f"[MSG {i}] [{role}]\n{text}")
                head_text = "\n\n".join(head_parts)

                tail_start = total_msgs - tail_count
                tail_parts = []
                for i, m in enumerate(msgs[tail_start:]):
                    role = m.role if hasattr(m, "role") else "?"
                    text = m.text if hasattr(m, "text") else str(m.content)
                    tail_parts.append(f"[MSG {tail_start + i}] [{role}]\n{text}")
                tail_text = "\n\n".join(tail_parts)

                head_budget = MAX_CHARS * 4 // 10
                tail_budget = MAX_CHARS * 6 // 10
                if len(head_text) > head_budget:
                    head_text = head_text[:head_budget] + "\n\n[... HEAD TRUNCATED ...]"
                if len(tail_text) > tail_budget:
                    tail_text = "[... TAIL START TRUNCATED ...]\n\n" + tail_text[-tail_budget:]

                result = {
                    "model": model, "resolved": resolved, "turns": total_msgs,
                    "total_chars": total_chars,
                    "total_tokens_approx": total_chars // 4,
                    "full_text": None, "head_text": head_text, "tail_text": tail_text,
                    "agent_patch": agent_patch, "truncated": True,
                    "run_id": run_id,
                    "head_msgs": head_count, "tail_msgs": tail_count,
                }

            DataManager.save_trajectory(instance_id, result)
            return result

        except Exception as e:
            return {"error": str(e), "model": model, "resolved": resolved}


def _extract_agent_patch(messages) -> str | None:
    """Extract the agent's submitted patch from the last few messages."""
    for m in reversed(messages[-15:]):
        text = m.text if hasattr(m, "text") else str(m.content)
        if "diff --git" in text:
            lines = text.split("\n")
            diff_lines = []
            in_diff = False
            for line in lines:
                if line.startswith("diff --git"):
                    in_diff = True
                if in_diff:
                    diff_lines.append(line)
            if diff_lines:
                return "\n".join(diff_lines)
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ContaminationAuditor — LLM analysis engine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Framed as benchmark misalignment monitoring, informed by OpenAI's methodology:
# - Contamination = misalignment between task spec and evaluation harness
# - Blind analysis = production evaluation (no pipeline context bias)
# - Cross-validation between biased and blind runs catches evaluation awareness artifacts
AUDIT_SYSTEM_PROMPT = """\
You are a benchmark integrity auditor for SWE-bench Pro. Your job is to \
determine whether a software engineering task is CONTAMINATED — meaning \
the fail-to-pass (F2P) tests require implementation details not derivable \
from the problem statement alone.

IMPORTANT DISTINCTIONS:
- A task is CONTAMINATED when: F2P tests assert on specific implementation \
choices, API shapes, internal magic strings, or architectural patterns that \
an agent could NOT reasonably infer from the problem description. An agent \
that correctly solves the stated problem would STILL fail these tests.
- A task is GENUINELY HARD when: the agent fails because the problem is \
complex, requires deep domain knowledge, or the agent made implementation \
mistakes. The tests are fair — they would accept ANY correct solution.
- A task has OVERPATCH when: the gold patch touches code beyond what the \
problem requires, but the tests only check the stated functionality. This \
is noise, not contamination.

CONTAMINATION PATTERNS TO LOOK FOR:
1. approach_lock: Tests pre-staged via before_repo_set_cmd assert on exact \
implementation patterns (specific function names, CSS class names, DOM structure, \
configuration keys) not mentioned in the problem statement.
2. task_patch_mismatch: The problem describes Feature A but the gold patch \
implements Feature B (and tests verify Feature B).
3. test_coupling: Tests are tightly coupled to gold patch internals — they \
import specific helper functions, check exact error messages, or verify \
internal state changes only the gold patch would produce.
4. undiscoverable_requirement: Tests check for behavior or edge cases \
never mentioned or implied in the problem statement.

WHAT TO LOOK FOR IN THE TRAJECTORY:
- Did the agent READ and UNDERSTAND the problem statement?
- Did the agent's implementation CORRECTLY address the stated problem?
- WHY did the agent's implementation fail the tests? Was it because:
  (a) The agent's solution was wrong/incomplete (= genuinely hard), or
  (b) The agent solved the problem correctly but the tests rejected it \
      for reasons not in the spec (= contamination evidence)
- Did the agent iterate? Did it try multiple approaches? Were they all rejected?

TRAJECTORY FORMAT:
Messages are delimited by [MSG N] [role]. User messages contain file contents \
and command outputs. Assistant messages contain the agent's reasoning and actions.

Respond in JSON with these fields:
{
    "agent_understood_problem": true/false,
    "agent_approach_summary": "1-2 sentence summary of what the agent tried",
    "agent_addressed_spec": true/false,
    "failure_reason": "contamination" | "genuine_difficulty" | "agent_error" | "timeout" | "unclear",
    "contamination_type": null | "approach_lock" | "task_patch_mismatch" | "test_coupling" | "undiscoverable_requirement",
    "contamination_evidence": "Specific evidence if contamination found, else null",
    "difficulty_evidence": "Specific evidence if genuinely hard, else null",
    "evidence_strength": "strong/moderate/weak",
    "reasoning": "Detailed paragraph explaining your classification",
    "verdict": "CONFIRMED_SEVERE" | "LIKELY_SEVERE" | "INCONCLUSIVE" | "LIKELY_CLEAN" | "CLEAN"
}

VERDICT GUIDELINES:
- CONFIRMED_SEVERE: Clear contamination evidence — agent solved spec correctly, tests reject it for undiscoverable reasons.
- LIKELY_SEVERE: Strong contamination signals but not conclusive.
- INCONCLUSIVE: Cannot determine from this trajectory alone.
- LIKELY_CLEAN: Agent clearly failed due to skill/difficulty, no contamination signals.
- CLEAN: Task is clearly fair — agent's failure is entirely due to incorrect/incomplete implementation."""


class ContaminationAuditor:
    """LLM-powered trajectory analysis for contamination detection."""

    def __init__(self):
        self._llm = None

    def _init_llm(self):
        """Lazy LLM initialization — only when analyze is actually called."""
        if self._llm is not None:
            return
        # Add project root so library imports work
        project_root = str(CFG.PROJECT_ROOT)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from bench_cleanser.llm_client import LLMClient
        from bench_cleanser.models import PipelineConfig

        config = PipelineConfig()
        self._llm = LLMClient(config)
        print(f"LLM initialized: {config.llm_model}")

    @staticmethod
    def build_prompt(task: dict, trajectory: dict, tracker_row: dict) -> str:
        """Build the user prompt for LLM contamination audit."""
        problem = task.get("problem_statement", "N/A")
        gold_patch = task.get("patch", "N/A")
        test_patch = task.get("test_patch", "N/A")
        f2p_tests = task.get("fail_to_pass", "N/A")
        before_cmd = task.get("before_repo_set_cmd", "N/A")

        traj_text = (
            trajectory.get("full_text")
            or trajectory.get("head_text")
            or trajectory.get("tail_text")
            or ""
        )
        agent_patch = trajectory.get("agent_patch") or "(no patch submitted)"
        model = trajectory.get("model", "unknown")
        resolved = trajectory.get("resolved", False)
        turns = trajectory.get("turns", 0)

        # Truncate trajectory: head(55%) + tail(40%) at 200K chars max
        max_traj_chars = 200_000
        if len(traj_text) > max_traj_chars:
            head_size = int(max_traj_chars * 0.55)
            tail_size = int(max_traj_chars * 0.40)
            traj_text = (
                traj_text[:head_size]
                + "\n\n[... TRAJECTORY MIDDLE TRUNCATED FOR LENGTH ...]\n\n"
                + traj_text[-tail_size:]
            )

        return f"""CASE #{tracker_row.get('case_num', 'N/A')} — {tracker_row.get('instance_id', 'N/A')}

PROBLEM STATEMENT:
{problem[:20000]}

GOLD PATCH:
{gold_patch[:30000]}

TEST PATCH (F2P tests that the gold patch must pass):
{test_patch[:20000]}

FAIL-TO-PASS TEST NAMES:
{f2p_tests}

BEFORE_REPO_SET_CMD (pre-stages test files from gold commit):
{before_cmd}

REPRESENTATIVE AGENT TRAJECTORY ({model}, resolved={resolved}, turns={turns}):
{traj_text}

AGENT'S SUBMITTED PATCH:
{agent_patch[:20000]}

---

Analyze this case. Determine whether the agent failed because:
(a) The task is genuinely difficult and the agent couldn't solve it
(b) The task is contaminated — the tests require implementation details not in the problem statement

Focus on WHAT the agent tried, WHETHER it addressed the stated problem correctly, \
and WHY it failed the F2P tests. Look at the test_patch carefully — do the tests \
check things that are stated in the problem, or do they assert on implementation \
details that could only come from knowing the gold patch?"""

    def analyze_case(
        self,
        case_num: int,
        swebench: dict[str, dict],
        tracker_rows: list[dict],
        *,
        force: bool = False,
    ) -> dict | None:
        """Analyze a single case using LLM trajectory analysis."""
        self._init_llm()

        row = next((r for r in tracker_rows if int(r["case_num"]) == case_num), None)
        if row is None:
            print(f"Case {case_num} not found in tracker")
            return None

        instance_id = row["instance_id"]

        # Check if already analyzed
        if not force:
            existing = DataManager.load_analysis(case_num)
            if existing:
                print(f"Case {case_num}: already analyzed (use --force to rerun)")
                return existing

        task = swebench.get(instance_id)
        if not task:
            print(f"Case {case_num}: instance {instance_id} not found in SWE-bench cache")
            return None

        trajectory = DataManager.load_trajectory(instance_id)
        if not trajectory:
            print(f"Case {case_num}: no trajectory cache for {instance_id}")
            return None

        prompt = self.build_prompt(task, trajectory, row)
        print(f"Case {case_num}: analyzing {instance_id} ({row.get('resolution_rate', 'N/A')} resolution)...")

        try:
            result = self._llm.query_json_sync(
                system_prompt=AUDIT_SYSTEM_PROMPT,
                user_prompt=prompt,
            )
        except Exception as e:
            print(f"Case {case_num}: LLM error — {e}")
            return None

        result["case_num"] = case_num
        result["instance_id"] = instance_id
        result["resolution_rate"] = row.get("resolution_rate", "N/A")
        result["all_fail"] = row.get("all_fail", "N/A")
        result["model_analyzed"] = trajectory.get("model", "unknown")
        result["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

        DataManager.save_analysis(case_num, result)

        verdict = result.get("verdict", "UNKNOWN")
        evidence_strength = result.get("evidence_strength", "moderate")
        failure = result.get("failure_reason", "unknown")
        print(f"  -> Verdict: {verdict} (strength={evidence_strength}, failure={failure})")
        return result

    @staticmethod
    def update_tracker(tracker_rows: list[dict], *, blind: bool = False) -> int:
        """Update tracker with LLM analysis results."""
        status_value = "LLM_ANALYZED_BLIND" if blind else "LLM_ANALYZED"
        verdict_map = {
            "CONFIRMED_SEVERE": "CONFIRMED_SEVERE",
            "LIKELY_SEVERE": "CONFIRMED_SEVERE",
            "INCONCLUSIVE": "PENDING_HUMAN_REVIEW",
            "LIKELY_CLEAN": "DOWNGRADE_CLEAN",
            "CLEAN": "DOWNGRADE_CLEAN",
        }
        updated = 0
        for row in tracker_rows:
            case_num = int(row["case_num"])
            result = DataManager.load_analysis(case_num)
            if not result:
                continue
            verdict = result.get("verdict", "")
            human_verdict = verdict_map.get(verdict, "")
            if human_verdict and row.get("human_verdict", "") == "":
                row["human_verdict"] = human_verdict
                row["verdict_reason"] = (
                    f"LLM({result.get('evidence_strength', 'N/A')}): "
                    f"{result.get('failure_reason', 'unknown')} — "
                    f"{result.get('reasoning', '')[:200]}"
                )
                row["audited_by"] = "gpt-5.4-20260305"
                row["audit_date"] = result.get("timestamp", "")
                row["audit_status"] = status_value
                updated += 1
        return updated

    @staticmethod
    def update_notes() -> int:
        """Append LLM analysis section to per-case notes."""
        updated = 0
        for path in sorted(CFG.LLM_ANALYSIS_DIR.glob("case_*.json")):
            case_num = int(path.stem.split("_")[1])
            note_path = CFG.NOTES_DIR / f"case_{case_num:03d}.md"
            if not note_path.exists():
                continue
            with open(path, encoding="utf-8") as f:
                result = json.load(f)

            note_content = note_path.read_text(encoding="utf-8")
            if "## LLM Trajectory Analysis" in note_content:
                continue

            section = f"""

## LLM Trajectory Analysis

**Model**: {result.get('model_analyzed', 'unknown')}
**Analyzed**: {result.get('timestamp', 'N/A')}

| Field | Value |
|-------|-------|
| Agent understood problem | {result.get('agent_understood_problem', 'N/A')} |
| Agent addressed spec | {result.get('agent_addressed_spec', 'N/A')} |
| Failure reason | {result.get('failure_reason', 'N/A')} |
| Contamination type | {result.get('contamination_type', 'N/A')} |
| Evidence Strength | {result.get('evidence_strength', 'N/A')} |
| **Verdict** | **{result.get('verdict', 'N/A')}** |

**Agent approach**: {result.get('agent_approach_summary', 'N/A')}

**Contamination evidence**: {result.get('contamination_evidence', 'None')}

**Difficulty evidence**: {result.get('difficulty_evidence', 'None')}

**Full reasoning**: {result.get('reasoning', 'N/A')}
"""
            if "PENDING_HUMAN_REVIEW" in note_content:
                note_content = note_content.replace(
                    "PENDING_HUMAN_REVIEW",
                    f"LLM_ANALYZED -> {result.get('verdict', 'UNKNOWN')}",
                )
            note_content += section
            note_path.write_text(note_content, encoding="utf-8")
            updated += 1
        return updated


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ReportGenerator — output generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LABEL_EXPLANATIONS = {
    "clean": "No contamination. Problem, patch, and tests are all fair and aligned.",
    "approach_lock": "Tests require a specific implementation approach the problem doesn't determine.",
    "over_test": "F2P tests verify behavior not described in the problem description.",
    "over_patch": "The gold patch includes behavioral changes beyond what the problem asks for.",
    "unclear_description": "The problem description is too ambiguous — multiple incompatible approaches are equally reasonable.",
    "hidden_context": "Critical solution info appears only in hints text, not the main problem description.",
    "weak_coverage": "Tests or patch don't fully cover stated acceptance criteria.",
}


class ReportGenerator:
    """Generates case studies, forensic analysis, triage CSVs."""

    # ── Triage CSV (export tracker from reports) ──────────────────

    @staticmethod
    def generate_triage_csv(reports: list[dict], output_path: Path) -> int:
        """Build reviewer-friendly triage CSV from pipeline reports."""
        rows = []
        for r in reports:
            labels = [la["label"] for la in r.get("task_labels", [])]
            contam_labels = [l for l in labels if l != "clean"]
            primary = contam_labels[0] if contam_labels else (labels[0] if labels else "clean")

            explanations = [
                f"[{l}] {LABEL_EXPLANATIONS.get(l, 'Unknown')}" for l in contam_labels
            ]
            plain_english = " | ".join(explanations) if explanations else "Clean task."

            sev = r["severity"]
            priority = {"SEVERE": 1, "MODERATE": 2, "MINOR": 3}.get(sev, 4)

            ep = r.get("patch_analysis", {})
            et = r.get("test_analysis", {})
            vs = r.get("description_clarity", {})

            evidence_bits = []
            for la in r.get("task_labels", []):
                if la["label"] in contam_labels and la.get("evidence"):
                    evidence_bits.append(f'[{la["label"]}] {la["evidence"][0][:200]}')

            rows.append({
                "instance_id": r["instance_id"],
                "severity": sev,
                "triage_priority": priority,
                "combined_score": round(r["combined_score"], 4),
                "primary_label": primary,
                "all_labels": ", ".join(labels),
                "label_count": len(labels),
                "plain_english": plain_english,
                "core_requirement": r.get("intent", {}).get("core_requirement", "")[:300],
                "ep_score": round(ep.get("score", 0), 4),
                "et_score": round(et.get("score", 0), 4),
                "vs_score": round(vs.get("score", 0), 4),
                "patch_hunks": f'{ep.get("required", 0)}R/{ep.get("ancillary", 0)}A/{ep.get("unrelated", 0)}U',
                "test_breakdown": f'{et.get("aligned", 0)}A/{et.get("tangential", 0)}T/{et.get("unrelated", 0)}U',
                "has_modified_tests": et.get("has_modified_tests", False),
                "ambiguity_score": r.get("intent", {}).get("ambiguity_score", 0),
                "evidence_summary": " || ".join(evidence_bits[:3]) if evidence_bits else "",
            })

        rows.sort(key=lambda x: (x["triage_priority"], -x["combined_score"]))

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        return len(rows)

    # ── Tracker regeneration from reports ─────────────────────────

    @staticmethod
    def regenerate_tracker(reports: list[dict]) -> int:
        """Regenerate audit_tracker.csv from pipeline reports (v1 format)."""
        # Preserve existing verdicts
        existing_verdicts = {}
        old_tracker = CFG.AUDIT_DIR / "audit_tracker.csv"
        if old_tracker.exists():
            with open(old_tracker, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("audit_status") == "DONE":
                        existing_verdicts[row["instance_id"]] = {
                            "human_verdict": row["human_verdict"],
                            "verdict_reason": row["verdict_reason"],
                            "audited_by": row["audited_by"],
                            "audit_date": row["audit_date"],
                        }

        severe_cases = []
        for r in reports:
            if r.get("severity") != "SEVERE":
                continue
            intent = r.get("intent", {})
            ep = r.get("patch_analysis", {})
            et = r.get("test_analysis", {})
            labels = r.get("task_labels", [])
            core_req = intent.get("core_requirement", "")
            is_error = core_req.startswith("PIPELINE_ERROR")

            label_names = [l["label"] for l in labels]
            unrelated = ep.get("unrelated", 0)
            total_hunks = ep.get("total_hunks", 0)

            sort_key = (
                1 if is_error else 0,
                -unrelated,
                -len(label_names),
            )
            severe_cases.append((sort_key, r, is_error, label_names))

        severe_cases.sort(key=lambda x: x[0])

        fieldnames = [
            "case_num", "instance_id", "pipeline_status", "legitimacy", "ambiguity_score",
            "patch_total", "patch_required", "patch_ancillary", "patch_unrelated",
            "patch_unrelated_pct", "patch_files_changed",
            "tests_total", "tests_aligned", "tests_tangential", "tests_unrelated",
            "tests_modified", "assertions_total", "assertions_on_topic", "assertions_off_topic",
            "labels", "label_count",
            "has_cross_ref", "cross_ref_count",
            "description_clarity_score", "core_requirement_excerpt",
            "audit_status", "human_verdict", "verdict_reason", "audited_by", "audit_date",
        ]

        with open(old_tracker, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for i, (_, r, is_error, label_names) in enumerate(severe_cases, 1):
                iid = r["instance_id"]
                intent = r.get("intent", {})
                ep = r.get("patch_analysis", {})
                et = r.get("test_analysis", {})
                vs = r.get("description_clarity", {})
                recs = r.get("recommendations", [])
                decomp = intent.get("decomposition", {})

                hunks = ep.get("hunks", [])
                patch_files = set(h.get("file_path", "") for h in hunks)
                total_h = ep.get("total_hunks", 0)
                unrel = ep.get("unrelated", 0)
                unrel_pct = (unrel / total_h * 100) if total_h > 0 else 0.0

                has_cross_ref = False
                cross_ref_count = 0
                for rec in recs:
                    if "CROSS_REF" in rec:
                        has_cross_ref = True
                        m = re.search(r"(\d+) overpatch-overtest", rec)
                        if m:
                            cross_ref_count += int(m.group(1))

                prev = existing_verdicts.get(iid, {})
                w.writerow({
                    "case_num": i,
                    "instance_id": iid,
                    "pipeline_status": "PIPELINE_ERROR" if is_error else "OK",
                    "legitimacy": decomp.get("legitimacy", ""),
                    "ambiguity_score": f"{intent.get('ambiguity_score', 0):.2f}",
                    "patch_total": total_h,
                    "patch_required": ep.get("required", 0),
                    "patch_ancillary": ep.get("ancillary", 0),
                    "patch_unrelated": unrel,
                    "patch_unrelated_pct": f"{unrel_pct:.0f}",
                    "patch_files_changed": len(patch_files),
                    "tests_total": et.get("total_tests", 0),
                    "tests_aligned": et.get("aligned", 0),
                    "tests_tangential": et.get("tangential", 0),
                    "tests_unrelated": et.get("unrelated", 0),
                    "tests_modified": et.get("has_modified_tests", False),
                    "assertions_total": et.get("total_assertions", 0),
                    "assertions_on_topic": et.get("on_topic", 0),
                    "assertions_off_topic": et.get("off_topic", 0),
                    "labels": ";".join(label_names),
                    "label_count": len(label_names),
                    "has_cross_ref": has_cross_ref,
                    "cross_ref_count": cross_ref_count,
                    "description_clarity_score": f"{vs.get('score', 0):.2f}",
                    "core_requirement_excerpt": intent.get("core_requirement", "")[:120].replace(",", ";").replace("\n", " "),
                    "audit_status": "DONE" if prev else "PENDING",
                    "human_verdict": prev.get("human_verdict", ""),
                    "verdict_reason": prev.get("verdict_reason", ""),
                    "audited_by": prev.get("audited_by", ""),
                    "audit_date": prev.get("audit_date", ""),
                })

        return len(severe_cases)

    # ── Forensic analysis ─────────────────────────────────────────

    @staticmethod
    def run_forensic(reports: list[dict]) -> dict:
        """Run full forensic analysis on all reports, save results."""
        CFG.FORENSIC_DIR.mkdir(exist_ok=True)

        # Distributions
        severity_counts = Counter()
        label_counts = Counter()
        scores = []

        for r in reports:
            severity_counts[r["severity"]] += 1
            scores.append(r["combined_score"])
            for la in r.get("task_labels", []):
                label_counts[la["label"]] += 1

        # Co-occurrence
        pair_counts: Counter = Counter()
        for r in reports:
            labels = [la["label"] for la in r.get("task_labels", []) if la["label"] != "clean"]
            for a, b in combinations(sorted(set(labels)), 2):
                pair_counts[(a, b)] += 1

        # Per-project
        project_stats: dict = defaultdict(lambda: {"total": 0, "severity": Counter(), "labels": Counter()})
        for r in reports:
            proj = r["instance_id"].split("__")[0]
            project_stats[proj]["total"] += 1
            project_stats[proj]["severity"][r["severity"]] += 1
            for la in r.get("task_labels", []):
                if la["label"] != "clean":
                    project_stats[proj]["labels"][la["label"]] += 1

        # Hunk stats
        hunk_stats = {"total": 0, "required": 0, "ancillary": 0, "unrelated": 0}
        for r in reports:
            ep = r.get("patch_analysis", {})
            hunk_stats["total"] += ep.get("total_hunks", 0)
            hunk_stats["required"] += ep.get("required", 0)
            hunk_stats["ancillary"] += ep.get("ancillary", 0)
            hunk_stats["unrelated"] += ep.get("unrelated", 0)

        analysis = {
            "total_reports": len(reports),
            "severity_counts": dict(severity_counts),
            "label_counts": dict(label_counts.most_common()),
            "score_stats": {
                "mean": round(sum(scores) / len(scores), 4) if scores else 0,
                "median": round(sorted(scores)[len(scores) // 2], 4) if scores else 0,
                "min": round(min(scores), 4) if scores else 0,
                "max": round(max(scores), 4) if scores else 0,
            },
            "top_cooccurrences": {f"{a} + {b}": c for (a, b), c in pair_counts.most_common(20)},
            "hunk_stats": hunk_stats,
            "per_project": {
                proj: {
                    "total": s["total"],
                    "contamination_rate": round(
                        (s["total"] - s["severity"].get("CLEAN", 0)) / s["total"] * 100, 1
                    ),
                    "severity": dict(s["severity"]),
                    "top_labels": dict(s["labels"].most_common(5)),
                }
                for proj, s in sorted(project_stats.items(), key=lambda x: x[1]["total"], reverse=True)
            },
        }

        output_path = CFG.FORENSIC_DIR / "forensic_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, default=str)

        # Print summary
        print(f"Forensic analysis: {len(reports)} reports")
        print(f"  Severity: {dict(severity_counts)}")
        print(f"  Labels: {dict(label_counts.most_common(5))}")
        print(f"  Hunks: {hunk_stats}")
        print(f"  Saved to {output_path}")
        return analysis

    # ── Case studies ──────────────────────────────────────────────

    @staticmethod
    def generate_case_studies(reports: list[dict], num_cases: int = 25) -> int:
        """Generate case study markdown files for top SEVERE cases."""
        CFG.CASE_STUDIES_DIR.mkdir(parents=True, exist_ok=True)

        # Rank severe cases
        severe = []
        for r in reports:
            if r.get("severity") != "SEVERE":
                continue
            labels = r.get("task_labels", [])
            score = min(len(labels) / 5, 1.0)
            severe.append((score, r))
        severe.sort(key=lambda x: -x[0])

        # Load SWE-bench Pro for full problem statements
        swebench = DataManager.load_swebench()

        generated = 0
        for idx, (_, report) in enumerate(severe[:num_cases], 1):
            iid = report["instance_id"]
            original = swebench.get(iid)
            print(f"[{idx:02d}/{num_cases}] {iid}")

            md = _build_case_study_md(idx, report, original)

            safe_name = iid.replace("/", "_").replace("\\", "_")[:100]
            path = CFG.CASE_STUDIES_DIR / f"case_{idx:02d}_{safe_name}.md"
            with open(path, "w", encoding="utf-8") as f:
                f.write(md)
            generated += 1

        print(f"Generated {generated} case studies in {CFG.CASE_STUDIES_DIR}/")
        return generated


def _build_case_study_md(idx: int, report: dict, original: dict | None) -> str:
    """Build a comprehensive case study markdown document."""
    iid = report.get("instance_id", "unknown")
    intent = report.get("intent", {})
    ep = report.get("patch_analysis", {})
    et = report.get("test_analysis", {})
    vs = report.get("description_clarity", {})
    labels = report.get("task_labels", [])
    recs = report.get("recommendations", [])

    repo = original.get("repo", "unknown") if original else "unknown"
    problem = original.get("problem_statement", "_Not available_") if original else "_Not available_"
    patch = original.get("patch", "") if original else ""
    test_patch = original.get("test_patch", "") if original else ""
    lang = original.get("repo_language", "unknown") if original else "unknown"

    label_names = [l["label"].upper() for l in labels]

    lines = [
        f"# Case Study {idx:02d}: {repo}",
        f"## Instance: `{iid}`\n",
        f"**Severity**: SEVERE | **Labels**: {', '.join(label_names) or 'NONE'} | "
        f"**Language**: {lang}\n",
        "---\n",
        "## 1. Problem Statement\n",
        "<details><summary>Full problem statement</summary>\n",
        f"{problem}\n",
        "</details>\n",
        "## 2. Intent Extraction\n",
        f"**Core Requirement**: {intent.get('core_requirement', 'N/A')}\n",
        f"**Behavioral Contract**: {intent.get('behavioral_contract', 'N/A')}\n",
    ]

    criteria = intent.get("acceptance_criteria", [])
    if criteria:
        lines.append("**Acceptance Criteria**:\n")
        for i, c in enumerate(criteria, 1):
            lines.append(f"{i}. {c}")
        lines.append("")

    # Patch analysis
    total_h = ep.get("total_hunks", 0)
    lines.append("## 3. Gold Patch Analysis\n")
    lines.append(f"| Required | Ancillary | Unrelated | Total |")
    lines.append(f"|----------|-----------|-----------|-------|")
    lines.append(f"| {ep.get('required', 0)} | {ep.get('ancillary', 0)} | {ep.get('unrelated', 0)} | {total_h} |\n")

    # Test analysis
    lines.append("## 4. F2P Test Assessment\n")
    lines.append(f"| Aligned | Tangential | Unrelated | Total |")
    lines.append(f"|---------|------------|-----------|-------|")
    lines.append(f"| {et.get('aligned', 0)} | {et.get('tangential', 0)} | {et.get('unrelated', 0)} | {et.get('total_tests', 0)} |\n")

    # Labels
    lines.append("## 5. Contamination Labels\n")
    for la in labels:
        lines.append(f"### `{la['label'].upper()}`\n")
        lines.append(f"{la.get('reasoning', 'N/A')}\n")
        if la.get("evidence"):
            for j, ev in enumerate(la["evidence"], 1):
                lines.append(f"{j}. {ev}")
            lines.append("")

    # Recommendations
    if recs:
        lines.append("## 6. Recommendations\n")
        for r in recs:
            lines.append(f"- {r}")
        lines.append("")

    # Gold patch diff
    lines.append("## 7. Gold Patch\n")
    lines.append("<details><summary>Full diff</summary>\n")
    lines.append("```diff")
    if patch:
        patch_lines = patch.splitlines()
        if len(patch_lines) > 500:
            lines.append("\n".join(patch_lines[:500]))
            lines.append(f"\n... [{len(patch_lines) - 500} more lines]")
        else:
            lines.append(patch)
    lines.append("```\n</details>\n")

    lines.append("---\n*Generated by bench-cleanser v2.0*\n")
    return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI Subcommands
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def cmd_fetch(args):
    """Subcommand: fetch data from external sources."""
    target = args.target
    if target == "trajectories":
        TrajectoryFetcher.fetch_summary()
    elif target == "swebench":
        DataManager.load_swebench()
        print("SWE-bench Pro cache ready.")
    else:
        print(f"Unknown fetch target: {target}. Use 'trajectories' or 'swebench'.")


def cmd_analyze(args):
    """Subcommand: run LLM trajectory audit."""
    tracker_rows = DataManager.load_tracker()
    swebench = DataManager.load_swebench()
    auditor = ContaminationAuditor()

    # Parse case numbers
    case_nums = []
    for c in (args.cases or ["all"]):
        if c == "all":
            case_nums = [int(r["case_num"]) for r in tracker_rows]
            break
        elif c == "allfail":
            case_nums = [
                int(r["case_num"]) for r in tracker_rows
                if r.get("all_fail") == "YES"
            ]
            break
        else:
            case_nums.append(int(c))

    print(f"Analyzing {len(case_nums)} cases (force={args.force})...")
    batch_size = getattr(args, "batch_size", 5)
    for i, cn in enumerate(case_nums):
        if i > 0 and i % batch_size == 0:
            print(f"--- Batch pause ({i}/{len(case_nums)}) ---")
            time.sleep(2)
        auditor.analyze_case(cn, swebench, tracker_rows, force=args.force)

    # Update tracker and notes
    n_tracker = ContaminationAuditor.update_tracker(
        tracker_rows, blind=getattr(args, "blind", False)
    )
    DataManager.save_tracker(tracker_rows)
    n_notes = ContaminationAuditor.update_notes()
    print(f"\nDone. Updated {n_tracker} tracker rows, {n_notes} notes.")


def cmd_status(_args):
    """Subcommand: show audit progress dashboard."""
    tracker_rows = DataManager.load_tracker()
    total = len(tracker_rows)

    analyzed = len(list(CFG.LLM_ANALYSIS_DIR.glob("case_*.json"))) if CFG.LLM_ANALYSIS_DIR.exists() else 0

    verdicts: Counter = Counter()
    contamination_types: Counter = Counter()
    failure_reasons: Counter = Counter()

    if CFG.LLM_ANALYSIS_DIR.exists():
        for path in CFG.LLM_ANALYSIS_DIR.glob("case_*.json"):
            with open(path, encoding="utf-8") as f:
                result = json.load(f)
            verdicts[result.get("verdict", "UNKNOWN")] += 1
            ct = result.get("contamination_type") or "none"
            contamination_types[ct] += 1
            failure_reasons[result.get("failure_reason", "unknown")] += 1

    # Human verdict counts from tracker
    human_verdicts: Counter = Counter()
    for r in tracker_rows:
        hv = r.get("human_verdict", "")
        if hv:
            human_verdicts[hv] += 1

    print(f"\n{'='*60}")
    print(f"  AUDIT STATUS: {analyzed}/{total} cases analyzed")
    print(f"{'='*60}")

    if verdicts:
        print(f"\n  LLM Verdicts:")
        for v, count in sorted(verdicts.items()):
            bar = "#" * count
            print(f"    {v:<25s} {count:>3d}  {bar}")

    if human_verdicts:
        print(f"\n  Human Verdicts:")
        for v, count in sorted(human_verdicts.items()):
            bar = "#" * count
            print(f"    {v:<25s} {count:>3d}  {bar}")

    if contamination_types:
        print(f"\n  Contamination Types:")
        for ct, count in sorted(contamination_types.items()):
            print(f"    {ct}: {count}")

    if failure_reasons:
        print(f"\n  Failure Reasons:")
        for fr, count in sorted(failure_reasons.items()):
            print(f"    {fr}: {count}")

    remaining = total - analyzed
    if remaining > 0:
        print(f"\n  Remaining: {remaining} cases")

    # Resolution rate quick stat
    if CFG.TRAJECTORY_SUMMARY.exists():
        all_fail, total_tasks = DataManager.count_all_fail_tasks()
        print(f"\n  Resolution: {all_fail}/{total_tasks} tasks have 0% resolve rate")

    print()


def cmd_show(args):
    """Subcommand: show detailed case summary."""
    case_num = args.case_num
    tracker_rows = DataManager.load_tracker()

    row = next((r for r in tracker_rows if int(r["case_num"]) == case_num), None)
    if not row:
        print(f"Case #{case_num} not found.")
        return

    instance_id = row["instance_id"]
    print(f"\n{'='*70}")
    print(f"  CASE #{case_num}: {instance_id}")
    print(f"{'='*70}")

    # Tracker metadata
    print(f"\n  Status: {row.get('audit_status', 'N/A')}")
    if row.get("human_verdict"):
        print(f"  Verdict: {row['human_verdict']}")
        print(f"  Reason: {row.get('verdict_reason', '')}")

    print(f"\n  Resolution: {row.get('resolution_rate', 'N/A')} ({row.get('total_runs', '?')} runs)")
    print(f"  All fail: {row.get('all_fail', 'N/A')}")
    print(f"  Patch hunks: {row.get('patch_hunks', 'N/A')}")
    print(f"  F2P count: {row.get('f2p_count', 'N/A')}")

    # LLM analysis if available
    analysis = DataManager.load_analysis(case_num)
    if analysis:
        print(f"\n--- LLM Analysis ---")
        print(f"  Verdict: {analysis.get('verdict', 'N/A')}")
        print(f"  Evidence: {analysis.get('evidence_strength', 'N/A')}")
        print(f"  Failure reason: {analysis.get('failure_reason', 'N/A')}")
        print(f"  Contamination type: {analysis.get('contamination_type', 'N/A')}")
        print(f"  Agent understood: {analysis.get('agent_understood_problem', 'N/A')}")
        print(f"  Agent addressed spec: {analysis.get('agent_addressed_spec', 'N/A')}")
        approach = analysis.get("agent_approach_summary", "")
        if approach:
            for line in textwrap.wrap(approach, 66):
                print(f"    {line}")
        reasoning = analysis.get("reasoning", "")
        if reasoning:
            print(f"\n  Reasoning:")
            for line in textwrap.wrap(reasoning, 66):
                print(f"    {line}")

    # Pipeline report if available
    report = DataManager.load_report(instance_id)
    if report:
        labels = report.get("task_labels", [])
        if labels:
            print(f"\n--- Pipeline Labels ---")
            for la in labels:
                print(f"  {la['label']}")
    print()


def cmd_record(args):
    """Subcommand: record a human verdict."""
    verdict = args.verdict.upper()
    if verdict not in CFG.VALID_VERDICTS:
        print(f"Invalid verdict: {verdict}")
        print(f"Valid: {', '.join(CFG.VALID_VERDICTS)}")
        return

    tracker_rows = DataManager.load_tracker()
    found = False
    for r in tracker_rows:
        if int(r["case_num"]) == args.case_num:
            r["audit_status"] = "DONE"
            r["human_verdict"] = verdict
            r["verdict_reason"] = args.reason
            r["audited_by"] = "human_expert"
            r["audit_date"] = date.today().isoformat()
            found = True
            break

    if not found:
        print(f"Case #{args.case_num} not found.")
        return

    DataManager.save_tracker(tracker_rows)
    print(f"Recorded: Case #{args.case_num} -> {verdict}")
    print(f"Reason: {args.reason}")


def cmd_export(args):
    """Subcommand: export data."""
    target = args.target
    reports = DataManager.load_reports()

    if target == "tracker":
        n = ReportGenerator.regenerate_tracker(reports)
        print(f"Regenerated tracker with {n} SEVERE cases")
    elif target == "triage":
        path = CFG.REPORTS_DIR.parent / "triage_review.csv"
        n = ReportGenerator.generate_triage_csv(reports, path)
        print(f"Wrote {n} rows to {path}")
    else:
        print(f"Unknown export target: {target}. Use 'tracker' or 'triage'.")


def cmd_report(args):
    """Subcommand: generate reports."""
    target = args.target
    reports = DataManager.load_reports()

    if target == "case-studies":
        num = getattr(args, "num", 25)
        ReportGenerator.generate_case_studies(reports, num_cases=num)
    elif target == "forensic":
        ReportGenerator.run_forensic(reports)
    else:
        print(f"Unknown report target: {target}. Use 'case-studies' or 'forensic'.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI Entry Point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main():
    parser = argparse.ArgumentParser(
        description="Unified audit CLI for SWE-bench Pro contamination analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # fetch
    p_fetch = sub.add_parser("fetch", help="Fetch data from external sources")
    p_fetch.add_argument("target", choices=["trajectories", "swebench"],
                         help="What to fetch")
    p_fetch.set_defaults(func=cmd_fetch)

    # analyze
    p_analyze = sub.add_parser("analyze", help="Run LLM trajectory audit")
    p_analyze.add_argument("cases", nargs="*",
                           help="Case numbers, 'all', or 'allfail'")
    p_analyze.add_argument("--force", action="store_true",
                           help="Re-analyze even if already done")
    p_analyze.add_argument("--blind", action="store_true",
                           help="Mark as blind analysis (no pipeline context)")
    p_analyze.add_argument("--batch-size", type=int, default=5,
                           help="Cases per batch for rate limiting")
    p_analyze.set_defaults(func=cmd_analyze)

    # status
    p_status = sub.add_parser("status", help="Show audit progress dashboard")
    p_status.set_defaults(func=cmd_status)

    # show
    p_show = sub.add_parser("show", help="Show detailed case summary")
    p_show.add_argument("case_num", type=int, help="Case number to display")
    p_show.set_defaults(func=cmd_show)

    # record
    p_record = sub.add_parser("record", help="Record a human verdict")
    p_record.add_argument("case_num", type=int)
    p_record.add_argument("verdict", help=f"One of: {', '.join(CFG.VALID_VERDICTS)}")
    p_record.add_argument("reason", help="Reason for the verdict")
    p_record.set_defaults(func=cmd_record)

    # export
    p_export = sub.add_parser("export", help="Export data")
    p_export.add_argument("target", choices=["tracker", "triage"],
                          help="What to export")
    p_export.set_defaults(func=cmd_export)

    # report
    p_report = sub.add_parser("report", help="Generate reports")
    p_report.add_argument("target", choices=["case-studies", "forensic"],
                          help="Report type")
    p_report.add_argument("--num", type=int, default=25,
                          help="Number of case studies to generate")
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
