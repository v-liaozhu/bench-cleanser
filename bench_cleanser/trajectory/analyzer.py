"""Orchestrator for trajectory analysis.

Coordinates loading, classification, and reporting of agent trajectories.
Uses LLM-primary analysis for trajectory classification — heuristics
provide supporting signals but the LLM makes the final decision.
"""

from __future__ import annotations

import json
import logging
import pathlib
from collections import Counter, defaultdict
from typing import Any

from bench_cleanser.fusion import FusionVerdict, fuse
from bench_cleanser.models import ContaminationReport, TaskRecord
from bench_cleanser.trajectory.classifier import (
    classify_cross_agent,
    classify_heuristic_only,
    classify_with_llm,
    extract_heuristic_signals,
)
from bench_cleanser.trajectory.loader import load_trajectories
from bench_cleanser.trajectory.models import (
    LeakagePattern,
    TrajectoryAnalysis,
    TrajectoryRecord,
)

logger = logging.getLogger(__name__)


async def analyze_trajectories(
    trajectories: list[TrajectoryRecord],
    gold_patches: dict[str, str],
    f2p_tests: dict[str, list[str]],
    problem_statements: dict[str, str],
    llm: Any | None = None,
    contamination_reports: dict[str, ContaminationReport] | None = None,
    max_concurrency: int = 10,
) -> list[TrajectoryAnalysis]:
    import asyncio

    # Bound LLM fan-out so we don't fire N parallel Azure requests when called
    # with hundreds of trajectories. Without this cap rate-limit storms drive
    # _analyze_one into its heuristic-only fallback, silently degrading the
    # LLM-primary analysis the pipeline claims to perform.
    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    completed = 0
    pattern_counts: dict[str, int] = defaultdict(int)

    try:
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
        )
        use_rich = True
    except ImportError:
        use_rich = False

    async def _analyze_one(traj: TrajectoryRecord, progress=None, task_id=None) -> TrajectoryAnalysis:
        nonlocal completed
        async with semaphore:
            gold_patch = gold_patches.get(traj.instance_id, "")
            test_names = f2p_tests.get(traj.instance_id, [])
            problem = problem_statements.get(traj.instance_id, "")

            heuristic_signals = extract_heuristic_signals(traj, gold_patch, test_names)

            contamination_context = ""
            if contamination_reports and traj.instance_id in contamination_reports:
                report = contamination_reports[traj.instance_id]
                contamination_context = _build_contamination_context(report)

            if llm is not None and problem:
                result = await classify_with_llm(
                    traj, gold_patch, problem, test_names, llm,
                    heuristic_signals=heuristic_signals,
                    contamination_context=contamination_context,
                )
            else:
                result = classify_heuristic_only(traj, gold_patch, test_names)

            completed += 1
            pattern_counts[result.leakage_pattern.value] += 1

            if progress is not None and task_id is not None:
                status_parts = []
                for p, c in sorted(pattern_counts.items()):
                    status_parts.append(f"{p}:{c}")
                progress.update(task_id, advance=1, status=" ".join(status_parts))

            logger.debug(
                "%s/%s: %s (strength=%s, sim=%.2f)",
                traj.instance_id, traj.agent_name,
                result.leakage_pattern.value, result.evidence_strength,
                result.gold_patch_similarity,
            )
            return result

    if use_rich:
        console = Console()
        with Progress(
            SpinnerColumn(), TextColumn("[bold cyan]trajectory-analysis"),
            BarColumn(bar_width=40), MofNCompleteColumn(), TaskProgressColumn(),
            TimeElapsedColumn(), TextColumn("[dim]{task.fields[status]}"),
            console=console,
        ) as progress:
            task_id = progress.add_task("Classifying", total=len(trajectories), status="Starting...")
            analyses = list(await asyncio.gather(
                *[_analyze_one(t, progress, task_id) for t in trajectories]
            ))
    else:
        analyses = list(await asyncio.gather(
            *[_analyze_one(t) for t in trajectories]
        ))

    by_instance: dict[str, list[int]] = defaultdict(list)
    for i, traj in enumerate(trajectories):
        by_instance[traj.instance_id].append(i)

    for iid, indices in by_instance.items():
        if len(indices) >= 2:
            group_analyses = [analyses[i] for i in indices]
            group_trajs = [trajectories[i] for i in indices]
            updated = classify_cross_agent(group_analyses, group_trajs)
            for idx, analysis in zip(indices, updated):
                analyses[idx] = analysis

    return analyses


def _build_contamination_context(report: ContaminationReport) -> str:
    lines = []
    lines.append(f"Severity: {report.severity.value}")

    labels = [tl.label.value for tl in report.task_labels]
    if labels:
        lines.append(f"Labels: {', '.join(labels)}")

    ep = report.patch_analysis
    if ep.unrelated_count > 0:
        lines.append(f"OVER_PATCH: {ep.unrelated_count} UNRELATED / {ep.total_hunks} hunks")

    et = report.test_analysis
    if et.off_topic_assertions > 0 or et.unrelated_count > 0:
        lines.append(
            f"OVER_TEST: {et.off_topic_assertions} OFF_TOPIC / "
            f"{et.total_assertions} assertions"
        )

    lines.append(f"Core requirement: {report.intent.core_requirement[:300]}")
    return "\n".join(lines)


def generate_trajectory_summary(
    analyses: list[TrajectoryAnalysis],
) -> str:
    lines = ["## Trajectory Analysis Results\n"]

    pattern_counts: dict[str, int] = defaultdict(int)
    for a in analyses:
        pattern_counts[a.leakage_pattern.value] += 1

    lines.append("### Overview\n")
    lines.append(f"- **Total trajectories analyzed:** {len(analyses)}")
    for pattern, count in sorted(pattern_counts.items()):
        pct = count / max(len(analyses), 1) * 100
        lines.append(f"- **{pattern}:** {count} ({pct:.1f}%)")
    lines.append("")

    by_instance: dict[str, list[TrajectoryAnalysis]] = defaultdict(list)
    for a in analyses:
        by_instance[a.instance_id].append(a)

    lines.append("### Per-Instance Results\n")

    for iid, instance_analyses in sorted(by_instance.items()):
        lines.append(f"#### `{iid}`\n")
        lines.append(
            "| Agent | Pattern | Evidence Strength | "
            "Gold Sim | Pip Installs | Test Refs | Behavior Summary |"
        )
        lines.append("|---|---|---|---|---|---|---|")

        for a in instance_analyses:
            behavior = a.agent_behavior_summary[:100] if a.agent_behavior_summary else ""
            pip_str = str(len(a.pip_install_commands))
            ref_str = str(len(a.test_references))
            lines.append(
                f"| {a.agent_name} | **{a.leakage_pattern.value}** | "
                f"{a.evidence_strength} | {a.gold_patch_similarity:.2f} | "
                f"{pip_str} | {ref_str} | {behavior} |"
            )

        for a in instance_analyses:
            if a.causal_chain:
                lines.append(f"\n**Causal chain ({a.agent_name}):** {a.causal_chain}")
            if a.llm_reasoning:
                lines.append(f"\n<details><summary>LLM reasoning ({a.agent_name})</summary>\n")
                lines.append(a.llm_reasoning)
                lines.append("\n</details>")

        lines.append("")

    return "\n".join(lines)


def compute_leakage_rates(
    analyses: list[TrajectoryAnalysis],
) -> dict[str, dict[str, Any]]:
    by_agent: dict[str, list[TrajectoryAnalysis]] = defaultdict(list)
    for a in analyses:
        by_agent[a.agent_name].append(a)

    rates = {}
    for agent_name, agent_analyses in by_agent.items():
        total = len(agent_analyses)
        genuine = sum(
            1 for a in agent_analyses
            if a.leakage_pattern == LeakagePattern.GENUINE_SOLUTION
        )
        leaked = sum(
            1 for a in agent_analyses
            if a.leakage_pattern in (
                LeakagePattern.GOLD_PATCH_LEAK,
                LeakagePattern.PACKAGE_LEAK,
                LeakagePattern.TEST_AWARE,
            )
        )
        partial = sum(
            1 for a in agent_analyses
            if a.leakage_pattern == LeakagePattern.PARTIAL_MATCH
        )
        mean_sim = (
            sum(a.gold_patch_similarity for a in agent_analyses) / total
            if total > 0 else 0.0
        )

        rates[agent_name] = {
            "total": total,
            "genuine": genuine,
            "leaked": leaked,
            "partial": partial,
            "leakage_rate": leaked / total if total > 0 else 0.0,
            "mean_gold_patch_similarity": round(mean_sim, 4),
        }

    return rates


def generate_narrative(
    report: ContaminationReport,
    record: TaskRecord,
    analyses: list[TrajectoryAnalysis],
) -> str:
    lines = []
    iid = report.instance_id

    lines.append(f"## Contamination Narrative: `{iid}`\n")

    lines.append("### Task Context\n")
    lines.append(f"**Repository:** `{record.repo}` (version {record.version})")
    lines.append(f"**Core requirement:** {report.intent.core_requirement}")
    lines.append(f"**Severity:** {report.severity.value}")
    labels = ", ".join(f"`{tl.label.value}`" for tl in report.task_labels)
    if labels:
        lines.append(f"**Labels:** {labels}")
    lines.append("")

    lines.append("### Contamination Signals\n")
    ep = report.patch_analysis
    et = report.test_analysis
    if ep.unrelated_count > 0:
        lines.append(
            f"- **OVER_PATCH:** {ep.unrelated_count} of "
            f"{ep.total_hunks} hunks are UNRELATED to the stated problem"
        )
    if et.off_topic_assertions > 0 or et.unrelated_count > 0:
        parts = []
        if et.off_topic_assertions > 0:
            pct = et.off_topic_assertions / max(et.total_assertions, 1) * 100
            parts.append(f"{et.off_topic_assertions} of {et.total_assertions} assertions ({pct:.0f}%) are OFF_TOPIC")
        if et.unrelated_count > 0:
            parts.append(f"{et.unrelated_count} UNRELATED tests")
        for part in parts:
            lines.append(f"- **OVER_TEST:** {part}")
    if report.description_clarity.score > 0.3:
        lines.append(f"- **UNCLEAR_DESCRIPTION:** Problem statement has significant ambiguity ({report.description_clarity.score:.2f})")
    lines.append("")

    if report.task_labels:
        lines.append("### Label Analysis\n")
        for tl in report.task_labels:
            lines.append(f"**{tl.label.value}**:")
            if tl.reasoning:
                lines.append(f"  {tl.reasoning}")
            if tl.evidence:
                for ev in tl.evidence[:3]:
                    lines.append(f"  - {ev}")
        lines.append("")

    if analyses:
        lines.append("### Agent Evaluation Behavior\n")
        instance_analyses = [a for a in analyses if a.instance_id == iid]

        for a in instance_analyses:
            lines.append(f"#### Agent: {a.agent_name}\n")
            lines.append(f"- **Classification:** {a.leakage_pattern.value}")
            lines.append(f"- **Gold patch similarity:** {a.gold_patch_similarity:.1%}")
            if a.pip_install_commands:
                lines.append(f"- **Pip installs:** {', '.join(a.pip_install_commands)}")
            if a.causal_chain:
                lines.append(f"- **Causal chain:** {a.causal_chain}")
            if a.agent_behavior_summary:
                lines.append(f"- **Behavior:** {a.agent_behavior_summary}")
            if a.llm_reasoning:
                lines.append(f"\n> {a.llm_reasoning[:500]}")
            lines.append("")

    lines.append("### Diagnosis\n")
    lines.append(_generate_diagnosis(report, analyses))
    lines.append("")

    lines.append("---\n")
    return "\n".join(lines)


def _generate_diagnosis(
    report: ContaminationReport,
    analyses: list[TrajectoryAnalysis],
) -> str:
    parts = []

    instance_analyses = [a for a in analyses if a.instance_id == report.instance_id]
    leaked_count = sum(
        1 for a in instance_analyses
        if a.leakage_pattern in (
            LeakagePattern.GOLD_PATCH_LEAK,
            LeakagePattern.PACKAGE_LEAK,
            LeakagePattern.TEST_AWARE,
        )
    )
    total = len(instance_analyses)

    if total > 0:
        parts.append(
            f"**Agent impact:** {leaked_count}/{total} analyzed agents showed "
            f"leakage patterns on this task."
        )

    if report.test_analysis.off_topic_assertions > 0:
        parts.append(
            f"**Action:** Remove or quarantine {report.test_analysis.off_topic_assertions} "
            f"OFF_TOPIC assertions from the test patch."
        )
    if report.patch_analysis.unrelated_count > 0:
        parts.append(
            f"**Action:** Review {report.patch_analysis.unrelated_count} UNRELATED "
            f"patch hunks — the gold patch may need to be scoped down."
        )

    return " ".join(parts) if parts else "Task flagged for manual review."


async def run_trajectory_analysis(
    reports_dir: str | pathlib.Path,
    trajectory_source: str,
    output_path: str | pathlib.Path | None = None,
    severity_filter: str = "SEVERE",
    instance_ids: list[str] | None = None,
    agent_name: str = "",
    hf_split: str = "train",
    llm: Any | None = None,
    api_key: str = "",
    model_filter: str = "",
    max_concurrency: int = 10,
) -> str:
    from bench_cleanser.deep_dive import load_reports_from_dir

    reports = load_reports_from_dir(
        reports_dir, severity_filter=severity_filter, instance_ids=instance_ids,
    )
    logger.info("Found %d matching contamination reports", len(reports))

    if not reports:
        return "No matching contamination reports found."

    target_ids = {r.instance_id for r in reports}
    gold_patches: dict[str, str] = {}
    f2p_tests: dict[str, list[str]] = {}
    problem_statements: dict[str, str] = {}
    contamination_reports: dict[str, ContaminationReport] = {}

    # Batch-load tasks from SWE-bench datasets ONCE instead of per-report.
    # load_single_task loads entire datasets per call — catastrophically slow
    # for 100+ reports. Instead, load each dataset once and index by instance_id.
    from bench_cleanser.data_loader import load_swebench_pro, load_swebench_verified

    logger.info("Batch-loading SWE-bench datasets for %d target instances", len(target_ids))
    all_records: dict[str, TaskRecord] = {}
    for loader_fn, label in [
        (load_swebench_pro, "SWE-bench Pro"),
        (load_swebench_verified, "SWE-bench Verified"),
    ]:
        try:
            records_list = loader_fn(max_tasks=10000)
            for rec in records_list:
                if rec.instance_id in target_ids:
                    all_records[rec.instance_id] = rec
            logger.info("Indexed %s: found %d/%d target tasks so far",
                        label, len(all_records), len(target_ids))
            if len(all_records) >= len(target_ids):
                break
        except Exception as exc:
            logger.warning("Failed to load %s: %s", label, exc)

    matched = 0
    for report in reports:
        record = all_records.get(report.instance_id)
        if record:
            gold_patches[report.instance_id] = record.patch
            f2p_tests[report.instance_id] = record.fail_to_pass
            problem_statements[report.instance_id] = record.full_problem_context
            matched += 1
        else:
            logger.warning("No SWE-bench task found for %s — LLM analysis will be skipped", report.instance_id)
        contamination_reports[report.instance_id] = report

    logger.info("Matched %d/%d reports to SWE-bench tasks (problem_statements populated: %d)",
                matched, len(reports), len(problem_statements))

    trajectories = load_trajectories(
        trajectory_source,
        instance_ids=target_ids,
        agent_name=agent_name,
        hf_split=hf_split,
        api_key=api_key,
        model_filter=model_filter,
    )
    logger.info("Loaded %d trajectories for %d target instances",
                len(trajectories), len(target_ids))

    if not trajectories:
        return "No trajectories found for the target instances."

    analyses = await analyze_trajectories(
        trajectories, gold_patches, f2p_tests, problem_statements,
        llm=llm,
        contamination_reports=contamination_reports,
        max_concurrency=max_concurrency,
    )

    summary = generate_trajectory_summary(analyses)

    rates = compute_leakage_rates(analyses)
    summary += "\n### Per-Agent Leakage Rates\n\n"
    summary += "| Agent | Total | Genuine | Leaked | Partial | Leakage Rate | Mean Similarity |\n"
    summary += "|---|---|---|---|---|---|---|\n"
    for agent, stats in sorted(rates.items()):
        summary += (
            f"| {agent} | {stats['total']} | {stats['genuine']} | "
            f"{stats['leaked']} | {stats['partial']} | "
            f"{stats['leakage_rate']:.1%} | "
            f"{stats['mean_gold_patch_similarity']:.3f} |\n"
        )

    # Reuse already-loaded records instead of calling load_single_task again
    records_map: dict[str, TaskRecord] = all_records

    summary += "\n---\n\n# End-to-End Contamination Narratives\n"
    for report in reports:
        if report.instance_id in records_map:
            summary += "\n" + generate_narrative(
                report, records_map[report.instance_id], analyses,
            )

    # Stage 7 — Task-Trajectory Fusion summary.
    fusion_counter: Counter[str] = Counter()
    per_agent_counter: dict[str, Counter[str]] = defaultdict(Counter)
    invalidated = 0
    for a in analyses:
        rep = contamination_reports.get(a.instance_id)
        if rep is None or rep.pipeline_error is not None:
            # Pipeline-error reports carry no analytic signal; fusing
            # against them would produce spurious verdicts.
            continue
        f = fuse(rep, a)
        fusion_counter[f.verdict.value] += 1
        per_agent_counter[a.agent_name][f.verdict.value] += 1
        if f.invalidates_measurement:
            invalidated += 1

    if fusion_counter:
        summary += "\n---\n\n# Stage 7 — Task-Trajectory Fusion\n\n"
        summary += (
            f"Total fused (task, agent) pairs: {sum(fusion_counter.values())}. "
            f"Measurement-invalidating verdicts: **{invalidated}** "
            f"({invalidated / sum(fusion_counter.values()):.1%}).\n\n"
        )
        summary += "## Verdict totals\n\n| Verdict | Count |\n|---|---|\n"
        for v in FusionVerdict:
            if fusion_counter[v.value]:
                summary += f"| {v.value} | {fusion_counter[v.value]} |\n"
        summary += "\n## Per-agent breakdown\n\n"
        verdicts_in_use = [v.value for v in FusionVerdict if fusion_counter[v.value]]
        summary += "| Agent | " + " | ".join(verdicts_in_use) + " |\n"
        summary += "|" + "|".join(["---"] * (len(verdicts_in_use) + 1)) + "|\n"
        for agent in sorted(per_agent_counter):
            row = [agent] + [str(per_agent_counter[agent].get(v, 0)) for v in verdicts_in_use]
            summary += "| " + " | ".join(row) + " |\n"

    if output_path:
        out = pathlib.Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(summary, encoding="utf-8")
        logger.info("Trajectory analysis written to %s", out)

        json_path = pathlib.Path(output_path).with_suffix(".json")

        # Stage 7 fusion: combine Axis 1 (task severity) with Axis 2 (trajectory label).
        fusions: list[dict] = []
        for a in analyses:
            rep = contamination_reports.get(a.instance_id)
            if rep is None or rep.pipeline_error is not None:
                continue
            fusions.append(fuse(rep, a).to_dict())

        json_data = {
            "analyses": [a.to_dict() for a in analyses],
            "leakage_rates": rates,
            "fusion": fusions,
        }
        json_path.write_text(
            json.dumps(json_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return summary
