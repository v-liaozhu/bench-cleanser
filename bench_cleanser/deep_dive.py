"""Auto-generate deep-dive case study reports from pipeline JSON reports.

Post-processing module that reads completed JSON reports + original dataset
records and produces Case A-D style markdown documents with assertion-level
traceability.
"""

from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass
from datetime import datetime

from bench_cleanser.models import (
    ContaminationReport,
    TaskRecord,
)

logger = logging.getLogger(__name__)

_CASE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass
class DeepDiveContext:
    case_index: int
    report: ContaminationReport
    record: TaskRecord


def _case_letter(index: int) -> str:
    return _CASE_LETTERS[index % 26]


def generate_header(ctx: DeepDiveContext) -> str:
    letter = _case_letter(ctx.case_index)
    rpt = ctx.report
    anchor = f"case-{letter.lower()}-{rpt.instance_id}"
    labels = ", ".join(f"`{tl.label.value}`" for tl in rpt.task_labels) or "CLEAN"
    return (
        f'## Case {letter}: {rpt.instance_id} {{#{anchor}}}\n\n'
        f"**Severity:** {rpt.severity.value} | "
        f"**Labels:** {labels}\n"
    )


def generate_dataset_record_table(ctx: DeepDiveContext) -> str:
    rec = ctx.record
    letter = _case_letter(ctx.case_index)
    lines = [
        f"### {letter}.1 Dataset Record\n",
        "| Field | Value |", "|---|---|",
        f"| instance_id | `{rec.instance_id}` |",
        f"| repo | `{rec.repo}` |",
        f"| version | `{rec.version}` |",
        f"| base_commit | `{rec.base_commit[:12]}` |",
        f"| F2P tests | {len(rec.fail_to_pass)} |",
        f"| P2P tests | {len(rec.pass_to_pass)} |",
        "",
    ]
    return "\n".join(lines)


def generate_problem_statement_section(ctx: DeepDiveContext) -> str:
    letter = _case_letter(ctx.case_index)
    ps = ctx.record.problem_statement.strip()
    parts = [
        f"### {letter}.2 Problem Statement\n",
        f"> {_blockquote(ps[:2000])}\n",
        f"{'*(truncated)*' if len(ps) > 2000 else ''}\n",
    ]
    if ctx.record.requirements:
        req = ctx.record.requirements.strip()
        parts.append(f"**Requirements:**\n")
        parts.append(f"> {_blockquote(req[:2000])}\n")
    if ctx.record.interface:
        iface = ctx.record.interface.strip()
        parts.append(f"**Interface:**\n")
        parts.append(f"> {_blockquote(iface[:2000])}\n")
    return "\n".join(parts)


def generate_hints_section(ctx: DeepDiveContext) -> str:
    letter = _case_letter(ctx.case_index)
    hints = ctx.record.hints_text.strip()
    if not hints:
        return f"### {letter}.3 Hints\n\n*(No hints available)*\n"
    return (
        f"### {letter}.3 Hints\n\n"
        f"<details>\n<summary>Show hints text</summary>\n\n"
        f"> {_blockquote(hints[:2000])}\n\n"
        f"</details>\n"
    )


def generate_gold_patch_section(ctx: DeepDiveContext) -> str:
    letter = _case_letter(ctx.case_index)
    ep = ctx.report.patch_analysis
    patch = ctx.record.patch.strip()

    lines = [f"### {letter}.4 Gold Patch Analysis\n"]

    if ep.hunk_verdicts:
        lines.append("#### Per-Hunk Verdicts\n")
        lines.append("| Hunk | File | Verdict | Evidence | Reasoning |")
        lines.append("|---|---|---|---|---|")
        for h in ep.hunk_verdicts:
            verdict_fmt = f"**{h.verdict.value}**" if h.verdict.value == "UNRELATED" else h.verdict.value
            reason = h.reasoning[:200].replace("|", "\\|") if h.reasoning else ""
            lines.append(f"| {h.hunk_index} | `{h.file_path}` | {verdict_fmt} | {h.evidence_strength} | {reason} |")
        lines.append("")

    lines.append("#### Patch Summary\n")
    lines.append(f"- **Total hunks:** {ep.total_hunks}")
    lines.append(f"- **REQUIRED:** {ep.required_count}")
    lines.append(f"- **ANCILLARY:** {ep.ancillary_count}")
    lines.append(f"- **UNRELATED:** {ep.unrelated_count}")
    lines.append(f"- **Has excess:** {'Yes' if ep.unrelated_count > 0 else 'No'}")
    lines.append("")

    lines.append("<details>")
    lines.append("<summary>Full gold patch diff</summary>\n")
    lines.append("```diff")
    lines.append(patch)
    lines.append("```\n")
    lines.append("</details>\n")

    return "\n".join(lines)


def generate_test_analysis_section(ctx: DeepDiveContext) -> str:
    letter = _case_letter(ctx.case_index)
    et = ctx.report.test_analysis

    lines = [f"### {letter}.5 Test Patch Analysis\n"]

    if et.test_verdicts:
        lines.append("#### Per-Test Verdicts\n")
        lines.append("| Test | Verdict | Modified | Mod Aligned | ON_TOPIC | OFF_TOPIC |")
        lines.append("|---|---|---|---|---|---|")
        for t in et.test_verdicts:
            lines.append(
                f"| `{t.test_name}` | {t.intent_match.value} | "
                f"{'Yes' if t.is_modified else 'No'} | "
                f"{'Yes' if t.modification_aligned else '**No**'} | "
                f"{t.on_topic_count} | {t.off_topic_count} |"
            )
        lines.append(f"| **Total** | | | | **{et.on_topic_assertions}** | **{et.off_topic_assertions}** |")
        lines.append("")

    for t in et.test_verdicts:
        if t.assertion_verdicts:
            lines.append(f"#### `{t.test_name}` — Per-Assertion Verdicts\n")
            lines.append("| # | Assertion | Verdict | Reason |")
            lines.append("|---|---|---|---|")
            for i, a in enumerate(t.assertion_verdicts):
                stmt = a.statement[:120].replace("|", "\\|") if a.statement else ""
                reason = a.reason[:200].replace("|", "\\|") if a.reason else ""
                lines.append(f"| {i} | `{stmt}` | **{a.verdict.value}** | {reason} |")
            lines.append("")

    lines.append("#### Test Summary\n")
    lines.append(f"- **Total tests:** {et.total_tests}")
    lines.append(f"- **ALIGNED:** {et.aligned_count}")
    lines.append(f"- **TANGENTIAL:** {et.tangential_count}")
    lines.append(f"- **UNRELATED:** {et.unrelated_count}")
    lines.append(f"- **Total assertions:** {et.total_assertions}")
    lines.append(f"- **ON_TOPIC:** {et.on_topic_assertions}")
    lines.append(f"- **OFF_TOPIC:** {et.off_topic_assertions}")
    lines.append(f"- **Has modified tests:** {'Yes' if et.has_modified_tests else 'No'}")
    lines.append(f"- **Has excess:** {'Yes' if et.off_topic_assertions > 0 or et.unrelated_count > 0 else 'No'}")
    lines.append("")

    test_patch = ctx.record.test_patch.strip()
    if test_patch:
        lines.append("<details>")
        lines.append("<summary>Full test patch diff</summary>\n")
        lines.append("```diff")
        lines.append(test_patch)
        lines.append("```\n")
        lines.append("</details>\n")

    return "\n".join(lines)


def generate_pipeline_verdict_section(ctx: DeepDiveContext) -> str:
    letter = _case_letter(ctx.case_index)
    rpt = ctx.report
    intent = rpt.intent

    lines = [f"### {letter}.6 Pipeline Verdict Detail\n"]

    lines.append("#### Intent Extraction (Stage 2)\n")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| **core_requirement** | {intent.core_requirement} |")
    lines.append(f"| **behavioral_contract** | {intent.behavioral_contract[:300]} |")
    ac_str = "; ".join(f"({i+1}) {c}" for i, c in enumerate(intent.acceptance_criteria))
    lines.append(f"| **acceptance_criteria** | {ac_str} |")
    lines.append(f"| **out_of_scope** | {intent.out_of_scope} |")
    lines.append(f"| **ambiguity_score** | {intent.ambiguity_score:.2f} |")
    lines.append("")

    lines.append("#### Signal Summary\n")
    lines.append("| Signal | Value |")
    lines.append("|---|---|")
    lines.append(f"| OVER_PATCH | {'Yes' if rpt.patch_analysis.unrelated_count > 0 else 'No'} ({rpt.patch_analysis.unrelated_count} UNRELATED / {rpt.patch_analysis.total_hunks} hunks) |")
    lines.append(f"| OVER_TEST | {'Yes' if rpt.test_analysis.off_topic_assertions > 0 or rpt.test_analysis.unrelated_count > 0 else 'No'} ({rpt.test_analysis.off_topic_assertions} OFF_TOPIC / {rpt.test_analysis.total_assertions} assertions) |")
    lines.append(f"| UNCLEAR_DESCRIPTION | {rpt.description_clarity.score:.2f} |")
    lines.append("")

    if rpt.task_labels:
        lines.append("#### Task Labels\n")
        lines.append("| Label | Evidence |")
        lines.append("|---|---|")
        for tl in rpt.task_labels:
            evidence = "; ".join(tl.evidence[:3]) if tl.evidence else ""
            evidence_clean = evidence[:200].replace("|", "\\|")
            lines.append(f"| **{tl.label.value}** | {evidence_clean} |")
        lines.append("")

    if rpt.recommendations:
        lines.append("#### Recommendations\n")
        for rec in rpt.recommendations:
            lines.append(f"- {rec}")
        lines.append("")

    return "\n".join(lines)


def generate_independent_analysis_section(
    ctx: DeepDiveContext,
    llm_analysis: str | None = None,
) -> str:
    letter = _case_letter(ctx.case_index)
    rpt = ctx.report

    lines = [f"### {letter}.7 Label Analysis\n"]

    if rpt.task_labels:
        for tl in rpt.task_labels:
            lines.append(f"- **{tl.label.value}**: {tl.reasoning}")
        lines.append("")

    lines.append(f"### {letter}.8 Independent Analysis\n")
    if llm_analysis:
        lines.append(llm_analysis)
    else:
        lines.append(_auto_analyze(ctx))

    lines.append("")
    lines.append(f"**Contamination verdict: {rpt.severity.value}**")
    lines.append("")
    lines.append("---\n")
    return "\n".join(lines)


def _auto_analyze(ctx: DeepDiveContext) -> str:
    rpt = ctx.report
    et = rpt.test_analysis
    ep = rpt.patch_analysis

    findings = []

    if ep.unrelated_count > 0:
        findings.append(
            f"**Patch approach divergence:** {ep.unrelated_count} of "
            f"{ep.total_hunks} gold patch hunk(s) are classified as UNRELATED."
        )

    if et.off_topic_assertions > 0:
        pct = et.off_topic_assertions / max(et.total_assertions, 1) * 100
        findings.append(
            f"**Off-topic test assertions:** {et.off_topic_assertions} of "
            f"{et.total_assertions} assertions ({pct:.0f}%) verify behavior beyond scope."
        )

    if et.has_modified_tests:
        misaligned = [t for t in et.test_verdicts if t.is_modified and not t.modification_aligned]
        if misaligned:
            names = ", ".join(f"`{t.test_name}`" for t in misaligned)
            findings.append(f"**Test mutation:** Modified tests with misaligned changes: {names}")
        else:
            findings.append("**Modified existing tests:** Test modifications are aligned with the problem.")

    if et.unrelated_count > 0:
        findings.append(f"**Unrelated tests:** {et.unrelated_count} F2P test(s) are unrelated to the problem.")

    if not findings:
        findings.append("No strong single contamination signal dominates.")

    return "\n\n".join(findings)


def generate_cross_case_synthesis(cases: list[DeepDiveContext]) -> str:
    lines = ["## Cross-Case Synthesis\n"]

    lines.append("### Label Distribution\n")
    lines.append("| Label | Cases | Count |")
    lines.append("|---|---|---|")

    label_cases: dict[str, list[str]] = {}
    for ctx in cases:
        letter = _case_letter(ctx.case_index)
        for tl in ctx.report.task_labels:
            label_cases.setdefault(tl.label.value, []).append(letter)

    for label, case_letters in sorted(label_cases.items(), key=lambda x: -len(x[1])):
        lines.append(f"| **{label}** | {', '.join(case_letters)} | {len(case_letters)} |")
    lines.append("")

    lines.append("### Severity Summary\n")
    lines.append("| Case | Instance | Severity | Labels |")
    lines.append("|---|---|---|---|")

    for ctx in cases:
        letter = _case_letter(ctx.case_index)
        rpt = ctx.report
        labels = ", ".join(tl.label.value for tl in rpt.task_labels) or "CLEAN"
        lines.append(f"| {letter} | `{rpt.instance_id}` | {rpt.severity.value} | {labels} |")
    lines.append("")

    return "\n".join(lines)


def generate_deep_dive_document(
    cases: list[DeepDiveContext],
    title: str = "Deep-Dive Case Studies: SEVERE Contamination Cases",
    dataset_name: str = "SWE-bench Verified",
) -> str:
    now = datetime.now().strftime("%Y-%m-%d")

    parts = []
    parts.append(f"# {title}\n")
    parts.append(f"> **Generated:** {now}")
    parts.append(f"> **Pipeline:** bench-cleanser v1.2")
    parts.append(f"> **Dataset:** {dataset_name}")
    parts.append(f"> **Cases:** {len(cases)} contamination instances with assertion-level traceability")
    parts.append("\n---\n")

    parts.append("## Table of Contents\n")
    for ctx in cases:
        letter = _case_letter(ctx.case_index)
        iid = ctx.record.instance_id
        anchor = f"case-{letter.lower()}-{iid}"
        parts.append(f"{ctx.case_index + 1}. [Case {letter}: {iid}](#{anchor})")
    parts.append("")
    parts.append("---\n")

    for ctx in cases:
        parts.append(generate_header(ctx))
        parts.append(generate_dataset_record_table(ctx))
        parts.append(generate_problem_statement_section(ctx))
        parts.append(generate_hints_section(ctx))
        parts.append(generate_gold_patch_section(ctx))
        parts.append(generate_test_analysis_section(ctx))
        parts.append(generate_pipeline_verdict_section(ctx))
        parts.append(generate_independent_analysis_section(ctx))

    if len(cases) > 1:
        parts.append(generate_cross_case_synthesis(cases))

    return "\n".join(parts)


def load_reports_from_dir(
    reports_dir: str | pathlib.Path,
    severity_filter: str | None = "SEVERE",
    instance_ids: list[str] | None = None,
) -> list[ContaminationReport]:
    reports_path = pathlib.Path(reports_dir)
    reports = []

    for report_file in reports_path.glob("*.json"):
        try:
            data = json.loads(report_file.read_text(encoding="utf-8"))
            report = ContaminationReport.from_dict(data)
            if severity_filter and report.severity.value != severity_filter:
                continue
            if instance_ids and report.instance_id not in instance_ids:
                continue
            reports.append(report)
        except Exception as exc:
            logger.warning("Failed to load report %s: %s", report_file.name, exc)

    reports.sort(key=lambda r: r.severity.value, reverse=True)
    return reports


def load_records_for_reports(
    reports: list[ContaminationReport],
) -> dict[str, TaskRecord]:
    from bench_cleanser.data_loader import load_single_task

    records: dict[str, TaskRecord] = {}
    for report in reports:
        iid = report.instance_id
        try:
            record = load_single_task(iid)
            if record is not None:
                records[iid] = record
        except Exception as exc:
            logger.warning("Failed to load record for %s: %s", iid, exc)

    return records


def build_deep_dive(
    reports_dir: str | pathlib.Path,
    severity_filter: str | None = "SEVERE",
    instance_ids: list[str] | None = None,
    title: str = "Deep-Dive Case Studies: SEVERE Contamination Cases",
    dataset_name: str = "SWE-bench",
) -> str:
    logger.info("Loading reports from %s", reports_dir)
    reports = load_reports_from_dir(reports_dir, severity_filter, instance_ids)
    logger.info("Found %d matching report(s)", len(reports))

    if not reports:
        return f"# {title}\n\nNo matching reports found.\n"

    logger.info("Loading dataset records...")
    records = load_records_for_reports(reports)
    logger.info("Loaded %d record(s)", len(records))

    cases = []
    for i, report in enumerate(reports):
        if report.instance_id not in records:
            continue
        cases.append(DeepDiveContext(case_index=i, report=report, record=records[report.instance_id]))

    return generate_deep_dive_document(cases, title=title, dataset_name=dataset_name)


def _blockquote(text: str) -> str:
    return "\n> ".join(text.split("\n"))
