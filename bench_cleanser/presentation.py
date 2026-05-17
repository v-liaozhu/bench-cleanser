"""Generate MARP markdown slide decks from bench-cleanser analysis outputs."""

from __future__ import annotations

import json
import logging
import pathlib
from collections import defaultdict
from datetime import datetime
from typing import Any

from bench_cleanser.models import ContaminationReport, Severity

logger = logging.getLogger(__name__)


def generate_slide_deck(
    reports: list[ContaminationReport],
    deep_dive_path: str | pathlib.Path | None = None,
    trajectory_path: str | pathlib.Path | None = None,
    title: str = "SWE-bench Contamination Analysis",
    subtitle: str = "bench-cleanser Findings",
    author: str = "",
) -> str:
    slides = []
    slides.append(_frontmatter(title))
    slides.append(_title_slide(title, subtitle, author))
    slides.append(_problem_slide())
    slides.append(_methodology_slide())
    slides.append(_taxonomy_slide())

    severity_dist = _compute_severity_distribution(reports)
    slides.append(_dataset_summary_slide(reports, severity_dist))
    slides.append(_label_distribution_slide(reports))

    severe_reports = [r for r in reports if r.severity == Severity.SEVERE]
    for i, report in enumerate(severe_reports[:8]):
        slides.append(_case_highlight_slide(report, i))

    if severe_reports:
        slides.append(_pattern_slide(severe_reports))

    if trajectory_path:
        trajectory_data = _load_trajectory_data(trajectory_path)
        if trajectory_data:
            slides.append(_trajectory_slide(trajectory_data))
            slides.append(_agent_impact_slide(trajectory_data))

    slides.append(_sensitivity_slide(reports, severity_dist))
    slides.append(_recommendations_slide(severe_reports, reports))
    slides.append(_appendix_methodology_slide())
    slides.append(_appendix_severity_rules_slide())

    return "\n".join(slides)


def _frontmatter(title: str) -> str:
    return f"""---
marp: true
theme: default
paginate: true
title: {title}
math: katex
style: |
  section {{
    font-size: 22px;
  }}
  h1 {{
    color: #1a1a2e;
  }}
  h2 {{
    color: #16213e;
  }}
  table {{
    font-size: 16px;
  }}
  .severe {{ color: #e63946; font-weight: bold; }}
  .moderate {{ color: #f4a261; font-weight: bold; }}
  .minor {{ color: #e9c46a; }}
  .clean {{ color: #2a9d8f; }}
  .label {{ color: #6c5ce7; font-weight: bold; }}
  blockquote {{ font-size: 18px; }}
---
"""


def _title_slide(title: str, subtitle: str, author: str) -> str:
    now = datetime.now().strftime("%B %Y")
    author_line = f"\n**{author}**" if author else ""
    return f"""
# {title}

## {subtitle}
{author_line}
{now}

---
"""


def _problem_slide() -> str:
    return """
# The Problem: SWE-bench Contamination

SWE-bench evaluates AI agents on real-world software engineering tasks.
However, some tasks have **contaminated evaluation criteria**:

- **APPROACH_LOCK**: Tests require a specific implementation approach not determined by the problem
- **OVER_TEST**: F2P tests verify behavior beyond the stated problem
- **OVER_PATCH**: Gold patch implements changes not described in the issue

> An agent that perfectly solves the stated problem can score **0%** on contaminated tasks.

---
"""


def _methodology_slide() -> str:
    return """
# bench-cleanser: 6-Stage Pipeline

```
┌─────────┐   ┌──────────┐   ┌────────┐   ┌──────────────┐   ┌────────┐   ┌──────────┐
│  PARSE  │──>│   CODE   │──>│ INTENT │──>│    INTENT    │──>│ TRIAGE │──>│ CLASSIFY │
│         │   │VISITATION│   │EXTRACT │   │   MATCHING   │   │& SCORE │   │  LABELS  │
└─────────┘   └──────────┘   └────────┘   └──────────────┘   └────────┘   └──────────┘
 Diff parse    Clone repo     LLM-based    Per-hunk patch     Binary       8 binary
 Hunk split    AST analysis   blind to     Per-assertion      signals      labels +
 F2P match     Source code    gold patch    test verdicts                   bucket sev
```

- **Stage 1-1.5**: Parse diffs, clone repo, extract code context
- **Stage 2**: Extract intent from problem statement (LLM, blind to gold patch)
- **Stage 3**: Structural diff with code visitation (AST-level)
- **Stage 4**: Match each hunk/assertion against intent (LLM)
- **Stage 5-6**: Binary signal triage + 8-label contamination taxonomy

---
"""


def _taxonomy_slide() -> str:
    return """
# Contamination Taxonomy: 7 Binary Labels

| Label | Description |
|---|---|
| <span class="label">APPROACH_LOCK</span> | Tests require specific approach not determined by problem |
| <span class="label">OVER_TEST</span> | Tests verify behavior beyond problem scope |
| <span class="label">OVER_PATCH</span> | Gold patch modifies code beyond problem scope |
| <span class="label">UNCLEAR_DESCRIPTION</span> | Problem description too ambiguous to determine correct fix |
| <span class="label">HIDDEN_CONTEXT</span> | Essential info only in hints, not problem description |
| <span class="label">WEAK_COVERAGE</span> | Tests don't fully cover stated acceptance criteria |
| <span class="label">CLEAN</span> | No contamination detected |

> Labels are **binary** — one instance triggers the same as many.
> **Severity** is determined by which labels are present, not by scores.

---
"""


def _dataset_summary_slide(
    reports: list[ContaminationReport],
    severity_dist: dict[str, int],
) -> str:
    total = len(reports)
    if total == 0:
        return "# Dataset Summary\n\nNo reports available.\n\n---\n"

    table_rows = []
    for sev in ["CLEAN", "MINOR", "MODERATE", "SEVERE"]:
        count = severity_dist.get(sev, 0)
        pct = count / total * 100
        css_class = sev.lower()
        table_rows.append(
            f"| <span class=\"{css_class}\">{sev}</span> | {count} | {pct:.1f}% |"
        )

    contaminated = severity_dist.get("MODERATE", 0) + severity_dist.get("SEVERE", 0)
    contaminated_pct = contaminated / total * 100 if total else 0

    return f"""
# Dataset Summary

**Total tasks analyzed:** {total} | **Contaminated (MODERATE+SEVERE):** {contaminated} ({contaminated_pct:.1f}%)

| Severity | Count | Percentage |
|---|---|---|
{chr(10).join(table_rows)}

---
"""


def _label_distribution_slide(reports: list[ContaminationReport]) -> str:
    if not reports:
        return ""

    label_counts: dict[str, int] = defaultdict(int)
    for r in reports:
        for tl in r.task_labels:
            label_counts[tl.label.value] += 1

    if not label_counts:
        return ""

    rows = []
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = count / len(reports) * 100
        rows.append(f"| <span class=\"label\">{label}</span> | {count} | {pct:.1f}% |")

    total_with_labels = sum(1 for r in reports if r.task_labels)

    return f"""
# Label Distribution

**Tasks with contamination labels:** {total_with_labels}/{len(reports)}

| Label | Count | % of tasks |
|---|---|---|
{chr(10).join(rows)}

> Labels are multi-label — a task can have multiple labels.

---
"""


def _case_highlight_slide(report: ContaminationReport, index: int) -> str:
    iid = report.instance_id
    ep = report.patch_analysis
    et = report.test_analysis

    labels = ", ".join(f"`{tl.label.value}`" for tl in report.task_labels) or "—"

    signals = []
    if ep.unrelated_count > 0:
        signals.append(f"OVER_PATCH: {ep.unrelated_count} UNRELATED / {ep.total_hunks} hunks")
    if et.off_topic_assertions > 0 or et.unrelated_count > 0:
        signals.append(f"OVER_TEST: {et.off_topic_assertions} OFF_TOPIC / {et.total_assertions} assertions")

    core_req = report.intent.core_requirement[:200]

    return f"""
# Case: `{iid}`

**Severity:** <span class="severe">{report.severity.value}</span> | **Labels:** {labels}

**Problem asks:** {core_req}

| Signal | Detail |
|---|---|
| OVER_PATCH | {ep.required_count}R / {ep.ancillary_count}A / {ep.unrelated_count}U of {ep.total_hunks} hunks |
| OVER_TEST | {et.on_topic_assertions} on-topic / {et.off_topic_assertions} off-topic of {et.total_assertions} assertions |
| UNCLEAR_DESCRIPTION | {report.description_clarity.score:.2f} |

---
"""


def _pattern_slide(severe_reports: list[ContaminationReport]) -> str:
    label_cases: dict[str, list[str]] = defaultdict(list)
    for r in severe_reports:
        for tl in r.task_labels:
            label_cases[tl.label.value].append(r.instance_id[:30])

    rows = []
    for label, cases in sorted(label_cases.items(), key=lambda x: -len(x[1])):
        examples = ", ".join(cases[:3])
        if len(cases) > 3:
            examples += "..."
        rows.append(f"| <span class=\"label\">{label}</span> | {len(cases)} | {examples} |")

    return f"""
# Contamination Patterns Across SEVERE Cases

| Label | Count | Example Cases |
|---|---|---|
{chr(10).join(rows)}

**Impact on evaluation:**
1. Agents that correctly solve the stated problem can fail the tests
2. Knowledge of the gold patch or code review is required to pass
3. Leaderboard rankings reflect contamination tolerance, not engineering skill

---
"""


def _trajectory_slide(trajectory_data: dict[str, Any]) -> str:
    rates = trajectory_data.get("leakage_rates", {})
    if not rates:
        return ""

    rows = []
    for agent, stats in sorted(rates.items()):
        rows.append(
            f"| {agent} | {stats['total']} | {stats['genuine']} | "
            f"{stats['leaked']} | {stats['leakage_rate']:.1%} | "
            f"{stats['mean_gold_patch_similarity']:.3f} |"
        )

    return f"""
# Trajectory Analysis: Agent Leakage Rates

| Agent | Total | Genuine | Leaked | Rate | Mean Similarity |
|---|---|---|---|---|---|
{chr(10).join(rows)}

---
"""


def _agent_impact_slide(trajectory_data: dict[str, Any]) -> str:
    analyses = trajectory_data.get("analyses", [])
    if not analyses:
        return ""

    by_instance: dict[str, list[dict]] = defaultdict(list)
    for a in analyses:
        by_instance[a.get("instance_id", "")].append(a)

    cheated_instances = []
    for iid, instance_analyses in by_instance.items():
        leaked = sum(
            1 for a in instance_analyses
            if a.get("leakage_pattern") in ("GOLD_PATCH_LEAK", "PACKAGE_LEAK", "TEST_AWARE")
        )
        if leaked > 0:
            cheated_instances.append((iid, leaked, len(instance_analyses)))

    if not cheated_instances:
        return ""

    rows = []
    for iid, leaked, total in cheated_instances[:6]:
        rows.append(f"| `{iid[:30]}` | {leaked}/{total} | {leaked/total:.0%} |")

    return f"""
# Agent Impact: Leakage on Contaminated Tasks

| Instance | Agents Leaked / Total | Rate |
|---|---|---|
{chr(10).join(rows)}

---
"""


def _sensitivity_slide(
    reports: list[ContaminationReport],
    severity_dist: dict[str, int],
) -> str:
    total = len(reports) or 1
    moderate_plus = severity_dist.get("MODERATE", 0) + severity_dist.get("SEVERE", 0)
    minor_plus = moderate_plus + severity_dist.get("MINOR", 0)

    return f"""
# Severity Distribution

| Severity | Count | Percentage |
|---|---|---|
| <span class="severe">SEVERE</span> | {severity_dist.get('SEVERE', 0)} | {severity_dist.get('SEVERE', 0)/total*100:.1f}% |
| <span class="moderate">MODERATE+</span> | {moderate_plus} | {moderate_plus/total*100:.1f}% |
| MINOR+ | {minor_plus} | {minor_plus/total*100:.1f}% |
| <span class="clean">CLEAN</span> | {severity_dist.get('CLEAN', 0)} | {severity_dist.get('CLEAN', 0)/total*100:.1f}% |

---
"""


def _recommendations_slide(
    severe_reports: list[ContaminationReport],
    all_reports: list[ContaminationReport] | None = None,
) -> str:
    n_severe = len(severe_reports)
    n_total = len(all_reports) if all_reports else n_severe

    return f"""
# Recommendations

Based on **{n_severe} SEVERE** and analysis of **{n_total}** total tasks:

1. **Flag contaminated tasks** in SWE-bench evaluation
   - Tasks with APPROACH_LOCK or compound signals should be excluded

2. **Label-specific remediation**
   - APPROACH_LOCK: Accept alternative valid solutions
   - OVER_TEST: Remove off-topic assertions from test patches
   - OVER_PATCH: Scope the gold patch to the stated problem

3. **Trajectory auditing for leaderboard integrity**
   - Check if agents access gold patches via package installs
   - LLM-based trajectory analysis detects subtle cheating patterns

---
"""


def _appendix_methodology_slide() -> str:
    return """
# Appendix: Detailed Methodology

**Per-hunk patch classification:**
- **REQUIRED**: Directly implements the described fix
- **ANCILLARY**: Supports the fix (imports, infrastructure)
- **UNRELATED**: Changes behavior not described in the problem

**Per-assertion test classification:**
- **ON_TOPIC**: Assertion checks behavior described in the problem
- **OFF_TOPIC**: Assertion checks behavior NOT described

**Per-test classification:**
- **ALIGNED**: Test targets the described problem
- **TANGENTIAL**: Test partially targets the problem
- **UNRELATED**: Test doesn't target the described problem

---
"""


def _appendix_severity_rules_slide() -> str:
    return """
# Appendix: Severity Rules (Bucket-Based)

Severity is determined by **which labels are present**, not by scores:

| Severity | Trigger |
|---|---|
| <span class="severe">SEVERE</span> | APPROACH_LOCK present, or OVER_TEST + OVER_PATCH both present |
| <span class="moderate">MODERATE</span> | OVER_TEST alone |
| <span class="minor">MINOR</span> | OVER_PATCH alone, UNCLEAR_DESCRIPTION, HIDDEN_CONTEXT, or WEAK_COVERAGE |
| <span class="clean">CLEAN</span> | No contamination labels |

> One signal triggers the same severity as many — **binary, not graduated**.

---
"""


def _compute_severity_distribution(
    reports: list[ContaminationReport],
) -> dict[str, int]:
    dist: dict[str, int] = {"CLEAN": 0, "MINOR": 0, "MODERATE": 0, "SEVERE": 0}
    for r in reports:
        dist[r.severity.value] += 1
    return dist


def _load_trajectory_data(path: str | pathlib.Path) -> dict[str, Any]:
    p = pathlib.Path(path)
    json_path = p.with_suffix(".json") if p.suffix != ".json" else p
    if json_path.exists():
        try:
            return json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to load trajectory data from %s: %s", json_path, exc)
    return {}


def build_slide_deck(
    reports_dir: str | pathlib.Path,
    deep_dive_path: str | pathlib.Path | None = None,
    trajectory_path: str | pathlib.Path | None = None,
    title: str = "SWE-bench Contamination Analysis",
    subtitle: str = "bench-cleanser Findings",
    author: str = "",
) -> str:
    from bench_cleanser.deep_dive import load_reports_from_dir

    reports = load_reports_from_dir(reports_dir, severity_filter=None)
    logger.info("Loaded %d reports for slide deck", len(reports))

    return generate_slide_deck(
        reports,
        deep_dive_path=deep_dive_path,
        trajectory_path=trajectory_path,
        title=title,
        subtitle=subtitle,
        author=author,
    )
