"""Live pipeline monitor with rich progress bar and severity charts.

Reads completed reports from the output directory without interfering
with the running pipeline.  Refreshes every few seconds.

Usage:
    python monitor_pipeline.py [--output-dir output_pro_v6] [--total 731] [--interval 5]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

# ── colour palette ──────────────────────────────────────────────────
SEV_STYLES = {
    "CLEAN": "bold green",
    "MINOR": "bold yellow",
    "MODERATE": "bold orange3",
    "SEVERE": "bold red",
}

LABEL_STYLES = {
    "clean": "green",
    "approach_lock": "red",
    "over_test": "magenta",
    "over_patch": "cyan",
    "unclear_description": "yellow",
    "hidden_context": "blue",
    "weak_coverage": "bright_yellow",
}

BAR_CHAR = "\u2588"  # full block


def _scan_reports(reports_dir: Path) -> list[dict]:
    """Read all completed JSON reports (read-only scan)."""
    reports = []
    if not reports_dir.exists():
        return reports
    for f in reports_dir.iterdir():
        if f.suffix == ".json":
            try:
                with open(f, encoding="utf-8") as fh:
                    reports.append(json.load(fh))
            except (json.JSONDecodeError, OSError):
                pass  # file still being written — skip
    return reports


def _severity_bar(label: str, count: int, total: int, width: int = 40) -> Text:
    """Build a coloured horizontal bar for a severity bucket."""
    pct = count / total if total else 0
    fill = int(pct * width)
    style = SEV_STYLES.get(label, "white")
    bar = Text()
    bar.append(f"{label:>9s} ", style="bold")
    bar.append(BAR_CHAR * fill, style=style)
    bar.append(" " * (width - fill))
    bar.append(f" {count:>4d}  ({pct:5.1%})", style=style)
    return bar


def _label_bar(label: str, count: int, max_count: int, width: int = 30) -> Text:
    """Build a coloured horizontal bar for a label bucket."""
    fill = int((count / max_count) * width) if max_count else 0
    style = LABEL_STYLES.get(label, "white")
    bar = Text()
    bar.append(f"{label:>16s} ", style="bold")
    bar.append(BAR_CHAR * fill, style=style)
    bar.append(" " * (width - fill))
    bar.append(f" {count:>3d}", style=style)
    return bar


def _build_severity_panel(sev_counts: dict, total: int) -> Panel:
    """Severity distribution panel with horizontal bars."""
    lines = Text()
    for sev in ["CLEAN", "MINOR", "MODERATE", "SEVERE"]:
        lines.append_text(_severity_bar(sev, sev_counts.get(sev, 0), total))
        lines.append("\n")
    return Panel(lines, title="[bold]Severity Distribution[/bold]", border_style="blue")


def _build_label_panel(label_counts: Counter) -> Panel:
    """Label distribution panel with horizontal bars."""
    if not label_counts:
        return Panel("[dim]No labels yet[/dim]", title="[bold]Label Distribution[/bold]")
    max_c = max(label_counts.values()) if label_counts else 1
    ordered = [
        "approach_lock", "over_test", "over_patch",
        "unclear_description", "hidden_context", "weak_coverage", "clean",
    ]
    lines = Text()
    for lab in ordered:
        c = label_counts.get(lab, 0)
        if c > 0:
            lines.append_text(_label_bar(lab, c, max_c))
            lines.append("\n")
    return Panel(lines, title="[bold]Label Distribution[/bold]", border_style="cyan")


def _build_repo_table(reports: list[dict]) -> Panel:
    """Top repos by report count."""
    repo_counter: Counter = Counter()
    repo_sev: dict[str, Counter] = {}
    for r in reports:
        repo = r.get("repo", "unknown")
        repo_counter[repo] += 1
        if repo not in repo_sev:
            repo_sev[repo] = Counter()
        repo_sev[repo][r["severity"]] += 1

    table = Table(show_header=True, header_style="bold", expand=True, show_lines=False)
    table.add_column("Repository", style="bold white", ratio=3)
    table.add_column("Done", justify="right", ratio=1)
    table.add_column("CLEAN", justify="right", style="green", ratio=1)
    table.add_column("MINOR", justify="right", style="yellow", ratio=1)
    table.add_column("MOD", justify="right", style="orange3", ratio=1)
    table.add_column("SEV", justify="right", style="red", ratio=1)

    for repo, count in repo_counter.most_common(12):
        sc = repo_sev[repo]
        table.add_row(
            repo,
            str(count),
            str(sc.get("CLEAN", 0)),
            str(sc.get("MINOR", 0)),
            str(sc.get("MODERATE", 0)),
            str(sc.get("SEVERE", 0)),
        )

    return Panel(table, title="[bold]Top Repositories[/bold]", border_style="magenta")


def _build_recent_table(reports: list[dict]) -> Panel:
    """Last N completed reports."""
    # Sort by file modification time proxy — use instance_id as fallback
    recent = sorted(reports, key=lambda r: r.get("instance_id", ""))[-8:]
    recent.reverse()

    table = Table(show_header=True, header_style="bold", expand=True, show_lines=False)
    table.add_column("Instance", ratio=5, no_wrap=True, overflow="ellipsis", max_width=60)
    table.add_column("Severity", justify="center", ratio=1)
    table.add_column("Labels", ratio=3, no_wrap=True, overflow="ellipsis")

    for r in recent:
        iid = r.get("instance_id", "?")
        # Truncate instance_id for display
        short_id = iid.replace("instance_", "").replace("__", "/")
        if len(short_id) > 55:
            short_id = short_id[:52] + "..."

        sev = r["severity"]
        sev_style = SEV_STYLES.get(sev, "white")

        labels = [la["label"] for la in r.get("task_labels", []) if la["label"] != "clean"]
        label_str = ", ".join(labels) if labels else "[green]clean[/green]"

        table.add_row(short_id, f"[{sev_style}]{sev}[/{sev_style}]", label_str)

    return Panel(table, title="[bold]Recent Reports[/bold]", border_style="yellow")


def _build_display(
    reports: list[dict],
    total_expected: int,
    progress: Progress,
    task_id,
    start_time: float,
) -> Group:
    """Build the full dashboard layout."""
    n = len(reports)

    # Update progress bar
    progress.update(task_id, completed=n)

    # Compute stats
    sev_counts: dict[str, int] = {"CLEAN": 0, "MINOR": 0, "MODERATE": 0, "SEVERE": 0}
    label_counts: Counter = Counter()
    for r in reports:
        sev_counts[r["severity"]] += 1
        for la in r.get("task_labels", []):
            label_counts[la["label"]] += 1

    elapsed = time.time() - start_time
    rate = n / elapsed if elapsed > 0 else 0
    remaining = (total_expected - n) / rate if rate > 0 else 0

    # Header
    header = Text()
    header.append("  bench-cleanser ", style="bold white on blue")
    header.append("  Pipeline Monitor  ", style="bold")
    header.append(f"  {n}/{total_expected}", style="bold cyan")
    header.append(f"  |  {rate:.1f} reports/min", style="dim")
    if remaining > 0 and n < total_expected:
        mins = int(remaining // 60)
        secs = int(remaining % 60)
        header.append(f"  |  ~{mins}m{secs:02d}s remaining", style="dim")
    header.append("\n")

    return Group(
        header,
        progress,
        Text(""),
        _build_severity_panel(sev_counts, n),
        _build_label_panel(label_counts),
        _build_repo_table(reports),
        _build_recent_table(reports),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor bench-cleanser pipeline run")
    parser.add_argument("--output-dir", default="output_pro_v6", help="Reports output directory")
    parser.add_argument("--total", type=int, default=731, help="Expected total tasks")
    parser.add_argument("--interval", type=float, default=5.0, help="Refresh interval in seconds")
    args = parser.parse_args()

    reports_dir = Path(args.output_dir) / "reports"
    console = Console()

    if not reports_dir.exists():
        console.print(f"[red]Reports directory not found: {reports_dir}[/red]")
        return

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Processing SWE-bench Pro"),
        BarColumn(bar_width=50, complete_style="green", finished_style="bold green"),
        MofNCompleteColumn(),
        TextColumn("[bold]{task.percentage:>5.1f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    )
    task_id = progress.add_task("pipeline", total=args.total)
    start_time = time.time()

    try:
        with Live(console=console, refresh_per_second=1, screen=False) as live:
            while True:
                reports = _scan_reports(reports_dir)
                display = _build_display(reports, args.total, progress, task_id, start_time)
                live.update(display)

                if len(reports) >= args.total:
                    console.print("\n[bold green]Pipeline complete![/bold green]")
                    break

                time.sleep(args.interval)

    except KeyboardInterrupt:
        console.print("\n[yellow]Monitor stopped (pipeline still running)[/yellow]")


if __name__ == "__main__":
    main()
