"""Pipeline orchestrator: wires Stages 1-6 together.

Provides rich terminal progress monitoring with per-stage timing,
severity distribution dashboard, and detailed per-task logging.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import os
import pathlib
import time
from typing import Any

import yaml
from tqdm import tqdm

from bench_cleanser.analysis.cross_ref import analyze_cross_references
from bench_cleanser.analysis.patch_analyzer import analyze_patch
from bench_cleanser.analysis.scope_analyzer import extract_intent
from bench_cleanser.analysis.structural_diff import compute_structural_diff
from bench_cleanser.analysis.test_analyzer import analyze_tests
from bench_cleanser.cache import ResponseCache
from bench_cleanser.classification.scorer import build_report
from bench_cleanser.code_visitor import (
    extract_fixtures,
    extract_imports,
    get_full_test_source,
    get_post_patch_test_source,
)
from bench_cleanser.llm_client import LLMClient
from bench_cleanser.models import (
    CodeContext,
    ContaminationReport,
    DescriptionClarity,
    IntentStatement,
    ParsedTask,
    PatchAnalysis,
    PipelineConfig,
    Severity,
    TaskContaminationLabel,
    TaskRecord,
    TestAnalysis,
)
from bench_cleanser.parsing.patch_parser import get_files_from_patch, parse_patch
from bench_cleanser.parsing.test_parser import (
    match_f2p_tests_to_hunks,
    parse_test_patch,
)
from bench_cleanser.repo_manager import RepoManager
from bench_cleanser.static_analysis import (
    build_call_targets,
    extract_assertions,
    identify_tested_functions,
    resolve_imports,
)

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> PipelineConfig:
    """Load pipeline configuration from a YAML file."""
    with open(config_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    def _expand(val: Any) -> Any:
        if isinstance(val, str) and "${" in val:
            import re
            return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), val)
        return val

    llm = raw.get("llm", {})
    pipeline = raw.get("pipeline", {})
    code_visit = raw.get("code_visitation", {})

    return PipelineConfig(
        llm_base_url=_expand(llm.get("base_url", "https://cloudgpt-openai.azure-api.net/")),
        llm_api_version=llm.get("api_version", "2025-04-01-preview"),
        llm_model=llm.get("model", "gpt-5.4-20260305"),
        llm_max_tokens=llm.get("max_tokens", 65536),
        llm_reasoning_effort=llm.get("reasoning_effort", "high"),
        max_concurrent_requests=llm.get("max_concurrent_requests", 10),
        retry_attempts=llm.get("retry_attempts", 7),
        retry_delay_seconds=llm.get("retry_delay_seconds", 5.0),
        concurrency=pipeline.get("concurrency", 5),
        cache_dir=pipeline.get("cache_dir", ".cache/llm_responses"),
        output_dir=pipeline.get("output_dir", "output"),
        repo_cache_dir=code_visit.get("repo_cache_dir", ".cache/repos"),
        clone_timeout_seconds=code_visit.get("clone_timeout_seconds", 120),
        max_source_context_lines=code_visit.get("max_source_context_lines", 200),
    )


def parse_task(record: TaskRecord) -> ParsedTask:
    """Stage 1: parse a raw task record into structured form."""
    patch_hunks = parse_patch(record.patch)
    test_hunks = parse_test_patch(record.test_patch)
    f2p_matched, f2p_unmatched = match_f2p_tests_to_hunks(
        record.fail_to_pass, test_hunks
    )
    return ParsedTask(
        record=record,
        patch_hunks=patch_hunks,
        test_hunks=test_hunks,
        f2p_test_hunks=f2p_matched,
        f2p_tests_with_no_hunk=f2p_unmatched,
        files_in_gold_patch=get_files_from_patch(record.patch),
        files_in_test_patch=get_files_from_patch(record.test_patch),
    )


def enrich_with_code_context(
    parsed: ParsedTask,
    repo_manager: RepoManager,
    config: PipelineConfig,
) -> None:
    """Stage 1.5: clone the repo and attach CodeContext to each F2P test hunk."""
    record = parsed.record
    repo_path = repo_manager.get_repo_path(record.repo, record.base_commit)
    if repo_path is None:
        logger.warning("Code visitation skipped for %s: clone failed", record.instance_id)
        return

    max_lines = config.max_source_context_lines

    for test_hunk in parsed.f2p_test_hunks:
        try:
            test_file = test_hunk.file_path
            test_file_content = repo_manager.get_file(repo_path, test_file) or ""

            pre_patch_source = get_full_test_source(
                repo_path, test_file, test_hunk.test_name, max_lines=max_lines
            )
            post_patch_source = get_post_patch_test_source(
                pre_patch_source, test_hunk.test_name,
                test_hunk.added_lines, test_hunk.removed_lines,
                max_lines=max_lines,
            )

            imports_text = extract_imports(test_file_content) if test_file_content else ""
            fixtures_text = extract_fixtures(test_file_content, test_hunk.test_name) if test_file_content else ""
            import_map = resolve_imports(test_file_content, repo_path) if test_file_content else {}

            analysis_source = post_patch_source or test_hunk.full_source
            tested_funcs = identify_tested_functions(
                analysis_source, import_map, parsed.files_in_gold_patch,
                repo_path, max_source_lines=max_lines,
            )
            call_targets = build_call_targets(analysis_source, import_map, parsed.files_in_gold_patch)
            assertions = extract_assertions(analysis_source)

            test_hunk.code_context = CodeContext(
                pre_patch_test_source=pre_patch_source,
                post_patch_test_source=post_patch_source,
                test_file_imports=imports_text,
                test_file_fixtures=fixtures_text,
                tested_functions=tested_funcs,
                call_targets=call_targets,
                assertions=assertions,
                test_file_path=test_file,
                repo_path=str(repo_path),
            )

            logger.debug(
                "Code context: %s — %d tested funcs, %d calls, %d assertions",
                test_hunk.test_name, len(tested_funcs), len(call_targets), len(assertions),
            )
        except Exception as exc:
            logger.warning(
                "Code visitation failed for %s: %s",
                test_hunk.test_name,
                exc,
                exc_info=True,
            )


def _log_code_context(parsed: ParsedTask, instance_id: str) -> None:
    """Print code context summary to terminal for visibility."""
    for th in parsed.f2p_test_hunks:
        ctx = th.code_context
        if ctx is None:
            logger.warning("[%s] %s: NO code context", instance_id, th.test_name)
            continue
        logger.info(
            "[%s] %s: %d tested functions, %d call targets, "
            "%d assertions, pre_patch=%s",
            instance_id, th.test_name,
            len(ctx.tested_functions), len(ctx.call_targets),
            len(ctx.assertions),
            "yes" if ctx.pre_patch_test_source else "no",
        )
        for tf in ctx.tested_functions:
            tag = "PATCHED" if tf.is_modified_by_patch else "unpatched"
            logger.info(
                "  -> %s [%s] %s (%d lines)",
                tf.name, tag, tf.file_path,
                len(tf.source.splitlines()) if tf.source else 0,
            )


async def process_single_task(
    record: TaskRecord,
    llm: LLMClient,
    config: PipelineConfig,
    repo_manager: RepoManager | None = None,
) -> ContaminationReport:
    """Run the full pipeline on a single task with per-stage timing.

    Stages:
      1.  PARSE — extract diffs from gold patch + test patch
      1.5 CODE VISITATION — clone repo, attach full test/function source
      2.  INTENT — extract intent from problem statement (blind to patch)
      3.  STRUCTURAL DIFF — AST-level changed block + test block extraction
      4.  INTENT MATCHING — classify patches (4A) and tests (4B) in parallel
      5.  CLASSIFICATION — dual taxonomy labels + bucket severity
    """
    stage_times: dict[str, float] = {}
    iid = record.instance_id

    # Stage 1: PARSE
    t0 = time.monotonic()
    parsed = parse_task(record)
    stage_times["parse"] = time.monotonic() - t0

    # Stage 1.5: CODE VISITATION
    t0 = time.monotonic()
    if repo_manager is not None:
        enrich_with_code_context(parsed, repo_manager, config)
    stage_times["code_visit"] = time.monotonic() - t0

    # Stage 2: INTENT
    t0 = time.monotonic()
    intent = await extract_intent(record, llm, problem_code_context=parsed.problem_code_context)
    stage_times["intent"] = time.monotonic() - t0

    # Stage 3: STRUCTURAL DIFF
    t0 = time.monotonic()
    structural_diff = None
    if repo_manager is not None:
        repo_path = repo_manager.get_repo_path(record.repo, record.base_commit)
        if repo_path is not None:
            try:
                structural_diff = compute_structural_diff(parsed, repo_path)
            except Exception as exc:
                logger.error("Structural diff failed for %s: %s", iid, exc, exc_info=True)
    if structural_diff is None:
        logger.warning(
            "No structural context for %s — intent matching will run without "
            "call graph or function source. Results may be less precise.",
            iid,
        )
    stage_times["structural"] = time.monotonic() - t0

    _log_code_context(parsed, iid)

    # Stage 4: INTENT MATCHING (4A + 4B in parallel)
    t0 = time.monotonic()
    patch_task = analyze_patch(parsed, intent, llm, structural_diff)
    test_task = analyze_tests(parsed, intent, llm, structural_diff)
    patch_analysis, test_analysis = await asyncio.gather(patch_task, test_task)
    stage_times["intent_match"] = time.monotonic() - t0

    # Cross-reference analysis
    t0 = time.monotonic()
    cross_ref = analyze_cross_references(
        patch_analysis=patch_analysis,
        test_analysis=test_analysis,
        f2p_test_hunks=parsed.f2p_test_hunks,
        structural_diff=structural_diff,
    )
    if cross_ref.has_coupling:
        logger.info(
            "[%s] Cross-reference: %d overpatch-overtest coupling(s) detected",
            iid, len(cross_ref.couplings),
        )
    stage_times["cross_ref"] = time.monotonic() - t0

    description_clarity = DescriptionClarity(
        score=intent.ambiguity_score,
        reasoning=intent.behavioral_contract[:500] if intent.behavioral_contract else "",
    )

    # Stage 5: CLASSIFICATION
    t0 = time.monotonic()
    report = await build_report(
        intent, patch_analysis, test_analysis, description_clarity, config,
        record=record, llm=llm, cross_ref=cross_ref,
    )
    stage_times["classify"] = time.monotonic() - t0

    total = sum(stage_times.values())
    logger.info(
        "[%s] %s (%.1fs) | parse=%.1f intent=%.1f struct=%.1f "
        "match=%.1f xref=%.1f classify=%.1f",
        iid, report.severity.value, total,
        stage_times["parse"], stage_times["intent"],
        stage_times["structural"], stage_times["intent_match"],
        stage_times["cross_ref"], stage_times["classify"],
    )

    return report


async def run_pipeline(
    records: list[TaskRecord],
    config: PipelineConfig,
    *,
    resume: bool = True,
) -> list[ContaminationReport]:
    """Run the pipeline on a batch of tasks with progress display.

    Reports are written to disk as they complete. Use resume=True to skip
    tasks with existing reports on disk.
    """
    cache = ResponseCache(config.cache_dir)
    llm = LLMClient(config, cache=cache)

    repo_manager = RepoManager(
        cache_dir=config.repo_cache_dir,
        clone_timeout=config.clone_timeout_seconds,
    )
    logger.info("Pre-cloning repos for code visitation and structural analysis")
    clone_results = repo_manager.pre_clone_repos(records)
    logger.info(
        "Pre-clone complete: %d/%d repos available",
            sum(1 for v in clone_results.values() if v),
            len(clone_results),
        )

    output_dir = pathlib.Path(config.output_dir)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    skipped_ids: set[str] = set()
    if resume:
        existing = {p.stem for p in reports_dir.glob("*.json")}
        skipped_ids = {r.instance_id for r in records if r.instance_id in existing}
        if skipped_ids:
            logger.info("Resume: skipping %d/%d tasks with existing reports", len(skipped_ids), len(records))
        records = [r for r in records if r.instance_id not in skipped_ids]

    semaphore = asyncio.Semaphore(config.concurrency)
    severity_counts = {sev.value: 0 for sev in Severity}

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
            TimeRemainingColumn,
        )
        from rich.table import Table
        use_rich = True
    except ImportError:
        use_rich = False

    reports: list[ContaminationReport] = []
    error_count = 0
    start_time = time.monotonic()

    async def _process(record: TaskRecord, progress_callback: Any = None) -> ContaminationReport:
        nonlocal error_count
        async with semaphore:
            try:
                report = await process_single_task(record, llm, config, repo_manager=repo_manager)
            except Exception as exc:
                error_count += 1
                logger.error("Failed to process %s: %s", record.instance_id, exc, exc_info=True)
                dummy_intent = IntentStatement(
                    instance_id=record.instance_id,
                    core_requirement=f"PIPELINE_ERROR: {exc}",
                    behavioral_contract="", acceptance_criteria=[], out_of_scope="",
                    ambiguity_score=0.0, raw_llm_response=f"Pipeline error: {exc}",
                )
                report = ContaminationReport(
                    instance_id=record.instance_id,
                    severity=Severity.SEVERE,
                    intent=dummy_intent,
                    patch_analysis=PatchAnalysis(total_hunks=0, required_count=0, ancillary_count=0, unrelated_count=0),
                    test_analysis=TestAnalysis(total_tests=0, aligned_count=0, tangential_count=0, unrelated_count=0, total_assertions=0, on_topic_assertions=0, off_topic_assertions=0, has_modified_tests=False),
                    description_clarity=DescriptionClarity(score=0.0, reasoning=f"PIPELINE_ERROR: {exc}"),
                )

            report_path = reports_dir / f"{record.instance_id}.json"
            report_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
            severity_counts[report.severity.value] += 1
            if progress_callback is not None:
                progress_callback()
            return report

    if use_rich:
        console = Console()
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]bench-cleanser"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TextColumn("|"),
            TimeRemainingColumn(),
            TextColumn("|"),
            TextColumn("[dim]{task.fields[status]}"),
            console=console,
        ) as progress:
            task_id = progress.add_task(
                "Processing", total=len(records),
                status="Starting...",
            )
            def _update_progress():
                elapsed = time.monotonic() - start_time
                completed = sum(severity_counts.values())
                rate_per_min = (completed / elapsed * 60) if elapsed > 0 else 0
                status_parts = [
                    f"[green]CLEAN:{severity_counts['CLEAN']}[/green]",
                    f"[yellow]MINOR:{severity_counts['MINOR']}[/yellow]",
                    f"[orange3]MOD:{severity_counts['MODERATE']}[/orange3]",
                    f"[red]SEV:{severity_counts['SEVERE']}[/red]",
                ]
                if error_count:
                    status_parts.append(f"[red bold]ERR:{error_count}[/red bold]")
                status_parts.append(f"[dim]{rate_per_min:.1f}/min[/dim]")
                progress.update(task_id, advance=1, status=" ".join(status_parts))
            tasks = [_process(record, _update_progress) for record in records]
            reports = list(await asyncio.gather(*tasks))

            if skipped_ids:
                for report_path in reports_dir.glob("*.json"):
                    if report_path.stem in skipped_ids:
                        try:
                            data = json.loads(report_path.read_text(encoding="utf-8"))
                            resumed = ContaminationReport.from_dict(data)
                            reports.append(resumed)
                            severity_counts[resumed.severity.value] += 1
                        except Exception as exc:
                            logger.warning(
                                "Failed to load resumed report %s: %s",
                                report_path.stem,
                                exc,
                            )

        # Print final summary table
        final_table = Table(title="Pipeline Complete", show_header=True, header_style="bold")
        final_table.add_column("Metric", style="bold")
        final_table.add_column("Value", justify="right")
        total_time = time.monotonic() - start_time
        final_table.add_row("Total tasks", str(len(reports)))
        final_table.add_row("Elapsed", f"{total_time:.1f}s")
        final_table.add_row("Rate", f"{len(reports)/total_time*60:.1f}/min" if total_time > 0 else "N/A")
        final_table.add_row("[green]CLEAN[/green]", str(severity_counts["CLEAN"]))
        final_table.add_row("[yellow]MINOR[/yellow]", str(severity_counts["MINOR"]))
        final_table.add_row("[orange3]MODERATE[/orange3]", str(severity_counts["MODERATE"]))
        final_table.add_row("[red]SEVERE[/red]", str(severity_counts["SEVERE"]))
        if error_count:
            final_table.add_row("[red bold]ERRORS[/red bold]", str(error_count))
        console.print()
        console.print(final_table)
    else:
        progress_bar = tqdm(total=len(records), desc="bench-cleanser", unit="task")
        def _update_tqdm():
            progress_bar.update(1)
            progress_bar.set_postfix(severity_counts)
        tasks = [_process(record, _update_tqdm) for record in records]
        reports = list(await asyncio.gather(*tasks))
        progress_bar.close()

    if skipped_ids and not use_rich:
        for report_path in reports_dir.glob("*.json"):
            if report_path.stem in skipped_ids:
                try:
                    data = json.loads(report_path.read_text(encoding="utf-8"))
                    resumed = ContaminationReport.from_dict(data)
                    reports.append(resumed)
                    severity_counts[resumed.severity.value] += 1
                except Exception as exc:
                    logger.warning("Failed to load resumed report %s: %s", report_path.stem, exc)

    _write_summary(reports, output_dir)
    return reports


def _write_summary(
    reports: list[ContaminationReport],
    output_dir: pathlib.Path,
) -> None:
    """Write aggregate summary CSV and stats JSON."""
    csv_path = output_dir / "summary.csv"
    fieldnames = [
        "instance_id", "severity",
        "task_labels", "primary_label", "label_count",
        "patch_hunks_total", "patch_unrelated", "has_unrelated_hunks",
        "tests_total", "tests_unrelated", "has_off_topic_assertions",
        "has_modified_tests", "clarity_score",
        "legitimacy", "suggested_fix",
        "mentioned_entities",
        "recommendations",
    ]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    LABEL_PRIORITY = [
        TaskContaminationLabel.APPROACH_LOCK,
        TaskContaminationLabel.OVER_TEST,
        TaskContaminationLabel.OVER_PATCH,
        TaskContaminationLabel.UNCLEAR_DESCRIPTION,
        TaskContaminationLabel.HIDDEN_CONTEXT,
        TaskContaminationLabel.WEAK_COVERAGE,
        TaskContaminationLabel.CLEAN,
    ]

    for r in reports:
        primary = min(r.task_labels, key=lambda tl: LABEL_PRIORITY.index(tl.label)).label.value if r.task_labels else ""

        decomp = r.intent.decomposition
        entities = []
        if decomp:
            for name in decomp.mentioned_files + decomp.mentioned_functions + decomp.mentioned_classes:
                if name and name not in entities:
                    entities.append(name)

        writer.writerow({
            "instance_id": r.instance_id,
            "severity": r.severity.value,
            "task_labels": ";".join(tl.label.value for tl in r.task_labels),
            "primary_label": primary,
            "label_count": len(r.task_labels),
            "patch_hunks_total": r.patch_analysis.total_hunks,
            "patch_unrelated": r.patch_analysis.unrelated_count,
            "has_unrelated_hunks": r.patch_analysis.unrelated_count > 0,
            "tests_total": r.test_analysis.total_tests,
            "tests_unrelated": r.test_analysis.unrelated_count,
            "has_off_topic_assertions": r.test_analysis.off_topic_assertions > 0 or r.test_analysis.unrelated_count > 0,
            "has_modified_tests": r.test_analysis.has_modified_tests,
            "clarity_score": f"{r.description_clarity.score:.4f}",
            "legitimacy": decomp.legitimacy if decomp else "",
            "suggested_fix": (decomp.suggested_fix[:200] if decomp and decomp.suggested_fix else ""),
            "mentioned_entities": ";".join(entities[:15]),
            "recommendations": "; ".join(r.recommendations),
        })

    csv_path.write_text(output.getvalue(), encoding="utf-8")

    severity_counts: dict[str, int] = {"CLEAN": 0, "MINOR": 0, "MODERATE": 0, "SEVERE": 0}
    for r in reports:
        severity_counts[r.severity.value] += 1

    label_counts: dict[str, int] = {}
    for r in reports:
        for tl in r.task_labels:
            label_counts[tl.label.value] = label_counts.get(tl.label.value, 0) + 1

    stats = {
        "total_tasks": len(reports),
        "severity_distribution": severity_counts,
        "label_distribution": label_counts,
    }
    stats_path = output_dir / "summary_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Summary written to %s", output_dir)
    logger.info("Severity: %s", severity_counts)
    logger.info("Labels: %s", label_counts)
