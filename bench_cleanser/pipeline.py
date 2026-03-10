"""Pipeline orchestrator: wires Stages 1-6 together.

v1: Processes tasks with 7-category taxonomy (Stages 1-6).
v2: Intent-matching pipeline with 4-verdict system (Stages 1-5).
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import os
import pathlib
from typing import Any

import yaml
from tqdm import tqdm

from bench_cleanser.analysis.cross_ref import analyze_cross_references
from bench_cleanser.analysis.patch_analyzer import analyze_patch, analyze_patch_v2
from bench_cleanser.analysis.scope_analyzer import analyze_scope, extract_intent
from bench_cleanser.analysis.structural_diff import compute_structural_diff
from bench_cleanser.analysis.test_analyzer import analyze_tests, analyze_tests_v2
from bench_cleanser.cache import ResponseCache
from bench_cleanser.classification.scorer import build_report, build_report_v2
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
    ContaminationReportV2,
    ParsedTask,
    PipelineConfig,
    Severity,
    TaskRecord,
    VagueSpecDetail,
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


# ------------------------------------------------------------------
# Config loading
# ------------------------------------------------------------------


def load_config(config_path: str) -> PipelineConfig:
    """Load pipeline configuration from a YAML file.

    Environment variable references like ``${LLM_API_KEY}`` are expanded.
    """
    with open(config_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    def _expand(val: Any) -> Any:
        if isinstance(val, str) and "${" in val:
            import re

            def _repl(m: Any) -> str:
                return os.environ.get(m.group(1), m.group(0))

            return re.sub(r"\$\{(\w+)\}", _repl, val)
        return val

    llm = raw.get("llm", {})
    pipeline = raw.get("pipeline", {})
    thresholds = raw.get("thresholds", {})
    astred = raw.get("astred", {})
    code_visit = raw.get("code_visitation", {})

    return PipelineConfig(
        llm_base_url=_expand(llm.get("base_url", "https://cloudgpt-openai.azure-api.net/")),
        llm_api_version=llm.get("api_version", "2025-04-01-preview"),
        llm_model=llm.get("model", "gpt-5.2-20251211"),
        llm_max_tokens=llm.get("max_tokens", 4096),
        llm_reasoning_effort=llm.get("reasoning_effort", "high"),
        max_concurrent_requests=llm.get("max_concurrent_requests", 10),
        retry_attempts=llm.get("retry_attempts", 7),
        retry_delay_seconds=llm.get("retry_delay_seconds", 5.0),
        concurrency=pipeline.get("concurrency", 5),
        cache_dir=pipeline.get("cache_dir", ".cache/llm_responses"),
        output_dir=pipeline.get("output_dir", "output"),
        clean_max=thresholds.get("clean_max", 0.2),
        minor_max=thresholds.get("minor_max", 0.5),
        moderate_max=thresholds.get("moderate_max", 0.8),
        astred_enabled=astred.get("enabled", False),
        astred_binary_path=astred.get("binary_path", ""),
        code_visitation_enabled=code_visit.get("enabled", True),
        repo_cache_dir=code_visit.get("repo_cache_dir", ".cache/repos"),
        clone_timeout_seconds=code_visit.get("clone_timeout_seconds", 120),
        max_source_context_lines=code_visit.get("max_source_context_lines", 200),
    )


# ------------------------------------------------------------------
# Stage 1: Parse
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# Stage 1.5: Code Visitation
# ------------------------------------------------------------------


def enrich_with_code_context(
    parsed: ParsedTask,
    repo_manager: RepoManager,
    config: PipelineConfig,
) -> None:
    """Stage 1.5: clone the repo and attach CodeContext to each F2P test hunk.

    Modifies *parsed.f2p_test_hunks* in place by setting their
    ``code_context`` attribute.
    """
    record = parsed.record
    repo_path = repo_manager.get_repo_path(record.repo, record.base_commit)
    if repo_path is None:
        logger.warning(
            "Code visitation skipped for %s: clone failed", record.instance_id
        )
        return

    max_lines = config.max_source_context_lines

    for test_hunk in parsed.f2p_test_hunks:
        try:
            test_file = test_hunk.file_path

            # Read the full test file from the pre-patch repo
            test_file_content = repo_manager.get_file(repo_path, test_file) or ""

            # Extract pre-patch test function
            pre_patch_source = get_full_test_source(
                repo_path, test_file, test_hunk.test_name, max_lines=max_lines
            )

            # Build post-patch test source from diff
            post_patch_source = get_post_patch_test_source(
                pre_patch_source,
                test_hunk.test_name,
                test_hunk.added_lines,
                test_hunk.removed_lines,
                max_lines=max_lines,
            )

            # Extract imports and fixtures
            imports_text = extract_imports(test_file_content) if test_file_content else ""
            fixtures_text = (
                extract_fixtures(test_file_content, test_hunk.test_name)
                if test_file_content
                else ""
            )

            # Resolve imports to file paths
            import_map = (
                resolve_imports(test_file_content, repo_path)
                if test_file_content
                else {}
            )

            # Identify tested functions (calls into patch files)
            analysis_source = post_patch_source or test_hunk.full_source
            tested_funcs = identify_tested_functions(
                analysis_source,
                import_map,
                parsed.files_in_gold_patch,
                repo_path,
                max_source_lines=max_lines,
            )

            # Build call targets
            call_targets = build_call_targets(
                analysis_source,
                import_map,
                parsed.files_in_gold_patch,
            )

            # Extract assertions
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
                "Code context built for %s: %d tested funcs, %d calls, %d assertions",
                test_hunk.test_name,
                len(tested_funcs),
                len(call_targets),
                len(assertions),
            )

        except Exception as exc:
            logger.warning(
                "Code visitation failed for test %s: %s",
                test_hunk.test_name,
                exc,
            )

    # Also try to build CodeContext for unmatched F2P tests
    for test_id in parsed.f2p_tests_with_no_hunk:
        # Extract file path and test name from test ID
        parts = test_id.rsplit("::", 1)
        if len(parts) != 2:
            continue
        test_file, test_name = parts
        # Only extract test name (strip class if present)
        if "." in test_name:
            test_name = test_name.split(".")[-1]

        pre_patch_source = get_full_test_source(
            repo_path, test_file, test_name, max_lines=max_lines
        )
        if pre_patch_source:
            logger.info(
                "Found pre-patch source for unmatched F2P test %s (%d lines)",
                test_id,
                pre_patch_source.count("\n") + 1,
            )


# ------------------------------------------------------------------
# Single-task pipeline
# ------------------------------------------------------------------


async def process_single_task(
    record: TaskRecord,
    llm: LLMClient,
    config: PipelineConfig,
    repo_manager: RepoManager | None = None,
) -> ContaminationReport:
    """Run the full pipeline on a single task."""
    # Stage 1: Parse
    parsed = parse_task(record)

    # Stage 1.5: Code visitation (if enabled and repo_manager available)
    if config.code_visitation_enabled and repo_manager is not None:
        enrich_with_code_context(parsed, repo_manager, config)

    # Stage 2: Scope analysis (LLM, unanchored)
    scope = await analyze_scope(parsed.record, llm)

    # Stages 3 & 4 can run in parallel
    patch_task = analyze_patch(parsed, scope, llm)
    test_task = analyze_tests(parsed, scope, llm)
    patch_analysis, test_analysis = await asyncio.gather(patch_task, test_task)

    # Stage 5: Cross-reference (pass test hunks for code-context-aware analysis)
    cross_ref = analyze_cross_references(
        patch_analysis, test_analysis,
        f2p_test_hunks=parsed.f2p_test_hunks,
    )

    # Stage 6: Classification
    report = build_report(scope, patch_analysis, test_analysis, cross_ref, config)

    return report


# ------------------------------------------------------------------
# Batch pipeline
# ------------------------------------------------------------------


async def run_pipeline(
    records: list[TaskRecord],
    config: PipelineConfig,
) -> list[ContaminationReport]:
    """Run the pipeline on a batch of tasks with bounded concurrency.

    Reports are written to disk as they complete.
    """
    cache = ResponseCache(config.cache_dir)
    llm = LLMClient(config, cache=cache)

    # Set up repo manager for code visitation (if enabled)
    repo_manager: RepoManager | None = None
    if config.code_visitation_enabled:
        repo_manager = RepoManager(
            cache_dir=config.repo_cache_dir,
            clone_timeout=config.clone_timeout_seconds,
        )
        logger.info("Code visitation enabled — pre-cloning repos")
        clone_results = repo_manager.pre_clone_repos(records)
        logger.info(
            "Pre-clone complete: %d/%d repos available",
            sum(1 for v in clone_results.values() if v),
            len(clone_results),
        )

    output_dir = pathlib.Path(config.output_dir)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(config.concurrency)
    reports: list[ContaminationReport] = []
    progress = tqdm(total=len(records), desc="Processing tasks", unit="task")

    async def _process(record: TaskRecord) -> ContaminationReport:
        async with semaphore:
            try:
                report = await process_single_task(
                    record, llm, config, repo_manager=repo_manager
                )
            except Exception as exc:
                logger.error(
                    "Failed to process %s: %s", record.instance_id, exc,
                    exc_info=True,
                )
                # Return an error report — marked SEVERE so it stands out
                # in summary stats rather than hiding as CLEAN.
                report = ContaminationReport(
                    instance_id=record.instance_id,
                    severity=Severity.SEVERE,
                    total_confidence=0.0,
                    categories={},
                    f2p_test_reports=[],
                    patch_hunk_reports=[],
                    compound_patterns=[],
                    evidence_summary=f"PIPELINE_ERROR: {exc}",
                )

            # Write per-task report
            report_path = reports_dir / f"{record.instance_id}.json"
            report_path.write_text(
                json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            progress.update(1)
            return report

    tasks = [_process(record) for record in records]
    reports = await asyncio.gather(*tasks)

    progress.close()

    # Write aggregate summary
    _write_summary(reports, output_dir)

    return list(reports)


def _write_summary(
    reports: list[ContaminationReport],
    output_dir: pathlib.Path,
) -> None:
    """Write aggregate summary CSV and stats JSON."""
    # CSV
    csv_path = output_dir / "summary.csv"
    header = (
        "instance_id,severity,total_confidence,"
        "C1_OVERTEST,C2_OVERPATCH,C3_SNEAKY_TEST_MOD,"
        "C4_SCOPE_CREEP,C5_TEST_DESC_MISALIGN,"
        "C6_CIRCULAR_DEPENDENCY,C7_AMBIGUOUS_SPEC,compound_patterns"
    )
    lines = [header]
    for r in reports:
        cats = r.categories
        row = [
            r.instance_id,
            r.severity.value,
            f"{r.total_confidence:.4f}",
        ]
        for cat_name in [
            "OVERTEST",
            "OVERPATCH",
            "SNEAKY_TEST_MOD",
            "SCOPE_CREEP",
            "TEST_DESC_MISALIGN",
            "CIRCULAR_DEPENDENCY",
            "AMBIGUOUS_SPEC",
        ]:
            cs = cats.get(cat_name)
            row.append(f"{cs.confidence:.4f}" if cs else "0.0000")
        row.append(";".join(r.compound_patterns) if r.compound_patterns else "")
        lines.append(",".join(row))

    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Stats JSON
    severity_counts = {"CLEAN": 0, "MINOR": 0, "MODERATE": 0, "SEVERE": 0}
    for r in reports:
        severity_counts[r.severity.value] += 1

    stats = {
        "total_tasks": len(reports),
        "severity_distribution": severity_counts,
        "mean_confidence": (
            sum(r.total_confidence for r in reports) / len(reports)
            if reports
            else 0.0
        ),
    }
    stats_path = output_dir / "summary_stats.json"
    stats_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("Summary written to %s", output_dir)
    logger.info("Severity distribution: %s", severity_counts)


# ═══════════════════════════════════════════════════════════════════════
# v2 Pipeline: Intent-matching architecture
# ═══════════════════════════════════════════════════════════════════════


async def process_single_task_v2(
    record: TaskRecord,
    llm: LLMClient,
    config: PipelineConfig,
    repo_manager: RepoManager | None = None,
) -> ContaminationReportV2:
    """Run the v2 pipeline on a single task.

    Stages:
      1. PARSE — extract diffs from gold patch + test patch
      2. INTENT — extract ground truth intent from problem statement
      3. STRUCTURAL DIFF — astred_core-powered structural analysis
      4. INTENT MATCHING — match tests + patches against intent
      5. TRIAGE & REPORT — 4-category scoring + actionable report
    """
    # Stage 1: Parse (same as v1)
    parsed = parse_task(record)

    # Stage 1.5: Code visitation (same as v1, enriches test hunks)
    if config.code_visitation_enabled and repo_manager is not None:
        enrich_with_code_context(parsed, repo_manager, config)

    # Stage 2: Intent extraction (enhanced, returns IntentStatement)
    intent = await extract_intent(record, llm)

    # Stage 3: Structural diff (if repo available)
    structural_diff = None
    if repo_manager is not None:
        repo_path = repo_manager.get_repo_path(record.repo, record.base_commit)
        if repo_path is not None:
            try:
                structural_diff = compute_structural_diff(parsed, repo_path)
            except Exception as exc:
                logger.warning(
                    "Structural diff failed for %s: %s", record.instance_id, exc
                )

    # Stage 4: Intent matching (patch + test analysis in parallel)
    patch_task = analyze_patch_v2(parsed, intent, llm, structural_diff)
    test_task = analyze_tests_v2(parsed, intent, llm, structural_diff)
    excess_patch, excess_test = await asyncio.gather(patch_task, test_task)

    # Stage 5: Triage & report
    vague_spec = VagueSpecDetail(
        score=intent.ambiguity_score,
        reasoning=intent.raw_llm_response[:500] if intent.raw_llm_response else "",
    )

    report = build_report_v2(intent, excess_patch, excess_test, vague_spec, config)
    return report


async def run_pipeline_v2(
    records: list[TaskRecord],
    config: PipelineConfig,
) -> list[ContaminationReportV2]:
    """Run the v2 pipeline on a batch of tasks with rich progress display.

    Reports are written to disk as they complete.
    """
    cache = ResponseCache(config.cache_dir)
    llm = LLMClient(config, cache=cache)

    # Set up repo manager for code visitation (if enabled)
    repo_manager: RepoManager | None = None
    if config.code_visitation_enabled:
        repo_manager = RepoManager(
            cache_dir=config.repo_cache_dir,
            clone_timeout=config.clone_timeout_seconds,
        )
        logger.info("Code visitation enabled — pre-cloning repos")
        clone_results = repo_manager.pre_clone_repos(records)
        logger.info(
            "Pre-clone complete: %d/%d repos available",
            sum(1 for v in clone_results.values() if v),
            len(clone_results),
        )

    output_dir = pathlib.Path(config.output_dir)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(config.concurrency)
    severity_counts = {"CLEAN": 0, "MINOR": 0, "MODERATE": 0, "SEVERE": 0}

    # Try to use rich for progress; fall back to tqdm
    try:
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

    reports: list[ContaminationReportV2] = []

    async def _process(
        record: TaskRecord,
        progress_callback: Any = None,
    ) -> ContaminationReportV2:
        async with semaphore:
            try:
                report = await process_single_task_v2(
                    record, llm, config, repo_manager=repo_manager
                )
            except Exception as exc:
                logger.error(
                    "Failed to process %s: %s", record.instance_id, exc,
                    exc_info=True,
                )
                from bench_cleanser.models import (
                    ExcessPatchDetail,
                    ExcessTestDetail,
                    IntentStatement,
                )
                # Return an error report — marked SEVERE so it stands out.
                dummy_intent = IntentStatement(
                    instance_id=record.instance_id,
                    core_requirement=f"PIPELINE_ERROR: {exc}",
                    behavioral_contract="",
                    acceptance_criteria=[],
                    out_of_scope="",
                    ambiguity_score=0.0,
                    raw_llm_response=f"Pipeline error: {exc}",
                )
                report = build_report_v2(
                    dummy_intent,
                    ExcessPatchDetail(score=0.0, total_hunks=0, required_count=0, ancillary_count=0, unrelated_count=0),
                    ExcessTestDetail(score=0.0, total_tests=0, aligned_count=0, tangential_count=0, unrelated_count=0, total_assertions=0, on_topic_assertions=0, off_topic_assertions=0, has_modified_tests=False),
                    VagueSpecDetail(score=0.0, reasoning=f"PIPELINE_ERROR: {exc}"),
                    config,
                )
                # Override severity — build_report_v2 scores 0.0 as CLEAN,
                # but errors should be visible in summaries.
                report.severity = Severity.SEVERE

            # Write per-task report
            report_path = reports_dir / f"{record.instance_id}.json"
            report_path.write_text(
                json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            severity_counts[report.severity.value] += 1

            if progress_callback is not None:
                progress_callback()

            return report

    if use_rich:
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
        )
        from rich.live import Live
        from rich.console import Console

        console = Console()

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]bench-cleanser v2"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TextColumn("[dim]{task.fields[status]}"),
            console=console,
        ) as progress:
            task_id = progress.add_task(
                "Processing",
                total=len(records),
                status="Starting...",
            )

            def _update_progress():
                status_parts = [
                    f"CLEAN:{severity_counts['CLEAN']}",
                    f"MINOR:{severity_counts['MINOR']}",
                    f"MOD:{severity_counts['MODERATE']}",
                    f"SEV:{severity_counts['SEVERE']}",
                ]
                progress.update(task_id, advance=1, status=" ".join(status_parts))

            tasks = [_process(record, _update_progress) for record in records]
            reports = list(await asyncio.gather(*tasks))
    else:
        # Fallback to tqdm
        progress_bar = tqdm(total=len(records), desc="bench-cleanser v2", unit="task")

        def _update_tqdm():
            progress_bar.update(1)
            progress_bar.set_postfix(severity_counts)

        tasks = [_process(record, _update_tqdm) for record in records]
        reports = list(await asyncio.gather(*tasks))
        progress_bar.close()

    # Write aggregate summary
    _write_summary_v2(reports, output_dir)

    return reports


def _write_summary_v2(
    reports: list[ContaminationReportV2],
    output_dir: pathlib.Path,
) -> None:
    """Write v2 aggregate summary CSV and stats JSON."""
    # CSV
    csv_path = output_dir / "summary.csv"
    fieldnames = [
        "instance_id", "severity", "combined_score",
        "excess_patch_score", "excess_test_score", "vague_spec_score",
        "patch_hunks_total", "patch_required", "patch_ancillary", "patch_unrelated",
        "tests_total", "tests_aligned", "tests_tangential", "tests_unrelated",
        "assertions_total", "assertions_on_topic", "assertions_off_topic",
        "has_modified_test", "recommendations",
    ]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for r in reports:
        writer.writerow({
            "instance_id": r.instance_id,
            "severity": r.severity.value,
            "combined_score": f"{r.combined_score:.4f}",
            "excess_patch_score": f"{r.excess_patch.score:.4f}",
            "excess_test_score": f"{r.excess_test.score:.4f}",
            "vague_spec_score": f"{r.vague_spec.score:.4f}",
            "patch_hunks_total": r.excess_patch.total_hunks,
            "patch_required": r.excess_patch.required_count,
            "patch_ancillary": r.excess_patch.ancillary_count,
            "patch_unrelated": r.excess_patch.unrelated_count,
            "tests_total": r.excess_test.total_tests,
            "tests_aligned": r.excess_test.aligned_count,
            "tests_tangential": r.excess_test.tangential_count,
            "tests_unrelated": r.excess_test.unrelated_count,
            "assertions_total": r.excess_test.total_assertions,
            "assertions_on_topic": r.excess_test.on_topic_assertions,
            "assertions_off_topic": r.excess_test.off_topic_assertions,
            "has_modified_test": r.excess_test.has_modified_tests,
            "recommendations": "; ".join(r.recommendations),
        })

    csv_path.write_text(output.getvalue(), encoding="utf-8")

    # Stats JSON
    severity_counts = {"CLEAN": 0, "MINOR": 0, "MODERATE": 0, "SEVERE": 0}
    for r in reports:
        severity_counts[r.severity.value] += 1

    scores = [r.combined_score for r in reports]
    sorted_scores = sorted(scores)
    n = len(sorted_scores)

    stats = {
        "total_tasks": len(reports),
        "severity_distribution": severity_counts,
        "mean_combined_score": (sum(scores) / n) if n else 0.0,
        "median_combined_score": (
            sorted_scores[n // 2] if n % 2 == 1
            else (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2
        ) if n else 0.0,
        "mean_excess_patch": (
            sum(r.excess_patch.score for r in reports) / n if n else 0.0
        ),
        "mean_excess_test": (
            sum(r.excess_test.score for r in reports) / n if n else 0.0
        ),
        "mean_vague_spec": (
            sum(r.vague_spec.score for r in reports) / n if n else 0.0
        ),
    }
    stats_path = output_dir / "summary_stats.json"
    stats_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("v2 Summary written to %s", output_dir)
    logger.info("Severity distribution: %s", severity_counts)
