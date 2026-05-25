"""Consolidated CLI entry points for bench-cleanser.

Three console scripts (declared in ``pyproject.toml [project.scripts]``):

* ``bench-cleanser``              -> :func:`main`             (contamination pipeline)
* ``bench-cleanser-trajectory``   -> :func:`trajectory_main`  (Stage 7 fusion + leakage)
* ``bench-cleanser-deep-dive``    -> :func:`deep_dive_main`   (forensic markdown)

The legacy ``run_pipeline.py`` / ``run_trajectory_analysis.py`` / ``run_deep_dive.py``
files at the repo root are now thin shims that delegate to the functions in this
module. Prefer the console-script names — they survive package upgrades.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import pathlib
import sys
from collections import Counter


def _parse_pipeline_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="bench-cleanser",
        description="SWE-bench benchmark contamination detector",
    )
    p.add_argument(
        "--config", default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)",
    )
    p.add_argument(
        "--dataset", choices=["verified", "pro", "live", "both"], default="verified",
        help="Which SWE-bench dataset(s) to analyse (default: verified)",
    )
    p.add_argument(
        "--max-tasks", type=int, default=500,
        help="Maximum tasks per dataset (default: 500)",
    )
    p.add_argument(
        "--instance-id", default=None,
        help="Analyse a single instance by ID (overrides --dataset)",
    )
    p.add_argument(
        "--output", default=None,
        help="Output directory (overrides config file setting)",
    )
    p.add_argument(
        "--concurrency", type=int, default=None,
        help="Number of tasks to process in parallel (overrides config)",
    )
    p.add_argument(
        "--split", default=None,
        help="Dataset split for SWE-bench Live (e.g., test, verified, full)",
    )
    p.add_argument(
        "--resume", action="store_true", default=True,
        help="Resume from checkpoint — skip tasks with existing reports (default)",
    )
    p.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="Reprocess all tasks even if reports already exist",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    return p.parse_args()


def _print_pipeline_summary(reports: list) -> None:
    severity_counts = {"CLEAN": 0, "MINOR": 0, "MODERATE": 0, "SEVERE": 0}
    label_counter: Counter = Counter()
    for r in reports:
        severity_counts[r.severity.value] += 1
        for la in r.task_labels:
            label_counter[la.label.value] += 1

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()

        sev_table = Table(title="bench-cleanser Results", show_header=True)
        sev_table.add_column("Severity", style="bold")
        sev_table.add_column("Count", justify="right")
        sev_table.add_column("Percentage", justify="right")

        colors = {"CLEAN": "green", "MINOR": "yellow",
                  "MODERATE": "orange3", "SEVERE": "red"}
        for sev, count in severity_counts.items():
            pct = (count / len(reports) * 100) if reports else 0
            sev_table.add_row(
                f"[{colors[sev]}]{sev}[/{colors[sev]}]",
                str(count), f"{pct:.1f}%",
            )

        console.print()
        console.print(sev_table)

        if label_counter:
            label_table = Table(title="Task Label Distribution", show_header=True)
            label_table.add_column("Label", style="bold")
            label_table.add_column("Count", justify="right")
            for label, count in label_counter.most_common():
                label_table.add_row(label, str(count))
            console.print()
            console.print(label_table)

        console.print(f"\n  Total tasks: {len(reports)}")

    except ImportError:
        print("\n=== bench-cleanser results ===")
        print(f"Total tasks analysed: {len(reports)}")
        for sev, count in severity_counts.items():
            pct = (count / len(reports) * 100) if reports else 0
            print(f"  {sev:10s}: {count:4d}  ({pct:.1f}%)")
        if label_counter:
            print("\nTask Label Distribution:")
            for label, count in label_counter.most_common():
                print(f"  {label:40s}: {count:4d}")


def main() -> None:
    """Console entry point for the contamination-detection pipeline."""
    from bench_cleanser.data_loader import (
        load_all,
        load_single_task,
        load_swebench_live,
        load_swebench_pro,
        load_swebench_verified,
    )
    from bench_cleanser.pipeline import load_config, run_pipeline

    args = _parse_pipeline_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(args.config)
    if args.output:
        config.output_dir = args.output
    if args.concurrency:
        config.concurrency = args.concurrency

    if args.instance_id:
        logging.info("Loading single task: %s", args.instance_id)
        record = load_single_task(args.instance_id)
        if record is None:
            logging.error("Instance %s not found in any dataset", args.instance_id)
            sys.exit(1)
        records = [record]
    else:
        logging.info("Loading dataset: %s (max %d per set)", args.dataset, args.max_tasks)
        if args.dataset == "verified":
            records = load_swebench_verified(max_tasks=args.max_tasks)
        elif args.dataset == "pro":
            records = load_swebench_pro(max_tasks=args.max_tasks)
        elif args.dataset == "live":
            split_kw = {"split": args.split} if args.split else {}
            records = load_swebench_live(max_tasks=args.max_tasks, **split_kw)
        else:
            records = load_all(max_per_dataset=args.max_tasks)

    logging.info("Loaded %d task(s)", len(records))

    reports = asyncio.run(run_pipeline(records, config, resume=args.resume))
    _print_pipeline_summary(reports)

    print(f"Output written to: {config.output_dir}/")


def _parse_trajectory_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="bench-cleanser-trajectory",
        description="Analyze agent trajectories for benchmark leakage patterns",
    )
    p.add_argument("--reports-dir", required=True,
                   help="Path to directory containing JSON contamination reports")
    p.add_argument("--trajectory-source", required=True,
                   help="JSONL file, JSON directory, HuggingFace dataset, "
                        "or Docent collection UUID")
    p.add_argument("--severity", default="SEVERE",
                   choices=["CLEAN", "MINOR", "MODERATE", "SEVERE"],
                   help="Only analyze trajectories for this severity (default: SEVERE)")
    p.add_argument("--instance-ids", nargs="+", default=None,
                   help="Only analyze specific instance IDs")
    p.add_argument("--agent-name", default="",
                   help="Override agent name (useful for HuggingFace sources)")
    p.add_argument("--hf-split", default="train",
                   help="HuggingFace dataset split (default: train)")
    p.add_argument("--docent-api-key", default="",
                   help="Docent API key (or set DOCENT_API_KEY env var)")
    p.add_argument("--model-filter", default="",
                   help="Filter trajectories by model name (Docent sources)")
    p.add_argument("--config", default="config.yaml",
                   help="Path to config YAML for LLM settings (default: config.yaml)")
    p.add_argument("--output", default=None,
                   help="Output markdown file path (default: stdout)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Enable verbose (DEBUG) logging")
    return p.parse_args()


def trajectory_main() -> None:
    """Console entry point for trajectory-leakage + Stage 7 fusion analysis."""
    args = _parse_trajectory_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    llm = None
    max_concurrency = 10
    try:
        from bench_cleanser.cache import ResponseCache
        from bench_cleanser.llm_client import LLMClient
        from bench_cleanser.pipeline import load_config

        config = load_config(args.config)
        cache = ResponseCache(config.cache_dir)
        llm = LLMClient(config, cache=cache)
        max_concurrency = config.max_concurrent_requests
        logging.info("LLM-primary trajectory analysis enabled (%s)", config.llm_model)
    except Exception as exc:
        logging.warning(
            "Failed to initialize LLM client: %s — using heuristic fallback", exc
        )

    from bench_cleanser.trajectory.analyzer import run_trajectory_analysis

    summary = asyncio.run(run_trajectory_analysis(
        reports_dir=args.reports_dir,
        trajectory_source=args.trajectory_source,
        output_path=args.output,
        severity_filter=args.severity,
        instance_ids=args.instance_ids,
        agent_name=args.agent_name,
        hf_split=args.hf_split,
        llm=llm,
        api_key=args.docent_api_key,
        model_filter=args.model_filter,
        max_concurrency=max_concurrency,
    ))

    if not args.output:
        print(summary)
    else:
        print(f"Trajectory analysis written to: {args.output}")


def _parse_deep_dive_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="bench-cleanser-deep-dive",
        description="Generate forensic per-instance markdown from contamination reports",
    )
    p.add_argument("--reports-dir", required=True,
                   help="Path to directory containing JSON reports")
    p.add_argument("--severity", default="SEVERE",
                   choices=["CLEAN", "MINOR", "MODERATE", "SEVERE"],
                   help="Severity filter (default: SEVERE)")
    p.add_argument("--no-filter", action="store_true",
                   help="Include all severities (ignores --severity)")
    p.add_argument("--instance-ids", nargs="+", default=None,
                   help="Only include specific instance IDs")
    p.add_argument("--output", default=None,
                   help="Output markdown file path (default: stdout)")
    p.add_argument("--title",
                   default="Deep-Dive Case Studies: Contamination Cases",
                   help="Document title")
    p.add_argument("--dataset-name", default="SWE-bench",
                   help="Dataset name for the document header")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Enable verbose (DEBUG) logging")
    return p.parse_args()


def deep_dive_main() -> None:
    """Console entry point for forensic deep-dive markdown."""
    args = _parse_deep_dive_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from bench_cleanser.deep_dive import build_deep_dive

    severity_filter = None if args.no_filter else args.severity

    document = build_deep_dive(
        reports_dir=args.reports_dir,
        severity_filter=severity_filter,
        instance_ids=args.instance_ids,
        title=args.title,
        dataset_name=args.dataset_name,
    )

    if args.output:
        output_path = pathlib.Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(document, encoding="utf-8")
        logging.info("Deep-dive report written to %s", output_path)
        print(f"Deep-dive report written to: {output_path}")
    else:
        print(document)
