# Sample run

This directory contains a small, hand-picked slice of a real `bench-cleanser`
pipeline run against SWE-bench Pro. It exists so that someone landing fresh
on the repo can read a real `ContaminationReport` without having to
re-execute the pipeline (which costs LLM calls and Azure credentials).

```
sample_run/
├── reports/                       3 representative ContaminationReport JSONs
└── summary_stats.json             aggregate severity + label distribution
```

The three reports cover different severities so a reader can compare what
`CLEAN`, `MINOR`, and labelled outputs look like in practice.

See the **How to interpret a report** section of the top-level
[README](../../README.md#how-to-interpret-a-report) for a field-by-field walk
through one of these files.

> The full run that produced these (50 SWE-bench Pro tasks, ~22 min wall
> clock against `gpt-5.4-20260305`) is no longer tracked in source control —
> it lives on disk in `output_pro_v6/` if you cloned this repo before v1.0.0.
> The pipeline rebuilds it on demand.
