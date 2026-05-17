# `tools/` — developer scripts

These are *not* part of the `bench_cleanser` package or its public API. They
are loose scripts for one-off operator tasks:

- `monitor_pipeline.py` — tail a running pipeline log and surface live progress.
- `audit.py` — manual audit-tracker helpers used while building the v1.0.0 taxonomy.

If a future change makes one of these load-bearing, promote it into
`bench_cleanser/` proper and wire it through `cli.py` instead of leaving
it as a loose script.
