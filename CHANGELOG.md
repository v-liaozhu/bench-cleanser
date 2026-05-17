# Changelog

All notable changes to `bench-cleanser` are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-05-17

First stable release. `bench-cleanser` ships a deterministic, evidence-grounded
pipeline for detecting contamination in SWE-bench family benchmarks
(Verified / Pro / Live) and a fairness-verdict engine for combining task
contamination with per-agent trajectory analysis.

### Headline guarantees

* **Reproducibility** — every label and every severity bucket is computable
  from the persisted report alone. No floating-point thresholds anywhere in
  the contamination logic; severity is pure set membership over the label set.
* **OpenAI-audit alignment** — the taxonomy maps directly to the categories
  used in OpenAI's April-2026 critique of SWE-bench Verified:
  `APPROACH_LOCK` ↔ "narrow test cases", `OVER_TEST` ↔ "wide test cases".
* **Stage 7 is deterministic** — the (task, agent) fairness verdict is a pure
  rule engine, no LLM call. The `invalidates_measurement` flag is the
  reproducible answer to *"did this benchmark row measure capability?"*

### Pipeline

* **7-stage architecture** — parse → code visitation → intent extraction →
  structural diff → patch/test/cross-ref matching → dual taxonomy
  classification → report build with bucket severity. Plus Stage 7 fusion
  for per-agent fairness.
* **Dual taxonomy** — Axis 1 (task contamination, 7 binary labels) +
  Axis 2 (agent trajectory, 8 single-label outcomes).
* **PR-authorship realism** — classifier and severity rules encode the
  audit insight that gold patches and F2P tests are co-authored, so
  `OVER_TEST` alone demands maximum attention rather than softer treatment.
* **Cite or shut up** — every assigned label carries explicit evidence
  (hunk indices, assertion indices, problem-text quotes).

### Engineering

* PEP 621 `pyproject.toml` with optional `[trajectory]` and `[structural]`
  extras and console-script entry points (`bench-cleanser`,
  `bench-cleanser-trajectory`, `bench-cleanser-deep-dive`).
* LLM prompts extracted to `bench_cleanser/prompts/*.md` for diffability.
* Trajectory classification now uses the same Pydantic structured-output
  path as task classification — one JSON parser, not two.
* Fusion Rule 4 (`AMBIGUOUS_PASS`) now correctly distinguishes pass-with-
  unknown-trajectory from fail-with-unknown-trajectory.
* Heuristic `OVER_TEST` pre-classification no longer emits duplicate
  candidates when modified-test signals overlap with off-topic signals.
* CLI: `--resume` now defaults to on; opt out with `--no-resume`.
* CI: GitHub Actions runs `ruff check` + `pytest tests/` on Python 3.11 and
  3.12 for every push and PR.
* Tests: expanded from 53 to ~110 covering trajectory, code visitation,
  static analysis, data loading, repo manager, cache, parsers, two
  deterministic E2E pipeline tests, and explicit regressions for every bug
  fixed in 1.0.0.

### Removed

* Tracked generated artefacts (`output_pro_v*`, `output_smoke_v*`,
  `output_v3`, `audits/`, `case_studies/`, `slides/`) — kept on disk but
  no longer in source control. The pipeline rebuilds them on demand and
  `examples/sample_run/` ships a small representative artefact.
* Dead `thresholds:` block in `config.yaml` (contradicted the "no float
  thresholds" design principle and was unread by any code path).
* `sys.path` hack for importing `cloudgpt_aoai`; now a regular package
  module at `bench_cleanser/_internal/cloudgpt.py`.
* `docs/README_v2_archive.md` (referenced renamed taxonomy labels).

[1.0.0]: https://github.com/v-liaozhu/bench-cleanser/releases/tag/v1.0.0
