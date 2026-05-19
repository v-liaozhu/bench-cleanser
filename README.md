# bench-cleanser

Automated contamination, fairness, and trajectory-leakage analysis for SWE-bench family benchmarks (Verified, Pro, Live).

[![CI](https://github.com/v-liaozhu/bench-cleanser/actions/workflows/ci.yml/badge.svg)](https://github.com/v-liaozhu/bench-cleanser/actions/workflows/ci.yml)

`bench-cleanser` evaluates benchmark quality, not just model outcomes. It produces deterministic, evidence-backed outputs that separate:

1. **Task contamination** (Axis 1): what is wrong with the benchmark item itself.
2. **Agent trajectory behavior** (Axis 2): how an agent reached its result.
3. **Fusion verdict** (Stage 7): whether a `(task, agent)` outcome is a fair capability measurement.

---

## State of the union

This repository is production-oriented and cleanup-hardened:

- Deterministic severity and fusion engines (no LLM in Stage 6 severity bucketing or Stage 7 verdicting).
- Strict structured-output contracts for LLM stages via Pydantic schemas.
- Retry, cache, and clone-path robustness improvements in core runtime.
- Expanded regression coverage for fusion and heuristic taxonomy branches.
- Docs aligned with implemented rules (taxonomy, severity, fusion behavior).
- CI quality gates for lint + tests + type checking.

Design target: **research-grade reproducibility** + **industry-grade operational hygiene**.

---

## Core guarantees

- **Evidence-first labels**: every contamination label includes concrete evidence in report JSON.
- **Deterministic severity**: severity is set-membership over Axis-1 labels.
- **Deterministic fairness verdicts**: Stage 7 fusion is a pure rule engine.
- **Schema-enforced LLM calls**: structured output must satisfy declared Pydantic schemas.
- **Reproducible outputs**: summary files are rebuilt directly from persisted report artifacts.

---

## Architecture

```text
TaskRecord
  -> Stage 1   parse patch/test diffs
  -> Stage 1.5 code visitation + repo context
  -> Stage 2   intent extraction (LLM, blind to gold patch)
  -> Stage 3   structural diff (astred_core or stdlib ast fallback)
  -> Stage 4A  patch intent matching
  -> Stage 4B  test/assertion intent matching
  -> Stage 4C  cross-reference coupling analysis
  -> Stage 5   dual taxonomy classification (Axis 1)
  -> Stage 6   report build + severity bucket

ContaminationReport + TrajectoryRecord
  -> Stage 7   deterministic fusion verdict per (task, agent)
```

Primary modules:

- `bench_cleanser/pipeline.py` — Stage 1–6 orchestration.
- `bench_cleanser/classification/dual_taxonomy.py` — Axis-1 labels + severity logic.
- `bench_cleanser/trajectory/classifier.py` — Axis-2 trajectory classification.
- `bench_cleanser/fusion.py` — Stage-7 deterministic fairness rules.

---

## Install

```bash
git clone https://github.com/v-liaozhu/bench-cleanser.git
cd bench-cleanser
pip install -e ".[dev,trajectory]"
```

Optional extras:

- `.[structural]` for `astred-core` structural backend.
- `.[trajectory]` for Docent trajectory ingestion.

---

## CLI

Public entry points:

- `bench-cleanser` — contamination pipeline.
- `bench-cleanser-trajectory` — trajectory analysis + Stage 7 fusion.
- `bench-cleanser-deep-dive` — per-instance forensic markdown.

Examples:

```bash
bench-cleanser --dataset pro --max-tasks 50 --output out/pro
bench-cleanser-trajectory --reports-dir out/pro/reports --trajectory-source <jsonl|HF|docent-uuid>
bench-cleanser-deep-dive --reports-dir out/pro/reports --severity SEVERE
```

Compatibility shims (`run_pipeline.py`, `run_trajectory_analysis.py`, `run_deep_dive.py`) remain for legacy invocation, but console scripts are canonical.

---

## Quality controls

Run locally before merging:

```bash
ruff check bench_cleanser tests
mypy bench_cleanser
pytest tests/ -q
```

CI (`.github/workflows/ci.yml`) runs quality checks on Python 3.11 and 3.12.

---

## Repository hygiene policy

- Generated outputs and audit artifacts are not source of truth; keep them untracked.
- `tools/` contains operator/developer scripts, not stable public APIs.
- Prompts live in `bench_cleanser/prompts/*.md` and are versioned as code.
- Line endings are normalized via `.gitattributes`.

---

## Documentation index

- [`docs/TAXONOMY.md`](docs/TAXONOMY.md) — Axis-1/Axis-2 labels, evidence rules, severity mapping.
- [`docs/FUSION.md`](docs/FUSION.md) — Stage-7 rules and verdict matrix.
- [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) — dev workflow and extension checklists.
- [`CHANGELOG.md`](CHANGELOG.md) — release history.

---

## Limits and non-goals

- `CLEAN` means no contamination signals were detected by these axes; it does not guarantee absence of all benchmark defects.
- Label quality is still bounded by LLM classification quality in labeling stages.
- Runtime/harness flakiness and broader reward-hacking concerns remain partly out of scope.

---

## License

MIT. See [LICENSE](LICENSE).
