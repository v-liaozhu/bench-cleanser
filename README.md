# bench-cleanser

> Deterministic, evidence-backed contamination, fairness, and trajectory-leakage analysis for the SWE-bench family of benchmarks (Verified, Pro, Live).

[![CI](https://github.com/v-liaozhu/bench-cleanser/actions/workflows/ci.yml/badge.svg)](https://github.com/v-liaozhu/bench-cleanser/actions/workflows/ci.yml)
[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![ruff](https://img.shields.io/badge/lint-ruff-000000)](https://docs.astral.sh/ruff/)
[![mypy](https://img.shields.io/badge/type--check-mypy-2A6DB2)](https://mypy.readthedocs.io/)

`bench-cleanser` evaluates **benchmark quality**, not just model outcomes. It separates three concerns that benchmark scores routinely conflate:

1. **Task contamination** (Axis 1) — what is wrong with the benchmark item itself.
2. **Agent trajectory behaviour** (Axis 2) — how an agent reached its result.
3. **Fusion verdict** (Stage 7) — whether a `(task, agent)` outcome is a fair capability measurement.

The output of every stage is reproducible, evidence-linked, and machine-readable.

---

## Table of contents

- [Why this exists](#why-this-exists)
- [Design principles](#design-principles)
- [Architecture](#architecture)
- [Taxonomy at a glance](#taxonomy-at-a-glance)
- [Install](#install)
- [Quickstart](#quickstart)
- [CLI reference](#cli-reference)
- [Outputs](#outputs)
- [Configuration](#configuration)
- [Quality controls](#quality-controls)
- [Repository layout](#repository-layout)
- [Limits and non-goals](#limits-and-non-goals)
- [Documentation index](#documentation-index)
- [License](#license)

---

## Why this exists

A model that solves 80 % of a benchmark is not necessarily 80 % capable — it might be 80 % lucky. The lucky 80 % comes from:

- **Tests that leak the solution** (assertions baked into the prompt, package‑hint imports, etc.).
- **Patches that overshoot the brief** (changes the spec never asked for, scored as correct because the harness ran).
- **Agents that read the gold patch out of band**, paste it, and pass.

`bench-cleanser` makes those three failure modes legible. It tags every contaminated task, classifies every agent trajectory, and emits a single deterministic fairness verdict per `(task, agent)` pair so that downstream leaderboards can be filtered honestly.

---

## Design principles

| Principle                       | Concretely means                                                                                                  |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Evidence first**              | Every label carries the concrete spans/lines/commands that triggered it. No floating verdicts.                    |
| **Deterministic where it can**  | Severity bucketing (Stage 6) and fairness verdict (Stage 7) are pure rule engines — no LLM.                       |
| **Schema-enforced LLM stages**  | Every LLM call uses `response_format=json_object` + Pydantic validation. Schema mismatch → retry, then surface.    |
| **Spare‑no‑cost LLM transport** | Bounded exponential backoff + jitter, Azure AD token re-acquisition on 401, hard wall-clock cap.                  |
| **Reproducible artefacts**      | Summaries are rebuilt from on-disk per-instance reports, not in-memory state. Resume after crash works correctly. |
| **Honest labels**               | `CLEAN` means "no signal on these axes", not "perfect task". Limits documented in [§ Limits](#limits-and-non-goals). |

---

## Architecture

```text
TaskRecord
  -> Stage 1   parse patch / test diffs
  -> Stage 1.5 code visitation + repo context (clone + AST)
  -> Stage 2   intent extraction (LLM, blind to gold patch)
  -> Stage 3   structural diff (astred-core, falls back to stdlib ast)
  -> Stage 4A  patch ↔ intent matching
  -> Stage 4B  test/assertion ↔ intent matching
  -> Stage 4C  cross-reference coupling analysis
  -> Stage 5   dual-taxonomy classification (Axis 1)
  -> Stage 6   report build + severity bucket          [deterministic]

ContaminationReport + TrajectoryRecord
  -> Stage 7   deterministic fusion verdict per (task, agent)
```

Module map:

| Path                                              | Responsibility                                                  |
| ------------------------------------------------- | --------------------------------------------------------------- |
| `bench_cleanser/pipeline.py`                      | Orchestrates Stages 1 – 6.                                      |
| `bench_cleanser/classification/dual_taxonomy.py`  | Axis-1 labels + severity logic.                                 |
| `bench_cleanser/trajectory/classifier.py`         | Axis-2 trajectory classification (heuristics + LLM).            |
| `bench_cleanser/fusion.py`                        | Stage-7 deterministic fairness rules.                           |
| `bench_cleanser/llm_client.py`                    | Azure OpenAI client with token caching + bounded retry policy.  |
| `bench_cleanser/repo_manager.py`                  | Idempotent clones, partial-clone recovery.                      |
| `bench_cleanser/cache.py`                         | Disk-backed response cache, deterministic keys.                 |
| `bench_cleanser/prompts/`                         | Versioned prompts shipped with the wheel (no eval-time `.md` loading from CWD). |

---

## Taxonomy at a glance

### Axis 1 — task contamination labels (multi-label, except `CLEAN`)

| Label                   | Meaning                                                                                  |
| ----------------------- | ---------------------------------------------------------------------------------------- |
| `approach_lock`         | Tests pin a specific implementation approach. Equivalent to "narrow test cases".          |
| `over_test`             | Tests assert behaviour the spec never requested. Equivalent to "wide test cases".         |
| `over_patch`            | Gold patch changes more than the spec implied.                                            |
| `unclear_description`   | Spec underspecifies the change set.                                                       |
| `hidden_context`        | Information required to solve is reachable only by reading the gold patch / tests.        |
| `weak_coverage`         | Tests do not actually pin down the desired behaviour.                                     |
| `clean`                 | No contamination signal on these axes (exclusive).                                        |

Severity buckets are derived **deterministically** from the label set; see [`docs/TAXONOMY.md`](docs/TAXONOMY.md).

### Axis 2 — agent trajectory labels (per `(task, agent)`)

| Label                            | Meaning                                                                  |
| -------------------------------- | ------------------------------------------------------------------------ |
| `agent_passed_genuine`           | Agent reasoned to the fix without leaks.                                 |
| `agent_passed_leak`              | Final patch matches the gold patch closely; high similarity threshold.   |
| `agent_passed_package_leak`      | Agent installed the fix instead of writing it (e.g. `pip install …`).     |
| `agent_passed_test_aware`        | Agent read the `fail_to_pass` test names from the trajectory.            |
| `agent_passed_trained_hack`      | Agent shortcut suggests memorised solution.                              |
| `agent_failed_completed_intent`  | Agent ran the spec but failed on hidden gotchas.                         |
| `agent_failed_no_intent`         | Agent did not engage substantively with the task.                        |
| `agent_unknown`                  | Signals insufficient to decide.                                          |

### Stage 7 fusion — fairness verdict per `(task, agent)`

| Verdict                | Means                                                                                |
| ---------------------- | ------------------------------------------------------------------------------------ |
| `fair_pass`            | Clean task + genuine pass. Counts as capability.                                     |
| `agent_cheated`        | Clean task but agent shortcut/leak. Should be excluded from capability counts.       |
| `contaminated_pass`    | Severe task contamination — pass not interpretable.                                  |
| `ambiguous_pass`       | Lesser contamination + ambiguous trajectory.                                         |
| `unfair_failure`       | Failure was caused by task defects, not agent capability.                            |
| `fair_failure`         | Clean task, genuine failure.                                                         |
| `agent_disengaged`     | Agent did not attempt the task.                                                      |
| `inconclusive`         | Insufficient evidence to decide.                                                     |

Each verdict comes with a textual `reasoning`, supporting `evidence` strings, and an `invalidates_measurement` flag. See [`docs/FUSION.md`](docs/FUSION.md) for the full rule matrix.

---

## Install

```bash
git clone https://github.com/v-liaozhu/bench-cleanser.git
cd bench-cleanser
pip install -e ".[dev]"
```

Optional extras:

- `.[trajectory]` — pulls `docent-python` for trajectory ingestion from Docent.
- `.[structural]` — pulls `astred-core` for the .NET-backed multilingual AST diff. The pipeline falls back transparently to stdlib `ast` if this is not installed.

Authentication: the LLM client uses CloudGPT (Azure OpenAI) via Azure AD. Make sure `az login` is available in your environment; tokens are acquired and cached automatically.

---

## Quickstart

```bash
# 1. Contamination analysis on SWE-bench Pro (first 50 tasks, smoke run)
bench-cleanser --dataset pro --max-tasks 50 --output out/pro

# 2. Trajectory + fusion analysis layered onto those reports
bench-cleanser-trajectory \
  --reports-dir out/pro/reports \
  --trajectory-source <jsonl|huggingface-dataset|docent-uuid> \
  --output out/pro/trajectory_analysis.md

# 3. Forensic deep dive on a severity bucket
bench-cleanser-deep-dive \
  --reports-dir out/pro/reports \
  --severity SEVERE \
  --output out/pro/deep_dive_severe.md
```

A full run on the public Verified split (500 tasks) takes ~1–2 h with `reasoning_effort=high` and concurrency 10, and resumes cleanly after interruption.

---

## CLI reference

Public entry points (declared in `pyproject.toml`):

| Command                        | Purpose                                                  |
| ------------------------------ | -------------------------------------------------------- |
| `bench-cleanser`               | Contamination pipeline (Stages 1 – 6).                   |
| `bench-cleanser-trajectory`    | Trajectory analysis + Stage-7 fusion.                    |
| `bench-cleanser-deep-dive`     | Per-instance forensic Markdown.                          |

`run_pipeline.py`, `run_trajectory_analysis.py`, `run_deep_dive.py` remain as **compatibility shims** for legacy invocation; the console-script names above are canonical.

Each command supports `--help` for the full flag list.

---

## Outputs

A run against `--output out/<name>` produces:

```text
out/<name>/
├── reports/                    # one JSON per instance — source of truth
│   └── <instance_id>.json
├── summary.csv                 # one row per instance, severity + label flags
├── summary_stats.json          # aggregate counts (rebuilt from reports/)
├── trajectory_analysis.md      # human-readable trajectory + fusion summary
└── trajectory_analysis.json    # machine-readable analyses + fusion records
```

Per-instance JSON includes:

- `severity`, `task_labels` (Axis 1).
- `intent`, `patch_analysis`, `test_analysis`, `description_clarity`.
- `recommendations` (action items, not labels).

Re-running with the same `--output` is idempotent: existing per-instance reports are reused, and `summary.csv` / `summary_stats.json` are rebuilt from disk.

---

## Configuration

Defaults live in `bench_cleanser.models.PipelineConfig`. The most relevant knobs:

| Field                       | Default                              | Purpose                                                |
| --------------------------- | ------------------------------------ | ------------------------------------------------------ |
| `llm_model`                 | `gpt-5.4-20260305`                   | Chat-completions model.                                |
| `llm_reasoning_effort`      | `"high"`                             | Reasoning effort for LLM stages.                       |
| `llm_max_tokens`            | `65536`                              | Max completion tokens.                                 |
| `max_concurrent_requests`   | `10`                                 | LLM-call concurrency.                                  |
| `retry_attempts`            | `7`                                  | Per-call attempt budget.                               |
| `retry_delay_seconds`       | `5.0`                                | Base backoff; exponential with jitter, capped at 60 s. |
| `cache_dir`                 | `.cache/llm_responses`               | Disk-backed response cache.                            |
| `repo_cache_dir`            | `.cache/repos`                       | Persistent clone cache.                                |

The LLM transport applies a hard 600 s wall-clock budget across retries per call, so a hung upstream cannot block the pipeline indefinitely.

---

## Quality controls

Run locally before pushing:

```bash
ruff check bench_cleanser tests
mypy bench_cleanser
pytest tests/ -q
```

CI (`.github/workflows/ci.yml`) runs the same three gates on Python 3.11 **and** 3.12. The current suite is **96 tests** covering:

- Patch parsing & similarity scoring.
- Dual-taxonomy heuristics (every Axis-1 label has a dedicated branch test).
- Fusion engine (every Stage-7 rule has a positive and a negative case).
- LLM client JSON extraction and cache-key determinism.
- Trajectory classifier (heuristic + LLM happy path + fallback).

Tests do not touch the network: any LLM interaction uses an in-test `FakeLLM`.

---

## Repository layout

```text
bench_cleanser/
├── analysis/                   # structural diff, cross-reference coupling
├── classification/             # dual taxonomy + severity rules
├── trajectory/                 # Axis-2 classification + Docent ingestion
├── prompts/                    # versioned LLM prompts (shipped in the wheel)
├── _internal/                  # vendored CloudGPT helpers (untouched in CI)
├── pipeline.py                 # Stage 1-6 orchestrator
├── fusion.py                   # Stage 7 deterministic fairness rules
├── llm_client.py               # Async Azure OpenAI transport
├── repo_manager.py             # Idempotent clone management
├── cache.py                    # Disk-backed response cache
├── schemas.py                  # Pydantic response models
└── models.py                   # Domain entities (TaskRecord, ContaminationReport, …)

docs/
├── TAXONOMY.md                 # Axis-1 + Axis-2 labels, evidence rules, severity mapping
├── FUSION.md                   # Stage-7 verdict matrix
└── CONTRIBUTING.md             # Dev workflow, extension checklists

tests/                          # 96 unit + integration tests, fully offline
.github/workflows/ci.yml        # ruff + mypy + pytest on 3.11 & 3.12
```

Generated outputs (`output/`, `audits/`, `case_studies/`, `.cache/`) are **not** source of truth and are excluded from lint, tests, and version control.

---

## Limits and non-goals

`bench-cleanser` is deliberately scoped. Things it does **not** do:

- It does not validate harness execution semantics (sandbox isolation, flakiness, runtime budgets).
- It does not detect every form of reward hacking — only those that surface in patch/test/trajectory signals.
- `CLEAN` is the absence of contamination signal on the axes we measure, **not** a guarantee that the task is perfect.
- Axis-1 labels are produced with LLM assistance and inherit its judgement noise; deterministic stages (severity, fusion) only amplify what the upstream labels say.

If you need a guarantee that a benchmark item is sound, you still need human review. `bench-cleanser` makes that review tractable — it does not replace it.

---

## Documentation index

- [`docs/TAXONOMY.md`](docs/TAXONOMY.md) — Axis-1 / Axis-2 labels, evidence rules, severity mapping.
- [`docs/FUSION.md`](docs/FUSION.md) — Stage-7 rule matrix and verdict definitions.
- [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) — Dev workflow, extension checklists, code style.
- [`CHANGELOG.md`](CHANGELOG.md) — Release history.

---

## License

MIT. See [LICENSE](LICENSE).
