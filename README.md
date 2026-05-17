# bench-cleanser

> *Automated contamination, fairness, and trajectory-leakage analysis for SWE-bench
> family benchmarks (Verified, Pro, Live).*

`bench-cleanser` is a multi-stage pipeline that takes raw SWE-bench task records
(problem statement + gold patch + test patch) and an optional set of agent
trajectories, and produces:

1. A **task-level contamination report** (Axis 1) — what is wrong with the
   benchmark item itself.
2. A **per-agent trajectory verdict** (Axis 2) — how each agent arrived at its
   answer and whether the trajectory shows a leakage signal.
3. A **fused fairness verdict** per `(task, agent)` pair (Stage 7) — combining
   the two axes into a single deterministic call: did this measurement actually
   measure capability?

Every classification is grounded in artefacts from the task: parsed diff hunks,
AST-level structural diff, call-graph edges between F2P tests and changed
blocks, per-assertion `ON_TOPIC` / `OFF_TOPIC` verdicts, and per-trajectory
patch-similarity / pip-install / test-reference signals. There are no
free-floating numerical thresholds in the contamination logic — labels are
assigned on evidence, severity is computed from label-set membership.

---

## Table of contents

- [Trust model](#trust-model)
- [Pipeline overview](#pipeline-overview)
- [Stages 1 – 7 in detail](#stages-1--7-in-detail)
- [Taxonomies](#taxonomies)
- [Repository layout](#repository-layout)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [How to interpret a report](#how-to-interpret-a-report)
- [Testing](#testing)
- [Design principles](#design-principles)
- [Provenance — what changed and why](#provenance--what-changed-and-why)

Reference docs ship under [`docs/`](docs/):

- [`docs/TAXONOMY.md`](docs/TAXONOMY.md) — every label, evidence rule, severity contribution, OpenAI-audit mapping.
- [`docs/FUSION.md`](docs/FUSION.md) — the eight Stage-7 verdicts with worked examples.
- [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) — dev install, prompt-editing workflow, label-addition checklist.

---

## Trust model

`bench-cleanser` is intended to support **SOTA SWE training and
evaluation work** — that means every output has to survive scrutiny
from someone reading the OpenAI [*Why we no longer evaluate
SWE-bench Verified*](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/)
critique. The guarantees in v1.0.0 are deliberately narrow:

**What this tool guarantees**

- Every assigned label carries explicit evidence (hunk indices,
  assertion indices, problem-text quotes) persisted in the report
  JSON. A label without evidence cannot be emitted — the classifier
  rejects it.
- Severity is **pure set membership** over the Axis-1 label set. No
  floating-point thresholds, no per-label weights, no counts. The
  severity bucket is reproducible from the persisted report alone.
- Stage 7 fusion is **deterministic** — no LLM call. The fairness
  verdict for `(task, agent)` is a pure rule engine over the two
  axis labels and the trajectory's `resolved` outcome.
- LLM-backed classification (intent, patch, test, task, trajectory)
  goes through Pydantic structured output (`response_format: json_object`
  + schema in system prompt). No ad-hoc JSON parsing; no silent field
  omissions.
- Prompts ship as plain markdown in [`bench_cleanser/prompts/`](bench_cleanser/prompts/)
  and are diffable in PRs.

**What this tool does NOT guarantee**

- That its LLM-assigned labels match a human labeller. The taxonomy
  is precise but the LLM remains the limiting factor; see
  [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) for how to refine a
  prompt without changing the deterministic logic underneath.
- That a `CLEAN` verdict means the row is suitable for every
  conceivable use — only that the bench-cleanser axes turned up
  nothing. Reward hacking, agent-runtime flakes, and harness bugs
  are out of scope.
- That `invalidates_measurement` should be the only filter your
  evaluation applies. It marks the rows the tool *knows* cannot be
  used; consumers may legitimately apply stricter filters (e.g.,
  drop `AMBIGUOUS_PASS` as well when computing a headline score).

**When to override a verdict**

The tool will rarely produce a wrong verdict given correct labels,
because Stage 7 is deterministic. Overrides should target the
**labels**, not the verdict — re-classify the task with corrected
evidence, then let fusion re-derive the verdict. A `task_labels.json`
override file is on the roadmap for v1.1; today the workflow is to
edit the report JSON and re-run `bench-cleanser-trajectory` against
the corrected report directory.

---

## Pipeline overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│   raw TaskRecord                                                        │
│   (problem_statement, patch, test_patch, F2P, P2P, repo, base_commit)   │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
   ┌─────────── Stage 1  PARSE ─────────────────────────────────────┐
   │  parsing/patch_parser.py   parsing/test_parser.py              │
   │  → ParsedTask (PatchHunk[], TestHunk[], F2P-to-hunk mapping)   │
   └───────────────────────────────────────────────────────────────┘
                             │
                             ▼
   ┌──────── Stage 1.5  CODE VISITATION ────────────────────────────┐
   │  repo_manager.py           code_visitor.py                     │
   │  static_analysis.py                                            │
   │  → CodeContext per F2P test (full source, fixtures, imports,   │
   │    tested functions, call targets, assertions, pre-patch src)  │
   └───────────────────────────────────────────────────────────────┘
                             │
                             ▼
   ┌──────── Stage 2  INTENT EXTRACTION (LLM, blind to patch) ──────┐
   │  analysis/scope_analyzer.py                                    │
   │  → IntentStatement (acceptance_criteria, behavioral_contract,  │
   │    out_of_scope, ambiguity_score, mentioned entities)          │
   └───────────────────────────────────────────────────────────────┘
                             │
                             ▼
   ┌──────── Stage 3  STRUCTURAL DIFF (astred_core / ast) ──────────┐
   │  analysis/structural_diff.py                                   │
   │  → StructuralDiff (changed_blocks[], test_blocks[],            │
   │    call_edges between tests and changed source)                │
   └───────────────────────────────────────────────────────────────┘
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
   ┌── Stage 4A  PATCH MATCH ─┐   ┌── Stage 4B  TEST MATCH ───────────┐
   │  analysis/patch_analyzer │   │  analysis/test_analyzer.py        │
   │  → PatchAnalysis         │   │  → TestAnalysis                   │
   │    (REQUIRED / ANCILLARY │   │    (ALIGNED / TANGENTIAL /         │
   │     / UNRELATED hunks)   │   │     UNRELATED tests, per-          │
   │                          │   │     assertion ON_TOPIC /           │
   │                          │   │     OFF_TOPIC, modification        │
   │                          │   │     alignment)                     │
   └──────────────────────────┘   └────────────────────────────────────┘
                ▼                         ▼
   ┌── Stage 4C  CROSS-REFERENCE ───────────────────────────────────┐
   │  analysis/cross_ref.py                                         │
   │  → CrossReferenceResult (overpatch–overtest couplings)          │
   └───────────────────────────────────────────────────────────────┘
                             │
                             ▼
   ┌──────── Stage 5  DUAL TAXONOMY CLASSIFIER (LLM) ───────────────┐
   │  classification/dual_taxonomy.py                               │
   │  → list[TaskLabelAssignment] (multi-label)                     │
   └───────────────────────────────────────────────────────────────┘
                             │
                             ▼
   ┌──────── Stage 6  REPORT BUILD + SEVERITY ──────────────────────┐
   │  classification/scorer.py                                      │
   │  classification/dual_taxonomy.py::compute_task_severity        │
   │  → ContaminationReport (CLEAN / MINOR / MODERATE / SEVERE +    │
   │    label set + recommendations) — written to output/reports/   │
   └───────────────────────────────────────────────────────────────┘

   ─── trajectory side (independent) ───────────────────────────────
   ┌──────── trajectory loading + classification ───────────────────┐
   │  trajectory/loader.py           trajectory/analyzer.py         │
   │  trajectory/classifier.py                                      │
   │  → TrajectoryAnalysis per (instance, agent), with a single     │
   │    AgentTrajectoryLabel from Axis 2                            │
   └───────────────────────────────────────────────────────────────┘
                             │
                             ▼
   ┌──────── Stage 7  TASK-TRAJECTORY FUSION (rule-based) ──────────┐
   │  fusion.py                                                     │
   │  → TaskTrajectoryFusion per (task, agent)                      │
   │    8 verdicts, invalidates_measurement flag                    │
   └───────────────────────────────────────────────────────────────┘
```

Stages 1–6 run inside `bench_cleanser/pipeline.py::process_single_task`.
Trajectory loading + Stage 7 run inside
`bench_cleanser/trajectory/analyzer.py::run_trajectory_analysis`, invoked from
`run_trajectory_analysis.py`.

---

## Stages 1 – 7 in detail

### Stage 1 – PARSE

Pure-Python diff parsing. No LLM, no IO.

* `parsing/patch_parser.py` – splits the gold patch into `PatchHunk` objects
  carrying `file_path`, hunk index, added/removed line lists, and the original
  diff text.
* `parsing/test_parser.py` – does the same for the test patch and emits
  `TestHunk` objects with `modification_type ∈ {NEW, MODIFIED, UNKNOWN}`.
* `match_f2p_tests_to_hunks` maps each F2P test ID to the test hunk that
  defines or modifies it. F2P entries with no matching hunk (gold-state tests
  that the patch does not touch) are tracked separately.

Output: a `ParsedTask` containing the original `TaskRecord`, both hunk lists,
and the F2P mapping. This is the only intermediate that Stage 1.5 mutates.

### Stage 1.5 – CODE VISITATION

Optional but recommended. Driven by `repo_manager.py` and `code_visitor.py`.

* `RepoManager` does shallow `git clone`s into `.cache/repos/<repo>@<commit>/`,
  with a per-clone timeout and disk reuse.
* For every F2P test hunk, `enrich_with_code_context`:
  * recovers the full pre-patch test source via `get_full_test_source`
  * derives the post-patch test source by replaying added/removed lines
  * extracts file imports and pytest fixtures
  * builds a Python-level import map (`resolve_imports`) and identifies the
    functions actually exercised by the test (`identify_tested_functions`)
  * records concrete call targets and the AST-level assertions
* The result is stored on the test hunk as `CodeContext`, including the
  resolved `repo_path` so downstream stages can re-read source if needed.

Failure mode: if the clone times out or the file cannot be located, Stage 1.5
logs a warning, attaches no `CodeContext`, and the pipeline continues — Stage 4B
will then operate on test source only.

### Stage 2 – INTENT EXTRACTION (LLM; blind to gold patch)

`analysis/scope_analyzer.py::extract_intent`.

The LLM sees: the problem statement, hints, requirements (Pro), interface spec
(Pro), and optionally pre-patch source for files the problem mentions. It does
**not** see the gold patch or test patch. Output schema (`schemas.py`):

```python
class IntentStatement:
    core_requirement: str
    behavioral_contract: str        # BEFORE → AFTER
    acceptance_criteria: list[str]  # explicit testable behaviours
    out_of_scope: str
    ambiguity_score: float          # qualitative, advisory only
    decomposition: ProblemDecomposition  # bug / suggested_fix / mentioned entities
```

`ambiguity_score` is **never** used as a gate by the contamination logic. It is
written into the report for human review and shown to the Stage 5 classifier
LLM as advisory context only. The classifier is explicitly instructed to make
its `UNCLEAR_DESCRIPTION` decision from the problem text directly, not from the
score.

### Stage 3 – STRUCTURAL DIFF

`analysis/structural_diff.py::compute_structural_diff`.

Uses `astred_core` (.NET-backed multilingual AST) when available — installed
into the venv by `pip install astred_core` plus a working .NET runtime. The
correct call sequence is:

```python
pre_graph = CodeGraph()
for path in changed_files:
    ast_file = AstFile.load_file(path)        # parse one file
    CodeGraphEdits.build(pre_graph, ast_file) # accumulate edit_status flags
```

Output (`StructuralDiff`):
* `changed_blocks[]` – function/class blocks touched by the patch, with
  `pre_source`, `post_source`, and `edit_status`
* `test_blocks[]` – the F2P test functions
* `call_edges[]` – `(test_name, callee_function_name)` pairs from the test
  body to functions in the changed blocks
* `astred_available: bool`

If astred fails or is missing, the module falls back to Python's stdlib `ast`
to extract function/class blocks and a coarse call list. The LLM still gets
*something*, just less precise.

### Stage 4A – PATCH INTENT MATCH

`analysis/patch_analyzer.py::analyze_patch`.

Single batched LLM call. Input is the full `IntentStatement`, the hunk diff
text for every hunk, and per-hunk structural context (the pre-patch source of
the function the hunk modifies, when astred provided it). Output is one of
three verdicts per hunk plus a per-hunk reasoning string:

* `REQUIRED` — directly implements an acceptance criterion
* `ANCILLARY` — supports a REQUIRED change (imports, exports, type hints,
  docstrings, whitespace) but is not itself demanded by the problem
* `UNRELATED` — behavioural change beyond the problem scope

`PatchAnalysis` aggregates counts plus per-hunk verdicts.

### Stage 4B – TEST INTENT MATCH

`analysis/test_analyzer.py::analyze_tests`.

Single batched LLM call across all F2P tests. Per test:

* full post-patch source and (when modified) pre-patch source
* fixtures, imports, tested-function source, call-target list with
  `IN GOLD PATCH` markers
* matched `Structural context` block listing call edges into changed blocks

Output per test: a test-level verdict (`ALIGNED` / `TANGENTIAL` / `UNRELATED`),
a per-assertion verdict (`ON_TOPIC` / `OFF_TOPIC`), and an
`is_modification_aligned` flag for modified tests. `TestAnalysis` aggregates
totals plus per-test verdicts.

The Stage 4B system prompt explicitly frames per-assertion verdicts as the
primary signal and instructs the model to be liberal with `OFF_TOPIC` —
because every `OFF_TOPIC` assertion drives Stage 5's `OVER_TEST` decision.

### Stage 4C – CROSS-REFERENCE

`analysis/cross_ref.py::analyze_cross_references`.

Pure heuristic, no LLM. Looks at every F2P test and, when it has structural
or call-target information, checks whether it reaches functions inside hunks
that Stage 4A labelled `UNRELATED`. Each match becomes an
`OverpatchOvertestLink` with the linked hunk indices and files. The result is
a `CrossReferenceResult`; `has_coupling` flips Stage 5 toward `APPROACH_LOCK`.

### Stage 5 – DUAL TAXONOMY CLASSIFIER

`classification/dual_taxonomy.py::classify_task_labels`.

Two phases:

1. **Heuristic pre-classification** (`_heuristic_labels`) – fast binary
   signals from prior stages: any `OFF_TOPIC` assertion or `UNRELATED` test or
   misaligned modified test → candidate `OVER_TEST`; any `UNRELATED` hunk →
   candidate `OVER_PATCH`; certain compound patterns (Task/Patch Mismatch,
   compilation barrier on `.go` / `.ts` / `.tsx` / `.rs`, pre-staged tests)
   → candidate `APPROACH_LOCK`; self-referential phrasing → candidate
   `HIDDEN_CONTEXT`. There is **no float gate** anywhere in this function.
   `UNCLEAR_DESCRIPTION` is *only* assigned by the LLM in phase 2.
2. **LLM classifier** (`TASK_CLASSIFIER_SYSTEM_PROMPT`) – receives the full
   intent, the per-hunk and per-test/per-assertion verdicts, the
   cross-reference result, the heuristic candidates, and the full problem
   text. Reassigns / refines / rejects candidates and emits the final label
   set with evidence and reasoning.

The classifier prompt explicitly encodes the PR-authorship insight from the
107-case audit: tests and gold patch are co-authored, so `OVER_TEST` without
`OVER_PATCH` is rare and demands a re-examination of `ANCILLARY` hunks.

### Stage 6 – REPORT BUILD + SEVERITY

`classification/scorer.py::build_report` + `compute_task_severity`.

Severity is **pure set membership** — no arithmetic, no weights:

| Severity | Rule |
| --- | --- |
| `SEVERE` | `APPROACH_LOCK ∈ labels` **OR** `OVER_TEST ∈ labels` |
| `MODERATE` | `OVER_PATCH ∈ labels` **AND** (`HIDDEN_CONTEXT ∈ labels` **OR** `UNCLEAR_DESCRIPTION ∈ labels`) |
| `MINOR` | any single one of `{OVER_PATCH, UNCLEAR_DESCRIPTION, HIDDEN_CONTEXT, WEAK_COVERAGE}` |
| `CLEAN` | empty (or only `clean`) label set |

Recommendations are gated by the final label set — only labels that survived
the classifier produce a recommendation line. The full `ContaminationReport`
is serialised to `output_dir/reports/<instance_id>.json` and rolled up into
`summary.csv` and `summary_stats.json`.

### Stage 7 – TASK-TRAJECTORY FUSION

`fusion.py::fuse`.

Deterministic rule engine, no LLM. For each `(task, agent)` pair, combines
the task's `Severity` + `task_labels` (Axis 1) with the agent's
`AgentTrajectoryLabel` (Axis 2) and emits a single `FusionVerdict`:

| Verdict | Trigger | Invalidates measurement |
| --- | --- | --- |
| `FAIR_PASS` | genuine pass on a clean / minor task | no |
| `AGENT_CHEATED` | pass with leakage trajectory (`agent_passed_leak`, `_test_aware`, `_package_leak`, `_trained_hack`) | **yes** |
| `CONTAMINATED_PASS` | genuine pass on a SEVERE / MODERATE task | **yes** |
| `AMBIGUOUS_PASS` | pass with `agent_unknown` on a clean task | no |
| `UNFAIR_FAILURE` | failed-with-completed-intent on a task carrying `APPROACH_LOCK` or `OVER_TEST` | **yes** |
| `FAIR_FAILURE` | failed-with-completed-intent on a clean / minor task | no |
| `AGENT_DISENGAGED` | failed without producing an intent | no |
| `INCONCLUSIVE` | none of the above | no |

The `invalidates_measurement` flag is the headline number — it tells you what
fraction of the benchmark's reported numbers are not actually measurements of
capability. `run_trajectory_analysis.py` writes a per-pair table into the
output JSON and a fusion summary section into the markdown report.

---

## Taxonomies

### Axis 1 — Task contamination (multi-label, binary)

| Label | Display | OpenAI Verified equivalent | Meaning |
| --- | --- | --- | --- |
| `approach_lock` | Approach Lock | "Narrow test cases" | F2P tests reject valid alternative solutions |
| `over_test` | Over Test | "Wide test cases" | F2P tests assert on behaviour not described in the problem |
| `over_patch` | Over Patch | — | Gold patch carries behavioural changes beyond the problem |
| `unclear_description` | Unclear Description | — | Problem statement is ambiguous or misleading |
| `hidden_context` | Hidden Context | — | Essential spec lives in hints, not the problem |
| `weak_coverage` | Weak Coverage | — | F2P tests / patch under-cover the stated criteria |
| `clean` | Clean | — | No contamination signals (mutually exclusive) |

### Axis 2 — Agent trajectory (single label per `(task, agent)`)

| Label | Meaning |
| --- | --- |
| `agent_passed_genuine` | Solved from problem statement; no leakage signals |
| `agent_passed_leak` | Final patch ≈ gold patch (high difflib similarity) |
| `agent_passed_test_aware` | Trajectory references F2P test names or expected values |
| `agent_passed_package_leak` | Solution comes via `pip install <package>` from PyPI |
| `agent_passed_trained_hack` | Patch matches a known training-set fingerprint |
| `agent_failed_completed_intent` | Implemented the described behaviour but failed F2P |
| `agent_failed_no_intent` | Never produced an intent matching the problem |
| `agent_unknown` | Insufficient signal — manual review |

---

## Repository layout

```
bench_cleanser/
├── pipeline.py             Stage orchestration + concurrency + IO
├── data_loader.py          HuggingFace loaders for Verified / Pro / Live
├── repo_manager.py         Shallow git clone + on-disk cache
├── code_visitor.py         Pre/post-patch test source recovery
├── static_analysis.py      Import resolver, assertion + call extraction
├── llm_client.py           AsyncAzureOpenAI w/ AAD token cache + retry
├── cache.py                Disk cache for LLM responses
├── schemas.py              JSON Schemas for structured-output enforcement
├── models.py               Dataclasses (TaskRecord, IntentStatement, …)
├── presentation.py         Rich-based summary panels + markdown writer
├── deep_dive.py            Per-instance forensic markdown generator
├── fusion.py               Stage 7 deterministic fusion engine
├── parsing/                Stage 1
│   ├── patch_parser.py
│   └── test_parser.py
├── analysis/               Stages 2 – 4C
│   ├── scope_analyzer.py
│   ├── structural_diff.py
│   ├── patch_analyzer.py
│   ├── test_analyzer.py
│   └── cross_ref.py
├── classification/         Stages 5 – 6
│   ├── dual_taxonomy.py
│   └── scorer.py
└── trajectory/             Trajectory side
    ├── loader.py
    ├── classifier.py
    ├── analyzer.py
    └── models.py

run_pipeline.py             Entry point: tasks → reports
run_trajectory_analysis.py  Entry point: reports + trajectories → fusion
run_deep_dive.py            Entry point: pick N reports, write forensic .md
run_slides.py               Findings deck generator
run_trajectory_analysis.py  (see above)
monitor_pipeline.py         Live progress tail
audit.py                    Manual audit-tracker helpers

tests/                      pytest suite (53 tests at HEAD)
config.yaml                 LLM endpoint, concurrency, cache dirs
```

---

## Quick start

### Install

```bash
# Development install with all extras (recommended)
git clone https://github.com/v-liaozhu/bench-cleanser.git
cd bench-cleanser
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,trajectory]"

# Minimal runtime install (subset; no dev tools)
pip install -r requirements.txt
```

After install three console scripts are on `$PATH`:

```bash
bench-cleanser --help              # contamination pipeline
bench-cleanser-trajectory --help   # Stage 7 fusion + leakage analysis
bench-cleanser-deep-dive --help    # per-instance forensic markdown
```

The legacy `python run_pipeline.py` / `run_trajectory_analysis.py` /
`run_deep_dive.py` entry points still work — they are thin shims over
the console scripts.

### Run

```powershell
# Azure CLI login (the LLM client uses az for AAD tokens)
az login

# 50 SWE-bench Pro tasks
bench-cleanser --dataset pro --max-tasks 50 --output output_pro_v21 --concurrency 5

# Stage 7 fusion against those reports
bench-cleanser-trajectory `
    --reports-dir output_pro_v21/reports `
    --trajectory-source <jsonl-file or HF dataset or Docent collection> `
    --output output_pro_v21/trajectory_fusion.md
```

For a single task:

```bash
bench-cleanser --instance-id ansible__ansible-a26c325b... --output output_one
```

For deep-dive forensic markdown on selected reports:

```bash
bench-cleanser-deep-dive --reports-dir output_pro_v21/reports --severity SEVERE
```

---

## Configuration

Everything that has a sensible default lives in `config.yaml`. Highlights:

```yaml
llm:
  base_url: "https://cloudgpt-openai.azure-api.net/"
  api_version: "2025-04-01-preview"
  model: "gpt-5.4-20260305"
  max_tokens: 65536
  reasoning_effort: "high"
  max_concurrent_requests: 10
  retry_attempts: 7              # bounded for API errors
  retry_delay_seconds: 5.0
pipeline:
  concurrency: 3                 # tasks in parallel
  cache_dir: ".cache/llm_responses"
  output_dir: "output_pro_v6"
code_visitation:
  enabled: true
  repo_cache_dir: ".cache/repos"
  clone_timeout_seconds: 120
  max_source_context_lines: 200
```

`llm_client.py` sets `timeout=None` on every chat call and retries
`APIConnectionError` indefinitely. `retry_attempts` only governs API errors
(rate limits, 5xx, validator errors).

Authentication is performed by `cloudgpt_aoai.get_openai_token_provider`,
which shells out to `az` once and reuses the token for ~50 minutes.

---

## Outputs

Each pipeline run writes:

```
output_dir/
├── reports/<instance_id>.json   # one ContaminationReport per task
├── summary.csv                  # severity + label flags per task
└── summary_stats.json           # aggregate counts
```

`ContaminationReport.to_dict()` keys: `instance_id`, `severity`, `intent`,
`patch_analysis` (per-hunk verdicts), `test_analysis` (per-test +
per-assertion), `description_clarity`, `task_labels`, `agent_labels`,
`recommendations`.

`run_trajectory_analysis.py` adds:

```
output_dir/
├── trajectory_fusion.md         # per-instance narratives + fusion table
└── trajectory_fusion.json       # analyses[], leakage_rates{}, fusion[]
```

---

## How to interpret a report

Every task produces a JSON file under `<output_dir>/reports/`. A
representative one ships under [`examples/sample_run/reports/`](examples/sample_run/);
walk through it field-by-field:

```jsonc
{
  "instance_id": "ansible__ansible-0ea40e09...",
  "severity": "MINOR",                    // bucket: CLEAN | MINOR | MODERATE | SEVERE
  "intent": { /* Stage 2 IntentStatement */ },
  "excess_patch": { /* Stage 4A summary */ },
  "excess_test":  { /* Stage 4B summary */ },
  "vague_spec":   { /* DescriptionClarity */ },
  "task_labels": [
    {
      "label": "weak_coverage",
      "evidence": [
        "Cross-reference Stage 4C: 0 call edges from F2P tests to changed blocks"
      ],
      "reasoning": "..."
    }
  ],
  "agent_labels": [ /* heuristic axis-2 candidates if no trajectory was joined */ ],
  "recommendations": [ /* short remediation hints for benchmark maintainers */ ]
}
```

Reading order, in priority:

1. **`severity`** → answers "should I drop this row?" at the coarsest
   level. `CLEAN` and `MINOR` are usually safe; `MODERATE` and `SEVERE`
   warrant either a fix or an exclusion.
2. **`task_labels[*].label`** → tells you *why* the severity is what
   it is. Each label maps to a section of
   [`docs/TAXONOMY.md`](docs/TAXONOMY.md).
3. **`task_labels[*].evidence`** → the concrete artefact that supports
   the label. If the evidence is wrong, the label is wrong; if the
   evidence is missing, treat the label as not-yet-trusted and
   re-classify with a manual override.
4. **`intent`** → if the labels look surprising, check whether intent
   extraction got the problem right. `intent.acceptance_criteria`
   is the spine — most contamination labels are downstream of that
   list.
5. **`recommendations`** → human-targeted hints, not machine-readable
   verdicts. Use them to write a fix or a justification, not to drive
   automated filtering.

To go further — combine with agent trajectories and get fairness
verdicts — run `bench-cleanser-trajectory` against the reports
directory. The resulting `trajectory_fusion.json` contains a `fusion[]`
array whose verdicts are explained in [`docs/FUSION.md`](docs/FUSION.md).

---

## Testing

```bash
pytest tests/ -q
```

86 tests at HEAD covering: patch parser, test parser, scorer, dual
taxonomy severity rules, OVER_TEST heuristic collapse regression,
fusion verdicts and the full Rule 4 (AMBIGUOUS_PASS / INCONCLUSIVE)
matrix, LLM client retry behaviour, and trajectory classifier
(similarity computation, pip-install detection, test-reference
detection, heuristic-only classification, cross-agent inference, and
the structured-output happy/error paths against a fake LLM client).

Lint:

```bash
ruff check bench_cleanser tests
```

CI runs both on Python 3.11 and 3.12 — see
[`.github/workflows/ci.yml`](.github/workflows/ci.yml).

---

## Design principles

1. **No float thresholds in contamination logic.** Severity is set membership;
   labels are evidence-driven binary assignments. The only floats in the
   pipeline are `ambiguity_score` and `gold_patch_similarity`, both of which
   are advisory context for the LLM, never gates.
2. **PR-authorship realism.** The classifier prompt and the severity rules
   both encode the audit insight that gold patch and F2P tests are co-authored.
   `OVER_TEST` without `OVER_PATCH` is rare and is treated as maximum-attention
   evidence rather than a softer bucket.
3. **Cite or shut up.** Every stage that emits a verdict also emits per-item
   evidence. Reports never contain a label without a reason.
4. **No silent truncation.** Prompts include the full problem statement,
   requirements, interface, hints, decomposition, intent, per-hunk reasoning,
   per-assertion list, and structural context. Truncation only happens when
   astred is genuinely unavailable.
5. **Connectivity-fault-tolerant.** The LLM client retries `APIConnectionError`
   forever and bounds retries on real API errors. Long-running batches survive
   transient network drops without losing work.
6. **Stage 7 is pure rules.** Fusion is deliberately deterministic. The LLM
   does not get a vote on whether a measurement is fair — that decision is
   reproducible from the two axis labels alone.

---

## Provenance — what changed and why

| Commit | What | Why |
| --- | --- | --- |
| `b2b3b1c` | v1.5.0 baseline | end-to-end pipeline + trajectory analysis |
| `33e185a` | v2.0 overhaul | binary severity, archaic class rename, astred API fix, infinite connectivity retry, no truncation |
| `493cfd6` | severity rule fix + over_test hardening + Stage 7 fusion | audit-driven severity rewrite (`OVER_TEST` → SEVERE), removal of the `description_clarity.score >= 0.4` heuristic gate, prompt enrichment for OVER_TEST detection, new `bench_cleanser/fusion.py` |

Latest run: `output_pro_v21/` (50 SWE-bench Pro tasks, 22 min wall clock,
zero `astred_core failed` warnings). Severity distribution under the
new rules:

| Severity | Count | % |
| --- | --- | --- |
| CLEAN | 12 | 24% |
| MINOR | 27 | 54% |
| MODERATE | 0 | 0% |
| SEVERE | 11 | 22% |

Label distribution: `weak_coverage`=28, `over_patch`=12, `over_test`=9,
`approach_lock`=4, `clean`=12.
