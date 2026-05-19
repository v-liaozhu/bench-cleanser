# Contributing

Bug fixes, prompt improvements, tests, and clarifying docs are all
welcome. Adding new labels or fusion verdicts is in scope but requires
extra care — see the checklists below.

## Dev install

```bash
git clone https://github.com/v-liaozhu/bench-cleanser.git
cd bench-cleanser
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev,trajectory]"
```

Optional extras:

- `[structural]` adds `astred-core` for the astred AST backend over the
  stdlib `ast` fallback.
- `[trajectory]` adds `docent-python` for Docent-backed trajectory
  loaders.

## Tests

```bash
pytest tests/ -q
pytest tests/ --cov=bench_cleanser --cov-report=term-missing
ruff check bench_cleanser tests
mypy bench_cleanser
```

The full suite is offline by design — no network, no Azure, no git
clones. LLM-backed code paths use an in-test fake client in
`tests/test_trajectory_classifier.py`; fusion has no LLM call.

## Editing prompts

Prompts live as markdown under [`bench_cleanser/prompts/`](../bench_cleanser/prompts/):

```
intent_extraction.md       Stage 2
patch_classifier.md        Stage 4A
test_classifier.md         Stage 4B
task_classifier.md         Stage 5
trajectory_analysis.md     Stage 7 per-agent
```

Workflow:

1. Edit the markdown. No Python changes required.
2. Run the regressions: `pytest tests/test_trajectory_classifier.py
   tests/test_fusion_rule4.py tests/test_dual_taxonomy_heuristics.py -q`.
3. If the edit changes the JSON contract (new field, renamed field,
   tightened enum), update the matching Pydantic model in
   [`bench_cleanser/schemas.py`](../bench_cleanser/schemas.py) so
   structured output still validates.

Modules load prompts at import time via
`bench_cleanser.prompts.load("name")`. A misspelled name raises
`FileNotFoundError` at import.

## Adding a new label

1. **Define the evidence rule first.** What artefact must the
   classifier have on hand to assign this label? Fuzzy intuition is
   not enough.
2. **Update severity** in `bench_cleanser.classification.dual_taxonomy.compute_task_severity`
   with a set-membership rule — never a count or threshold.
3. **Add a Pydantic `Literal`** to `schemas.py::TaskLabelItem` (Axis 1)
   or `TrajectoryClassificationResponse` (Axis 2). Structured output
   guarantees the LLM cannot emit a label outside the enum.
4. **Add a regression test** with at least one synthetic input that
   should trigger and one that should not.
5. **Document** in `docs/TAXONOMY.md` with definition, evidence
   requirement, and severity contribution.

## Adding a new fusion verdict

Verdicts appear in shipped report JSON; consumers depend on the value
set. Before adding one:

1. State the rule as `(axis-1 condition) ∧ (axis-2 condition) ∧
   (optional resolved predicate)`.
2. Show which of the existing 8 verdicts would fire on the same input
   and argue why the new verdict is preferable.
3. Decide `invalidates_measurement` — `True` only when the pass/fail
   outcome cannot be trusted for *any* reason.
4. Add a parametrised matrix test (model on `tests/test_fusion_rule4.py`).
5. Document in `docs/FUSION.md` with at least one worked example.

## Conventions

- **Cite, don't characterise.** Every label and verdict carries the
  concrete evidence that produced it. "Tests look too narrow" is not
  evidence; `OFF_TOPIC assertion at tests/test_foo.py::test_a:
  "asserts isinstance(result, MyClass)"` is.
- **Deterministic where possible.** Severity and fusion have zero LLM
  calls. LLM is used to assign labels; everything downstream of labels
  is pure logic.
- **Real imports.** No `sys.path` hacks. `_internal` is the package
  for backend-private modules; everything else lives at its
  semantically correct path.
- **Comments explain WHY.** WHAT is already in the code. Comments
  capture non-obvious constraints, invariants, or rationale.

## Reporting bugs

[github.com/v-liaozhu/bench-cleanser/issues](https://github.com/v-liaozhu/bench-cleanser/issues).
Minimum repro:

- Python version and OS.
- `tree -L 2 output/` or `ls -R` of the output directory.
- First 100 lines of `output/.../<instance_id>.json` for classification
  bugs, or the trajectory file for Stage 7 bugs.
- The exact CLI invocation.

For prompt issues, name the markdown file in `bench_cleanser/prompts/`
and quote the report fields that look wrong.
