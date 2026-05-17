# Contributing

`bench-cleanser` welcomes contributions that **make the contamination
signal more trustworthy** — bug fixes, prompt improvements, additional
tests, and clarifying docs are all in scope. Adding new heuristics or
labels is in scope but requires more care; see [Adding a new label](#adding-a-new-label).

---

## Dev install

```bash
git clone https://github.com/v-liaozhu/bench-cleanser.git
cd bench-cleanser
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev,trajectory]"
```

The `[dev]` extra pins `pytest`, `pytest-asyncio`, `pytest-cov`, `ruff`,
and `mypy`. `[trajectory]` adds `docent-python` for Docent-backed
trajectory loaders. `[structural]` adds `astred-core` if you want the
astred AST backend over the stdlib `ast` fallback (it's an optional
quality upgrade; the stdlib path is always available).

---

## Running tests

```bash
pytest tests/ -q
```

The full suite is offline by design — no network, no Azure, no git
clones. LLM-backed code paths are exercised against a fake client
defined inline in `tests/test_trajectory_classifier.py` and against the
fusion rule engine, which has no LLM call.

Coverage spot-check:

```bash
pytest tests/ --cov=bench_cleanser --cov-report=term-missing
```

Lint:

```bash
ruff check bench_cleanser tests
```

Lint config lives in `pyproject.toml` under `[tool.ruff]`. The defaults
are intentionally light — formatting is not enforced, only correctness
rules and import sorting.

---

## Editing prompts

LLM prompts live as plain markdown under
[`bench_cleanser/prompts/`](../bench_cleanser/prompts/):

```
bench_cleanser/prompts/
├── intent_extraction.md       # Stage 2
├── patch_classifier.md        # Stage 4A
├── test_classifier.md         # Stage 4B
├── task_classifier.md         # Stage 5
└── trajectory_analysis.md     # Stage 7 (per-agent)
```

To edit a prompt:

1. Edit the markdown file directly. No Python changes required.
2. Re-run the regression tests: `pytest tests/test_trajectory_classifier.py
   tests/test_fusion_rule4.py tests/test_dual_taxonomy_heuristics.py -q`.
3. If your edit changes the JSON contract returned by the LLM (new
   field, renamed field, tightened enum), also update the matching
   Pydantic model in [`bench_cleanser/schemas.py`](../bench_cleanser/schemas.py)
   so structured output still validates.

Every module loads its prompt at import time via
`bench_cleanser.prompts.load("name")` — there is no separate "register"
step. A misspelled name raises `FileNotFoundError` at import.

---

## Adding a new label

Adding a contamination label or trajectory label is a contract change.
Before opening a PR:

1. **Write the evidence rule first.** What artefact must the classifier
   have on hand to assign this label? If it's a fuzzy intuition rather
   than a concrete artefact, the label is not ready.
2. **Update the severity bucket** in
   `bench_cleanser.classification.scorer.compute_task_severity` *with a
   set-membership rule, never a count or threshold*. Severity stays
   pure set membership over the label set.
3. **Add a Pydantic Literal** to `schemas.py::TaskLabelItem` (Axis 1)
   or `TrajectoryClassificationResponse` (Axis 2). The structured-
   output guarantee means the LLM cannot emit a label that isn't in
   the enum.
4. **Add a regression test** with at least one synthetic
   `TestAnalysis` / `PatchAnalysis` that should and one that should
   not trigger the label.
5. **Document in `docs/TAXONOMY.md`**: definition, evidence
   requirement, severity contribution.

---

## Adding a new fusion verdict

Fusion verdicts (Stage 7) are even more committal — they appear in
shipped report JSON and consumers depend on the value set. Before
adding a verdict:

1. Show the **rule** in cause-and-effect terms (axis-1 condition AND
   axis-2 condition AND optional `resolved` predicate).
2. Show the existing 8 verdicts that would otherwise have fired on
   the same input and argue why the new verdict is preferable.
3. Decide `invalidates_measurement` — `True` only when the row's
   pass/fail outcome cannot be trusted for *any* reason.
4. Add a parametrised matrix test to `tests/test_fusion_rule4.py` (or
   a new file modelled on it).
5. Document in `docs/FUSION.md` with at least one worked example.

---

## Conventions

* **Citations, not adjectives.** Every label/verdict must carry the
  concrete evidence that produced it. "Tests look too narrow" is not
  evidence; `OFF_TOPIC assertion at tests/test_foo.py::test_a, "asserts
  isinstance(result, MyClass) which is implementation-locked"` is.
* **Deterministic where possible.** Severity, fusion, and severity
  rules have zero LLM calls. The LLM is used to assign labels;
  everything downstream of labels is pure logic.
* **No silent fallbacks in classification.** A LLM failure during
  trajectory classification falls back to the heuristic classifier,
  but the resulting `TrajectoryAnalysis` records that it came from
  the fallback in its `evidence_strength` and `evidence` fields.
* **Imports are real.** No `sys.path` hacks. `_internal` is the
  package for backend-private modules; everything else lives at its
  semantically correct path.

---

## Reporting bugs

Open an issue at
[github.com/v-liaozhu/bench-cleanser/issues](https://github.com/v-liaozhu/bench-cleanser/issues).
The minimum reproduction we need is:

* The Python version and OS.
* The output directory layout (a `tree -L 2 output/` or `ls -R` is fine).
* The first 100 lines of `output/.../<instance_id>.json` if the bug is
  in classification, or the trajectory file if it's in Stage 7.
* The exact CLI command you ran.

For prompt-engineering reports — "the LLM should have said X" — paste
the system prompt name (filename in `bench_cleanser/prompts/`) and the
specific report fields that look wrong. Prompts are versioned in git,
so referring to them by filename is enough.
