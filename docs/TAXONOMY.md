# Taxonomy

`bench-cleanser` classifies every benchmark item along two independent
axes. The axes serve different questions and must not be conflated:

| Axis | Question | Cardinality | Where it lives |
|---|---|---|---|
| **Axis 1 — Task contamination** | *Is the benchmark item itself fair?* | **Multi-label** (0 – 7 labels) | `bench_cleanser.classification.dual_taxonomy` |
| **Axis 2 — Agent trajectory** | *How did this agent arrive at its answer?* | **Single label** per `(task, agent)` | `bench_cleanser.trajectory.classifier` |

Both axes are then fused into a single fairness verdict — see
[FUSION.md](FUSION.md).

---

## Why two axes

A benchmark row has two things going on at once: the *measurement instrument*
(the task, gold patch, and F2P tests) and the *behaviour being measured* (a
specific agent on that task). Conflating them is the root cause of most
"contamination" debates.

* A task with `APPROACH_LOCK` is broken — any agent that solves it
  differently fails, regardless of whether it actually solves the bug.
  That's Axis 1.
* An agent that `pip install`s the fix and copies it back into the source
  tree has cheated — even if the task is pristine. That's Axis 2.
* Only by labelling both, then combining them in Stage 7, can you say
  whether a passed row counts as evidence of capability.

This is the same separation OpenAI used in its
[*Why we no longer evaluate SWE-bench Verified*](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/)
critique (April 2026), with directly compatible terminology:

| OpenAI term | bench-cleanser label |
|---|---|
| "Narrow test cases" | `APPROACH_LOCK` |
| "Wide test cases" | `OVER_TEST` |

The lock-in between gold patch and F2P tests — what OpenAI calls "co-authored
in the PR" — is encoded in the severity rules: the contamination is severe
because both halves of the measurement were written by the same author from
the same hypothesis, so independent verification is structurally impossible.

---

## Axis 1 — Task contamination

Seven binary labels, all multi-emittable. A task may carry any combination,
or none at all (`CLEAN`).

### `APPROACH_LOCK`

The F2P tests assert on a specific implementation strategy rather than on
observable behaviour described in the problem statement. An agent that
implements the described behaviour via a different (correct) strategy fails.

*Evidence requirement:* at least one `OFF_TOPIC` assertion **on the agent's
strategy choice** (data structure, algorithm, signature), or `is_modification_aligned=False`
on a modified test where the modification narrows accepted implementations.

*Severity:* contributes SEVERE.

### `OVER_TEST`

F2P tests assert on behaviour not described in the problem. An agent that
solves the described problem still fails.

*Evidence requirement:* one or more `OFF_TOPIC` assertions; or `UNRELATED`
F2P tests; or modifications-to-existing-tests that add OFF_TOPIC content.

*Severity:* contributes SEVERE.

> **Authorship realism.** Because gold patches and F2P tests are typically
> co-authored in the same PR, `OVER_TEST` is treated as a maximum-attention
> signal — not a soft warning. The author of the patch also wrote the
> tests, so a wide test is rarely an innocent oversight.

### `OVER_PATCH`

The gold patch modifies behaviour not described in the problem (e.g. a
"while we're here" refactor). Agents are then evaluated against a patch
that includes more than the problem asks for.

*Evidence requirement:* at least one `UNRELATED` hunk that changes
runtime behaviour. Pure ancillary edits (imports, dead-code removal) do
not trigger this.

*Severity:* contributes MODERATE.

### `UNCLEAR_DESCRIPTION`

The problem statement is too ambiguous to derive the expected behaviour.
The acceptance criteria the LLM intent-extracter could pull out are
underspecified relative to what the F2P tests check.

*Evidence requirement:* `IntentStatement.ambiguity_score >= 0.5` combined
with at least one F2P test verdict that is not `ALIGNED`.

*Severity:* contributes MINOR.

### `HIDDEN_CONTEXT`

The F2P tests reference identifiers, fixtures, constants, or behaviour
that is not present anywhere in the problem statement and not reachable
by reading the codebase pre-patch.

*Evidence requirement:* concrete identifier or assertion value that
exists only in the post-patch tests or gold patch.

*Severity:* contributes SEVERE.

### `WEAK_COVERAGE`

The F2P tests do not actually exercise the patched code paths — gold
patch passes, but the F2P tests would also pass on a no-op patch.

*Evidence requirement:* cross-reference analysis (Stage 4C) reports no
call edges from any F2P test to any changed block.

*Severity:* contributes MINOR.

### `CLEAN`

No contamination labels apply. Returned as a single label rather than as
an empty list so the consumer always sees an explicit "no contamination"
signal.

*Severity:* contributes CLEAN (the absence of every other label).

---

## Axis 2 — Agent trajectory

Single label per `(task, agent)`. The label describes the agent's
behaviour, not the task. Eight values, split by outcome:

### Passed labels (`resolved=True`)

| Label | Meaning |
|---|---|
| `agent_passed_genuine` | Legitimate problem-solving: explore → hypothesise → patch → test. Patch diverges from gold but solves the described problem. |
| `agent_passed_leak` | Final patch is `>= 90%` similar to the gold patch *and* the trajectory shows direct file/function jumps without exploration. |
| `agent_passed_package_leak` | Agent `pip install`-ed a newer version of the affected package and copied the fix from site-packages. |
| `agent_passed_test_aware` | Agent referenced F2P test names or expected values before they were derivable from exploration. |
| `agent_passed_trained_hack` | Memorised template applied with no task-specific reasoning. Pattern: agent emits a near-canonical fix on the first try, no debug steps. |

### Failed labels (`resolved=False`)

| Label | Meaning |
|---|---|
| `agent_failed_completed_intent` | Agent produced a patch that addresses the problem as described, but the F2P tests reject it. This is the failure mode that, combined with `APPROACH_LOCK` or `OVER_TEST` on the task, yields `UNFAIR_FAILURE`. |
| `agent_failed_no_intent` | Agent never produced an intent matching the problem description. Skill gap, not benchmark issue. |

### Unknown

| Label | Meaning |
|---|---|
| `agent_unknown` | Trajectory data was insufficient to classify (truncated, mis-formatted, or both). |

The trajectory classifier uses [`trajectory_analysis.md`](../bench_cleanser/prompts/trajectory_analysis.md)
as its system prompt and returns a strict
[`TrajectoryClassificationResponse`](../bench_cleanser/schemas.py)
via `LLMClient.query_structured` — no ad-hoc JSON parsing.

---

## Severity (bucket; Axis 1 only)

`Severity` is computed from the Axis-1 label set using **pure set
membership** — no floating-point thresholds, no per-label weights, no
counts. Severity bucketing lives in `bench_cleanser.classification.scorer.compute_task_severity`.

```
SEVERE     := APPROACH_LOCK ∨ OVER_TEST ∨ HIDDEN_CONTEXT  ∈ labels
MODERATE   := OVER_PATCH                                ∈ labels  ∧ not SEVERE
MINOR      := UNCLEAR_DESCRIPTION ∨ WEAK_COVERAGE       ∈ labels  ∧ not (SEVERE | MODERATE)
CLEAN      := labels = ∅ ∨ labels = {CLEAN}
```

The deterministic, set-based rule means:

* every severity is reproducible from the persisted report alone;
* re-running classification on the same evidence cannot change the
  severity without changing a label;
* there is no "tuning knob" that quietly shifts the population on a
  re-run.

That last property is the one that matters for evaluation use. If
you re-run the pipeline a year from now with a smarter LLM, the labels
may improve but the severity-from-labels function is frozen.

---

## Evidence requirements (the "cite or shut up" rule)

Every assigned label must carry concrete evidence drawn from artefacts
the classifier already had on hand:

| Label | Allowed evidence |
|---|---|
| `APPROACH_LOCK` | Specific assertion text + `OFF_TOPIC` reasoning, or a `is_modification_aligned=False` `TestVerdictReport`. |
| `OVER_TEST` | `OFF_TOPIC` assertion text, `UNRELATED` test name, or modified-test evidence line. |
| `OVER_PATCH` | `UNRELATED` `HunkVerdict` with reasoning. |
| `UNCLEAR_DESCRIPTION` | The `ambiguity_score` value and at least one non-aligned test verdict. |
| `HIDDEN_CONTEXT` | Specific identifier or assertion value, plus its absence from `problem_statement`. |
| `WEAK_COVERAGE` | Empty `call_edges` from `CrossReferenceResult`. |

A label without evidence is rejected by the classifier — labels are
emitted **with** their evidence lines or not at all.

---

## What this taxonomy deliberately does NOT cover

* **Reward hacking** (gaming the eval harness rather than the task).
  That's an evaluation-harness concern, not a benchmark-item concern.
* **Reproducibility failures of the agent runtime** (flaky CI, OOM
  kills). Captured by the agent's `resolved` outcome; classified as
  `agent_failed_no_intent` if the trajectory shows no engagement, or
  `agent_unknown` if data is missing.
* **Hallucinated `pass_to_pass` regressions** that pass-to-pass tests
  start failing. These are the agent's problem, not the task's.

Anything *bench-cleanser* does not label, it leaves alone — a CLEAN
verdict means the **bench-cleanser axes** are clean, not that the row
is suitable for every conceivable use.
