# Taxonomy

Two independent axes, fused at Stage 7 ([FUSION.md](FUSION.md)).

| Axis | Question | Cardinality | Module |
| --- | --- | --- | --- |
| **Axis 1 — Task contamination** | Is the benchmark item fair? | Multi-label (0–7) | `bench_cleanser.classification.dual_taxonomy` |
| **Axis 2 — Agent trajectory** | How did this agent reach its answer? | Single label per `(task, agent)` | `bench_cleanser.trajectory.classifier` |

The axes serve different questions and must not be conflated. A task
with `APPROACH_LOCK` is broken regardless of the agent. An agent that
`pip install`s the fix has cheated regardless of the task. Only Stage 7
combines them.

### OpenAI Verified-audit mapping

| OpenAI term | bench-cleanser label |
| --- | --- |
| "Narrow test cases" | `APPROACH_LOCK` |
| "Wide test cases" | `OVER_TEST` |

The PR-authorship insight — gold patch and F2P tests are co-authored —
is encoded in severity: both halves of the measurement come from one
author, so independent verification is structurally impossible.

---

## Axis 1 — Task contamination

Seven binary labels. A task may carry any combination, or `CLEAN`.

### `APPROACH_LOCK`

F2P tests assert on a specific implementation strategy rather than
observable behaviour. An agent implementing the same behaviour
differently fails.

- **Evidence:** ≥1 `OFF_TOPIC` assertion on strategy choice (data
  structure, algorithm, signature) **or** `is_modification_aligned=False`
  on a modified test that narrows accepted implementations.
- **Severity:** SEVERE.

### `OVER_TEST`

F2P tests assert on behaviour not described in the problem.

- **Evidence:** ≥1 `OFF_TOPIC` assertion, **or** an `UNRELATED` F2P
  test, **or** a modification adding `OFF_TOPIC` content.
- **Severity:** SEVERE. (Treated as maximum-attention because the patch
  author also wrote the tests — wide tests are rarely innocent.)

### `OVER_PATCH`

Gold patch modifies behaviour not described in the problem.

- **Evidence:** ≥1 `UNRELATED` hunk that changes runtime behaviour.
  Pure ancillary edits (imports, dead-code removal) do not trigger.
- **Severity:** MINOR on its own; MODERATE when combined with
  `HIDDEN_CONTEXT` or `UNCLEAR_DESCRIPTION`.

### `UNCLEAR_DESCRIPTION`

Problem statement too ambiguous to derive expected behaviour.

- **Evidence:** classifier finds the problem text ambiguous enough that
  multiple incompatible solutions are reasonable. This label is not
  triggered by a fixed ambiguity-score threshold.
- **Severity:** MINOR.

### `HIDDEN_CONTEXT`

Problem framing relies on hidden cues (for example self-referential
phrasing like “see the patch”) rather than a standalone issue
specification.

- **Evidence:** concrete hidden-context cue in problem/hints metadata
  that cannot be acted on from problem text alone.
- **Severity:** MINOR on its own; may contribute to MODERATE when paired with
  `OVER_PATCH`.

### `WEAK_COVERAGE`

F2P tests do not exercise the patched code paths.

- **Evidence:** classifier evidence that acceptance criteria are not
  exercised by F2P tests or by gold-patch-covered behaviors.
- **Severity:** MINOR.

### `CLEAN`

No contamination labels apply. Emitted as a single label, never with
others — gives consumers an explicit "no contamination" signal.

---

## Axis 2 — Agent trajectory

Single label per `(task, agent)`.

### Passed (`resolved=True`)

| Label | Pattern |
| --- | --- |
| `agent_passed_genuine` | Explore → hypothesise → patch → test. Patch diverges from gold but solves the described problem. |
| `agent_passed_leak` | Final patch ≥90% similar to gold; trajectory shows direct file/function jumps without exploration. |
| `agent_passed_package_leak` | Agent `pip install`-ed the affected package and copied the fix from site-packages. |
| `agent_passed_test_aware` | Agent referenced F2P test names or expected values before they were derivable from exploration. |
| `agent_passed_trained_hack` | Near-canonical fix on first try, no debug steps — memorised template. |

### Failed (`resolved=False`)

| Label | Pattern |
| --- | --- |
| `agent_failed_completed_intent` | Patch addresses the described behaviour but F2P tests reject it. The driver for `UNFAIR_FAILURE` when combined with `APPROACH_LOCK` / `OVER_TEST`. |
| `agent_failed_no_intent` | Agent never engaged the problem. Skill gap, not benchmark issue. |

### Unknown

| Label | Pattern |
| --- | --- |
| `agent_unknown` | Trajectory truncated, malformed, or otherwise insufficient. |

The trajectory classifier uses [`trajectory_analysis.md`](../bench_cleanser/prompts/trajectory_analysis.md)
as the system prompt and returns a strict
[`TrajectoryClassificationResponse`](../bench_cleanser/schemas.py) via
`LLMClient.query_structured`.

---

## Severity (bucket; Axis 1)

`Severity` is computed by pure set membership over the Axis-1 label set
in `bench_cleanser.classification.dual_taxonomy.compute_task_severity`.
No thresholds, no weights, no counts.

```
SEVERE   := APPROACH_LOCK ∈ labels  OR  OVER_TEST ∈ labels
MODERATE := OVER_PATCH ∈ labels AND
            (HIDDEN_CONTEXT ∈ labels OR UNCLEAR_DESCRIPTION ∈ labels)
MINOR    := any contamination label set that is neither SEVERE nor MODERATE
CLEAN    := labels = ∅  OR  labels = { CLEAN }
```

Severity is reproducible from the persisted report alone and frozen
across LLM upgrades.

---

## Evidence rule

A label without evidence is rejected by the classifier. Allowed
evidence sources per label:

| Label | Evidence sources |
| --- | --- |
| `APPROACH_LOCK` | `OFF_TOPIC` assertion text + reasoning; `is_modification_aligned=False` `TestVerdictReport`. |
| `OVER_TEST` | `OFF_TOPIC` assertion text; `UNRELATED` test name; modified-test evidence line. |
| `OVER_PATCH` | `UNRELATED` `HunkVerdict` with reasoning. |
| `UNCLEAR_DESCRIPTION` | Problem text evidence showing incompatible interpretations or missing specification detail. |
| `HIDDEN_CONTEXT` | Hidden specification cues in problem/hints metadata (for example self-referential phrasing not actionable from issue text alone). |
| `WEAK_COVERAGE` | Evidence that acceptance criteria are untested or under-constrained by F2P tests. |

---

## Out of scope

- **Reward hacking** (gaming the eval harness, not the task).
- **Agent runtime flakes** (OOM, CI brownouts). Captured by `resolved`
  and classified as `agent_failed_no_intent` or `agent_unknown`.
- **Hallucinated P2P regressions.** The agent's problem, not the task's.

A `CLEAN` verdict means the bench-cleanser axes turned up nothing —
not that the row is fit for every conceivable use.
