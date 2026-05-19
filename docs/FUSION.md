# Stage 7 — Fusion

Stage 7 combines the task contamination labels (Axis 1) and the agent
trajectory label (Axis 2) into a single fairness verdict per
`(task, agent)` pair. Deterministic; no LLM call. Implementation:
[`bench_cleanser/fusion.py`](../bench_cleanser/fusion.py).

The verdict answers: *given what the benchmark asked and what the agent
did, was this row a fair measurement of capability?*

---

## Verdicts

| Verdict | Meaning | `invalidates_measurement` |
| --- | --- | --- |
| `FAIR_PASS` | Genuine trajectory on a clean/minor task. | No |
| `AGENT_CHEATED` | Trajectory shows leakage (`agent_passed_leak`, `_test_aware`, `_package_leak`, `_trained_hack`). | **Yes** |
| `CONTAMINATED_PASS` | Genuine trajectory on a SEVERE/MODERATE task. The task is broken; pass cannot be trusted. | **Yes** |
| `AMBIGUOUS_PASS` | Passed a clean task with `agent_unknown` trajectory. Manual review. | No |
| `UNFAIR_FAILURE` | Failed after completing the described intent; task carries `APPROACH_LOCK` or `OVER_TEST`. | **Yes** |
| `FAIR_FAILURE` | Engaged the problem, failed; task clean. Genuine capability gap. | No |
| `AGENT_DISENGAGED` | Never produced an intent matching the problem. | No |
| `INCONCLUSIVE` | Neither axis is decisive. | No |

`invalidates_measurement` is `True` exactly when the pass/fail outcome
cannot be trusted for any reason. It's the single boolean a consumer
should consult to decide whether to drop a row.

---

## Rules (top-down; first match wins)

### Rule 1 — Pass with leak

```
trajectory.label ∈ { agent_passed_leak, agent_passed_test_aware,
                     agent_passed_package_leak, agent_passed_trained_hack }
  → AGENT_CHEATED, invalidates=True
```

Leakage dominates everything else. Even on a contaminated task, the
agent is the more important signal.

### Rule 2 — Pass over contaminated task

```
trajectory.label = agent_passed_genuine
∧ severity ∈ { SEVERE, MODERATE }
  → CONTAMINATED_PASS, invalidates=True
```

### Rule 3 — Fair pass

```
trajectory.label = agent_passed_genuine
∧ severity ∈ { CLEAN, MINOR }
  → FAIR_PASS, invalidates=False
```

### Rule 4 — Pass, trajectory unknown

```
trajectory.label = agent_unknown
∧ trajectory.resolved = True
∧ severity = CLEAN
  → AMBIGUOUS_PASS, invalidates=False

trajectory.label = agent_unknown
∧ trajectory.resolved = True
∧ severity ≠ CLEAN
  → INCONCLUSIVE, invalidates=False

trajectory.label = agent_unknown
∧ trajectory.resolved = False
  → INCONCLUSIVE, invalidates=False
```

Rule 4 splits on `trajectory.resolved` (the agent's reported pass/fail
on F2P tests) because an unknown label tells us the trajectory was
uncharacterised, not whether the agent passed. The full matrix is
regression-tested in `tests/test_fusion_rule4.py`.

### Rule 5 — Unfair failure

```
trajectory.label = agent_failed_completed_intent
∧ ({ APPROACH_LOCK, OVER_TEST } ∩ labels) ≠ ∅
  → UNFAIR_FAILURE, invalidates=True
```

### Rule 6 — Fair failure

```
trajectory.label = agent_failed_completed_intent
∧ ({ APPROACH_LOCK, OVER_TEST } ∩ labels) = ∅
  → FAIR_FAILURE, invalidates=False
```

### Rule 7 — Disengaged

```
trajectory.label = agent_failed_no_intent
  → AGENT_DISENGAGED, invalidates=False
```

### Rule 8 — Catch-all

```
otherwise
  → INCONCLUSIVE, invalidates=False
```

---

## Worked examples

**Clean task, genuine pass.** `severity=CLEAN`, `task_labels=[]`,
`trajectory.label=agent_passed_genuine`, `resolved=True`. → Rule 3 →
`FAIR_PASS`. Count toward the agent's score.

**Over-test failure.** `severity=SEVERE`, `task_labels=[OVER_TEST]`,
`trajectory.label=agent_failed_completed_intent`, `resolved=False`. →
Rule 5 → `UNFAIR_FAILURE`, `invalidates=True`. Drop from agent scoring;
flag the row for benchmark cleanup.

**Trained-hack pass on clean task.** `severity=CLEAN`,
`trajectory.label=agent_passed_trained_hack`, `resolved=True`. → Rule 1
→ `AGENT_CHEATED`. The agent is the problem; the task is fine.

**Ambiguous pass.** `severity=CLEAN`, `task_labels=[]`,
`trajectory.label=agent_unknown`, `resolved=True`. → Rule 4 (CLEAN +
resolved) → `AMBIGUOUS_PASS`. Manual review.

**Pass over contaminated task with unclear trajectory.**
`severity=MODERATE`,
`task_labels=[OVER_PATCH, HIDDEN_CONTEXT]`,
`trajectory.label=agent_unknown`, `resolved=True`.
→ Rule 4 (non-CLEAN + resolved) → `INCONCLUSIVE`.

---

## Out of scope

- Whether to **fix** vs **discard** a contaminated task. Maintainer
  choice; Stage 7 only labels.
- Whether to score a row as pass or fail. Stage 7 outputs verdict +
  `invalidates_measurement`; the consumer's scoring policy is its own.
- Calibration, runtime, or any non-fairness property.
