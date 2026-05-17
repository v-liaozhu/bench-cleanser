# Fusion (Stage 7)

Stage 7 combines the two axes — task contamination (Axis 1) and agent
trajectory (Axis 2) — into a single fairness verdict for each
`(task, agent)` pair. It answers the question a human reviewer actually
cares about:

> *Given what the benchmark asked and what the agent did, was this row a
> fair measurement of capability, a contaminated pass, an unfair
> failure, or something we cannot decide?*

Stage 7 is **deterministic and rule-based**. There is no LLM call at
this stage. Every verdict is reproducible from the two axis labels and
the trajectory's `resolved` outcome. Implementation lives in
[`bench_cleanser/fusion.py`](../bench_cleanser/fusion.py).

---

## The eight verdicts

| Verdict | Meaning | Invalidates measurement? |
|---|---|---|
| `FAIR_PASS` | Agent passed a clean/minor task via a genuine trajectory. | No |
| `AGENT_CHEATED` | Agent passed but the trajectory shows a leakage signal (`agent_passed_leak`, `agent_passed_test_aware`, `agent_passed_package_leak`, `agent_passed_trained_hack`). | **Yes** |
| `CONTAMINATED_PASS` | Agent passed a `SEVERE` or `MODERATE` task — even a genuine trajectory cannot produce a valid measurement against a broken benchmark item. | **Yes** |
| `AMBIGUOUS_PASS` | Agent passed a clean task but the trajectory could not be characterised (`agent_unknown`). Manual review recommended. | No |
| `UNFAIR_FAILURE` | Agent completed the described intent but failed F2P tests due to `APPROACH_LOCK` or `OVER_TEST` on the task. | **Yes** |
| `FAIR_FAILURE` | Agent engaged the problem, failed; task is clean. Genuine capability gap. | No |
| `AGENT_DISENGAGED` | Agent never produced an intent matching the problem description (`agent_failed_no_intent`). Behaviour failure, not benchmark signal. | No |
| `INCONCLUSIVE` | Neither axis provides a decisive signal. | No |

`invalidates_measurement = True` is the single boolean a downstream
consumer should consult when deciding whether to *drop* a row from a
benchmark suite. It is `True` exactly for the three verdicts where we
know the pass/fail outcome cannot be trusted for any reason.

---

## The rule engine

Rules evaluated top-down; first match wins. The full implementation is
in [`bench_cleanser.fusion.fuse`](../bench_cleanser/fusion.py).

### Rule 1 — Agent passed with leak

```
trajectory.label ∈ {agent_passed_leak,
                    agent_passed_test_aware,
                    agent_passed_package_leak,
                    agent_passed_trained_hack}
  → AGENT_CHEATED, invalidates=True
```

Trajectory leakage dominates everything else. Even on a contaminated
task, an agent that cheated is the more important signal — the row
cannot be used to measure that agent.

### Rule 2 — Pass over contaminated task

```
trajectory.label = agent_passed_genuine
∧ severity ∈ {SEVERE, MODERATE}
  → CONTAMINATED_PASS, invalidates=True
```

Even a genuine-looking trajectory cannot make a contaminated
measurement valid. If the underlying task is broken, the row tells us
nothing about capability.

### Rule 3 — Fair pass

```
trajectory.label = agent_passed_genuine
∧ severity ∈ {CLEAN, MINOR}
  → FAIR_PASS, invalidates=False
```

The good case. Valid evidence of capability.

### Rule 4 — Passed but trajectory unknown

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
outcome on the F2P tests) because the trajectory label alone tells us
the trajectory was *uncharacterised* — not whether the agent actually
passed. The split matters: a passed-but-unknown trajectory on a clean
task is genuinely an ambiguous pass; on a contaminated task we cannot
discriminate `FAIR_PASS` from `AGENT_CHEATED` from `CONTAMINATED_PASS`,
so we mark it `INCONCLUSIVE`.

A previous version of this rule used analysis-identity as the
discriminator and was always true on one branch, mis-firing
`AMBIGUOUS_PASS`. The bug is regression-tested in
`tests/test_fusion_rule4.py`.

### Rule 5 — Unfair failure

```
trajectory.label = agent_failed_completed_intent
∧ (APPROACH_LOCK ∈ labels  ∨  OVER_TEST ∈ labels)
  → UNFAIR_FAILURE, invalidates=True
```

Agent engaged the problem, produced a patch that addresses the
described behaviour, and failed only because the task carries
contamination labels that reject valid alternative solutions or demand
out-of-scope behaviour. This is the row to drop when scoring an agent.

### Rule 6 — Fair failure

```
trajectory.label = agent_failed_completed_intent
∧ ¬(APPROACH_LOCK ∈ labels  ∨  OVER_TEST ∈ labels)
  → FAIR_FAILURE, invalidates=False
```

Agent engaged the problem, failed; task does not carry the
contamination signals that would excuse the failure. Genuine
capability gap.

### Rule 7 — Disengaged

```
trajectory.label = agent_failed_no_intent
  → AGENT_DISENGAGED, invalidates=False
```

Agent never produced an intent matching the problem description. This
is an agent-behaviour failure, not a benchmark-quality signal — drop
the row from agent scoring but do not penalise the benchmark.

### Rule 8 — Catch-all

```
otherwise
  → INCONCLUSIVE, invalidates=False
```

---

## Worked examples

### Example 1 — clean task, genuine pass

```
report.severity = CLEAN
report.task_labels = []
trajectory.label = agent_passed_genuine
trajectory.resolved = True
```

→ Rule 3 fires → `FAIR_PASS`. Keep the row; count toward the agent's score.

### Example 2 — over-test failure

```
report.severity = SEVERE
report.task_labels = [OVER_TEST]
trajectory.label = agent_failed_completed_intent
trajectory.resolved = False
```

→ Rule 5 fires → `UNFAIR_FAILURE`, `invalidates_measurement=True`. The
agent solved the described problem but the F2P tests asked for behaviour
the problem did not. Drop the row from agent scoring; flag the row for
benchmark cleanup.

### Example 3 — trained-hack pass on a clean task

```
report.severity = CLEAN
report.task_labels = []
trajectory.label = agent_passed_trained_hack
trajectory.resolved = True
```

→ Rule 1 fires → `AGENT_CHEATED`. The task is clean; the agent is the
problem. Drop the row from agent scoring; the benchmark item is fine.

### Example 4 — ambiguous pass

```
report.severity = CLEAN
report.task_labels = []
trajectory.label = agent_unknown
trajectory.resolved = True
```

→ Rule 4 fires (CLEAN + resolved=True) → `AMBIGUOUS_PASS`. Manual
review recommended; not auto-dropped.

### Example 5 — passed contaminated task with unclear trajectory

```
report.severity = SEVERE
report.task_labels = [HIDDEN_CONTEXT]
trajectory.label = agent_unknown
trajectory.resolved = True
```

→ Rule 4 fires (non-CLEAN + resolved=True) → `INCONCLUSIVE`. We cannot
discriminate cheat from genuine pass without trajectory data.

---

## What fusion does NOT decide

* Whether the task should be **fixed** versus **discarded**. That's a
  benchmark-maintainer choice. Stage 7 only labels.
* Whether to **score** a row as a pass or fail. Stage 7 outputs a
  verdict + `invalidates_measurement`; the consumer applies its own
  policy.
* The agent's **calibration**, **runtime efficiency**, or any other
  out-of-axis property. Only fairness of measurement.
