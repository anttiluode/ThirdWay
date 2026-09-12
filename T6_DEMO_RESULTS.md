# T6 continual-learning demo — guarded persistent consolidation

**Status: the corrected guard reduces forgetting, but the binary veto is too conservative to count as a useful continual learner.**

The scientific T6 result remains a frozen FAIL because shuffled tangent order did not reduce predictive power. This separate engineering demo asks:

> Can the surviving direction-specific dynamic-risk signal make safer permanent learning decisions than blindly consolidating every locally useful update?

After pre-merge review, one bug in the first demo was corrected: the seed RNN already performs all three skills, so every non-target baseline skill must be protected **before its first new edit**. The earlier version protected a skill only after that skill had an accepted edit, making the guard schedule-dependent. The corrected result below is the one retained.

## Frozen protocol

One shared 24-state tanh RNN and one shared readout are used throughout. No task-specific heads are added.

The learning schedule was fixed before observing the demo outcome:

```text
A -> B -> C -> B -> A -> C -> A -> B -> C -> B -> A -> C
```

where:

```text
A = recent-cue temporal skill
B = delayed-cue temporal skill
C = order / relational temporal skill
```

Candidate identities begin at seed 32, outside the T6 train/test identity pools `0..31`.

For each event:

1. the **accept-all** trajectory generates the next rank-1 candidate using target-skill data only;
2. the exact same candidate matrix is offered to both learners;
3. accept-all permanently commits every locally useful proposal;
4. the guarded learner calls `consider_candidate(...)`;
5. the guard recomputes target usefulness in its own current state;
6. it estimates directional dynamic risk to **every non-target baseline skill represented in the calibration set**, plus any retained-edit provenance;
7. the candidate becomes permanent only if it remains locally useful and every protected-skill risk is below the threshold calibrated on the frozen T6 training split.

The demo never evaluates actual held-out cross-skill damage before deciding.

Frozen causal-risk threshold:

```text
0.054821
```

## Starting state

Held-out evaluation accuracy before the 12 learning events:

```text
A = 0.859
B = 0.664
C = 0.633
mean = 0.719
```

## Corrected event trace

| event | skill | candidate | target gain in accept-all | target gain in guarded | max predicted risk | guard |
|---:|---|---:|---:|---:|---:|---|
| 1 | A | 32 | +0.06587 | +0.06587 | 0.06050 | REJECT risk |
| 2 | B | 33 | +0.04006 | +0.04912 | 0.14863 | REJECT risk |
| 3 | C | 34 | +0.05039 | +0.05130 | 0.11301 | REJECT risk |
| 4 | B | 35 | +0.04072 | +0.04990 | 0.07557 | REJECT risk |
| 5 | A | 36 | +0.08094 | +0.03504 | 0.07217 | REJECT risk |
| 6 | C | 37 | +0.03876 | +0.05913 | 0.07580 | REJECT risk |
| 7 | A | 38 | +0.09747 | +0.03832 | 0.09223 | REJECT risk |
| 8 | B | 39 | +0.07670 | -0.01957 | 0.09915 | REJECT no longer useful |
| 9 | C | 40 | +0.04690 | +0.04597 | 0.08139 | REJECT risk |
| 10 | B | 41 | +0.08722 | +0.05157 | 0.04428 | **ACCEPT** |
| 11 | A | 42 | +0.11780 | +0.08571 | 0.15006 | REJECT risk |
| 12 | C | 43 | +0.02415 | -0.01155 | 0.08800 | REJECT no longer useful |

Guarded decisions:

```text
accepted   1 / 12
rejected  11 / 12
  dynamic-risk rejection            9
  no longer useful after divergence 2
```

Accept-all permanently commits all 12 proposals.

## Final behavior

```text
                 A      B      C      mean    minimum
base           0.859  0.664  0.633   0.719   0.633
guarded        0.844  0.656  0.648   0.716   0.648
accept-all     0.664  0.641  0.672   0.659   0.641
```

At the final C event, A and B are the old skills. Their mean retention is:

```text
guarded     0.750
accept-all  0.652
```

That is about `+0.098` absolute old-skill accuracy for the guarded network.

The clearest forgetting difference is A:

```text
base        0.859
guarded     0.844
accept-all  0.664
```

The guard also ends with a slightly better minimum skill accuracy:

```text
guarded     0.648
accept-all  0.641
```

but this should not be oversold: the guard accepts only one permanent proposal, and its final mean `0.716` is still slightly below the starting mean `0.719`.

## Switching transient

Mean carried-state switch penalty across the six ordered skill changes:

```text
guarded     +0.00966
accept-all  +0.02105
```

Lower is better. The guarded network is less disrupted by carried recurrent state across context switches.

## What the corrected demo establishes

The executable mechanism is:

```text
target-only useful proposal
-> estimate direction-specific susceptibility of every protected non-target skill
-> accept or reject before permanent consolidation
-> reduce catastrophic old-skill damage
```

The two learners receive the same proposal matrices, and the threshold is learned only from the frozen T6 training split.

The supported engineering statement is:

> **Pre-commit dynamic susceptibility can act as a useful safety filter for persistent edits in this shared recurrent model.**

That wording is intentionally narrower than “continual learning solved.”

## What it exposes

The corrected guard makes the next failure impossible to ignore:

> **protective stagnation — a binary safety veto preserves the repertoire by refusing almost all learning.**

The reject-only guard retains old behavior far better than accept-all, but commits only `1/12` proposals and fails to increase aggregate held-out accuracy beyond the starting network.

This is exactly where the old IttnasNoruen finite-change lesson becomes useful. A direction can be valuable while the full finite step is unsafe. The right action need not be “accept all” or “throw it away.”

## T7 target — compatible partial writes

For an unsafe but locally useful candidate `Delta W`, test a fixed descending scale bank:

```text
1.00, 0.75, 0.50, 0.25, 0.125
```

Choose the **largest** fraction `alpha Delta W` that simultaneously:

```text
1. remains useful on the target skill in the current guarded state;
2. keeps predicted risk below threshold for every protected non-target skill;
3. is committed through the same persistent-write API.
```

No actual held-out old-skill damage may be inspected during the decision.

The first T7 comparison should keep exactly the same candidate stream and compare:

```text
accept-all
reject-only guard      <- corrected T6 demo
safe-step guard        <- T7A
finite oracle          <- upper bound only
```

Only if no scalar fraction is safe should a later T7B introduce the IttnasNoruen-style measured compensation direction.

The target has now sharpened from **detect dangerous learning** to:

> **keep as much useful learning as possible while respecting compatibility constraints.**
