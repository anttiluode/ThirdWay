# T6 continual-learning demo — guarded persistent consolidation

**Status: practical demo succeeds on its frozen 12-event protocol, but it is not a retroactive pass of the scientific T6 gate.**

The scientific T6 result remains a frozen FAIL because shuffled tangent order did not reduce predictive power. This demo asks a different engineering question:

> Can the surviving direction-specific dynamic-risk signal make better permanent learning decisions than blindly consolidating every locally useful update?

The answer on the frozen demo stream is **yes for old-skill retention and mean final accuracy**, with an important limitation: the guard is conservative and gives up some progress on the weaker skills.

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
4. the guarded learner calls the production `consider_candidate(...)` API;
5. the guard recomputes target usefulness in its own current state and estimates directional dynamic risk to already-retained *other* skills;
6. the candidate is permanently committed only if it remains locally useful and every retained-skill risk is below the threshold calibrated on the frozen T6 training split.

The demo never evaluates actual cross-skill damage before deciding.

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

## Event trace

| event | skill | candidate | target gain in accept-all | target gain in guarded | max predicted risk | guard |
|---:|---|---:|---:|---:|---:|---|
| 1 | A | 32 | +0.06587 | +0.06587 | 0.00000 | ACCEPT |
| 2 | B | 33 | +0.04006 | +0.04006 | 0.15246 | REJECT risk |
| 3 | C | 34 | +0.05039 | +0.04377 | 0.05820 | REJECT risk |
| 4 | B | 35 | +0.04072 | +0.04933 | 0.08231 | REJECT risk |
| 5 | A | 36 | +0.08094 | +0.04259 | 0.00000 | ACCEPT |
| 6 | C | 37 | +0.03876 | +0.05805 | 0.06976 | REJECT risk |
| 7 | A | 38 | +0.09747 | +0.03044 | 0.00000 | ACCEPT |
| 8 | B | 39 | +0.07670 | +0.01011 | 0.06157 | REJECT risk |
| 9 | C | 40 | +0.04690 | +0.03806 | 0.05101 | ACCEPT |
| 10 | B | 41 | +0.08722 | +0.08933 | 0.04934 | ACCEPT |
| 11 | A | 42 | +0.11780 | +0.05461 | 0.16604 | REJECT risk |
| 12 | C | 43 | +0.02415 | -0.00303 | 0.09525 | REJECT no longer useful |

Guarded decisions:

```text
accepted  5 / 12
rejected  7 / 12
  risk rejection                  6
  no longer useful after divergence 1
```

Accept-all permanently commits all 12 proposals.

## Final behavior

```text
                 A      B      C      mean    minimum
base           0.859  0.664  0.633   0.719   0.633
guarded        0.859  0.625  0.664   0.716   0.625
accept-all     0.664  0.641  0.672   0.659   0.641
```

At the final C event, A and B are the already-existing skills. Their mean retention is:

```text
guarded     0.742
accept-all  0.652
```

The guarded policy therefore keeps about `+0.090` absolute old-skill accuracy relative to blindly consolidating every locally useful edit.

The clearest catastrophic-forgetting difference is A:

```text
base        0.859
guarded     0.859
accept-all  0.664
```

C still improves under the guard:

```text
base        0.633
guarded     0.664
```

so this is not simply a frozen network. B, however, falls from `0.664` to `0.625`, and accept-all ends slightly higher on B and C individually.

## Switching transient

Mean carried-state switch penalty across the six ordered skill changes:

```text
guarded     +0.01256
accept-all  +0.02105
```

Lower is better. The guarded model is less disrupted when recurrent state is carried across a context switch.

## What the demo establishes

The demonstration is executable evidence for a practical mechanism:

```text
target-only useful proposal
-> estimate direction-specific risk to retained computations
-> accept or reject before permanent consolidation
-> reduce catastrophic old-skill damage
```

It uses the same recurrent substrate, candidate generator and directional dynamic-risk function measured in T6. The two learners receive the same proposal matrices, so the guard is not winning by asking for easier candidates.

The useful result is:

> **Pre-commit dynamic susceptibility can make persistent learning safer than accepting every locally useful update in this shared recurrent model.**

## What it does not establish

This is still a small synthetic recurrent world with three seeded task definitions. It does not establish state-of-the-art continual learning, biological plausibility, or a universal advantage over EWC/replay/orthogonal-gradient methods.

More importantly, it exposes a new engineering failure:

> **rejection preserves knowledge by wasting potentially useful learning.**

The guarded model finishes with a better mean and much better A retention than accept-all, but it does not finish better on every skill and it does not improve its aggregate mean beyond the starting network.

That makes the next experiment concrete rather than philosophical.

## T7 target — compatible partial writes

When a full candidate is useful but unsafe, do not immediately discard it.

Try a fixed descending scale bank, for example

```text
1.00, 0.75, 0.50, 0.25, 0.125
```

and choose the largest candidate fraction that simultaneously:

```text
1. remains useful on the target skill;
2. falls below retained-skill dynamic-risk threshold;
3. is committed using the exact same permanent-write API.
```

If no scaled version is safe, add the IttnasNoruen lesson: search a small measured compensation direction that restores the endangered retained response while preserving as much target gain as possible.

The decisive comparison should keep the same proposal stream and compare:

```text
accept-all
reject-only guard      (current T6 demo)
safe-step guard        (T7)
finite oracle upper bound
```

T7 should improve **learning efficiency under compatibility constraints**, not merely rejection accuracy.
