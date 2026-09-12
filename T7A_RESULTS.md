# Gate T7A — compatible partial writes

**Status: frozen primary criterion FAILS.**

T7A asked whether the protective stagnation exposed by T6 could be repaired by
changing only the *amplitude* of each proposed learning direction.

The candidate direction was never regenerated or rotated.  For each proposal
`Delta W`, T7A tested the fixed descending bank

```text
1.0, 0.75, 0.5, 0.25, 0.125
```

and committed the largest fraction that simultaneously:

1. still improved its target skill by more than `1e-4`;
2. kept predicted directional dynamic risk to every non-target baseline skill
   below the frozen T6 threshold `0.054821`.

The same twelve candidate matrices were presented to all comparison learners.
No T7A threshold, scale, dataset, candidate identity, or success criterion was
changed after observing the result.

## Frozen success criteria

Before the four-learner experiment was run, T7A was required to satisfy all of:

```text
nonzero safe-step writes >= reject-only writes + 2
old A/B final accuracy    >= reject-only old A/B - 0.02
switch penalty            <= reject-only switch + 0.005
final mean accuracy       >= reject-only mean + 0.01
final mean accuracy       >= starting mean + 0.005
```

The candidate identities also had to be identical across all four learners, and
the finite-damage control was forbidden from influencing the safe-step policy.

## Four-learner result

Starting held-out accuracy:

```text
A = 0.859
B = 0.664
C = 0.633
mean = 0.719
```

Final result on the frozen twelve-event stream:

| learner | A | B | C | mean | old A/B mean | switch penalty | nonzero writes |
|---|---:|---:|---:|---:|---:|---:|---:|
| accept-all | 0.664 | 0.641 | 0.672 | 0.659 | 0.652 | 0.02105 | 12 |
| reject-only T6 | **0.844** | **0.656** | 0.648 | **0.716** | **0.750** | **0.00966** | 1 |
| T7A safe-step | 0.742 | **0.656** | **0.680** | 0.693 | 0.699 | 0.01784 | 12 |
| finite per-step control | 0.812 | 0.641 | 0.664 | 0.706 | 0.727 | 0.01931 | 12 |

T7A escaped protective stagnation completely: every one of the twelve candidate
directions acquired a nonzero safe fraction.

But the frozen gate still fails.

Safety floor required:

```text
old A/B >= 0.750 - 0.020 = 0.730
```

Observed:

```text
safe-step old A/B = 0.699
```

Switch requirement:

```text
safe-step switch <= 0.00966 + 0.005 = 0.01466
```

Observed:

```text
safe-step switch = 0.01784
```

Learning requirements also fail:

```text
safe-step mean      = 0.693
reject-only mean    = 0.716
starting mean       = 0.719
```

So scalar shrinking is better than accepting every full edit (`0.693` versus
`0.659` mean), but it neither preserves the T6 safety floor nor produces net
aggregate continual learning.

## Exact chosen fractions

T7A safe-step:

```text
seed   32   33   34   35   36   37   38   39   40   41   42   43
alpha .75  .25  .50  .50  .75  .50  .50  .75  .50  .75  .25  .50
```

Finite per-step control:

```text
seed   32   33   34   35   36   37   38   39   40   41   42   43
alpha 1.0  .25  .25  .25  .25 .125  .25  .25  .25  .50  .25  .25
```

The finite control chooses the largest fraction whose *actual finite calibration
loss increase for each non-target skill on that single step* is at most the T6
collision threshold `0.02`.  It is therefore a per-step information upper
bound, not a globally optimizing sequence oracle.  Its final mean `0.706` still
falls below the starting mean `0.719`.

## Post-failure accumulation diagnostic

The failure was then diagnosed without changing any T7A decision.

For every accepted safe-step write, the actual finite non-target calibration
loss increment was measured before and after that one write.

```text
positive finite harm steps                 12 / 12
sum of positive per-step max harm          0.380687
corr(predicted risk, finite step harm)     0.644540
```

All twelve writes were below the frozen *predicted* risk ceiling, yet all twelve
caused positive finite damage to at least one protected skill.

Calibration loss drift from the original network after all twelve writes:

```text
skill A   +0.061845
skill B   +0.077985
skill C   -0.022412
```

The event-level diagnostic was:

| event | skill | seed | scale | predicted risk | actual max non-target dMSE | cumulative dMSE A | cumulative dMSE B | cumulative dMSE C |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | A | 32 | .750 | .045372 | +.011745 | -.040830 | +.011745 | -.012636 |
| 2 | B | 33 | .250 | .037861 | +.013887 | -.026942 | -.016748 | -.020105 |
| 3 | C | 34 | .500 | .054008 | +.040512 | -.007557 | +.023764 | -.048419 |
| 4 | B | 35 | .500 | .045319 | +.023304 | +.015747 | +.018645 | -.063435 |
| 5 | A | 36 | .750 | .045525 | +.037250 | -.059109 | +.055895 | -.046032 |
| 6 | C | 37 | .500 | .048316 | +.043165 | -.015943 | +.052149 | -.059589 |
| 7 | A | 38 | .500 | .047169 | +.029078 | -.063489 | +.069171 | -.030511 |
| 8 | B | 39 | .750 | .054234 | +.050697 | -.012792 | +.079662 | -.021124 |
| 9 | C | 40 | .500 | .037901 | +.038480 | +.025688 | +.067350 | -.051103 |
| 10 | B | 41 | .750 | .042405 | +.032468 | +.058156 | +.068705 | -.041310 |
| 11 | A | 42 | .250 | .035820 | +.017081 | +.018825 | +.085786 | -.029383 |
| 12 | C | 43 | .500 | .049188 | +.043019 | +.061845 | +.077985 | -.022412 |

This distinguishes the failure from a simple broken radar.  The predicted risk
retains useful ordering information, but T7A treats the threshold as an
independent allowance for every write.  Small harmful remainders are allowed to
accumulate across the sequence.

The supported lesson is therefore:

> **Per-write compatibility does not compose into sequence-level compatibility.**

or, in the language suggested by the SighImageSuper connection:

> **Shrinking a proposed operator change can reduce its harmful residual, but it
> does not remove that residual; repeated residuals can accumulate into
> forgetting.**

## Boundary for the next experiment

T7A does **not** justify changing the scale bank or loosening the safety floor.
It establishes the reason to move beyond a scalar trust radius.

A next-stage mechanism would need to deal explicitly with the residual left by
a useful-but-harmful direction.  Candidate possibilities include a cumulative
compatibility budget or a measured compensation / residual-removal step, but
those are new architectural hypotheses and are not results of T7A.

In particular, no claim is made yet that sparse coding, residual projection, or
new-highway growth solves this problem.  T7A only supplies the executable reason
to test them.
