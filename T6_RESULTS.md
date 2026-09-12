# Gate T6 — persistent skill compatibility

**Status: frozen primary gate FAILS one required falsification control.**

T6 moved ThirdWay from time-local edit events to permanent learning changes in one shared nonlinear recurrent model. The important result is mixed rather than binary: a directional dynamic susceptibility score predicts cross-skill damage substantially better than the static parameter-space attackers, but shuffling the detailed Jacobian time order does not reduce that predictive power. Therefore T6 does **not** earn the strong claim that ordered recurrent propagation is the source of the advantage.

The frozen failure is retained as a result, not tuned away.

## World fixed before predictor evaluation

One model is shared by all tasks:

```math
h_{t+1}=\tanh(Wh_t + Bx_t + b),
\qquad y=C^\top h_T.
```

Fixed implementation:

```text
hidden states       24
input dimensions     6
sequence length     16
shared readout       1
skills               3
candidate edit rank  1
```

The three context-labelled temporal skills are:

1. recent cue;
2. delayed cue;
3. temporal order / relation cue.

Candidate generation sees only target-skill data. For each candidate identity it searches a fixed bundle of eight random rank-1 directions and a fixed amplitude set. A candidate enters the compatibility battery only if it improves its own target loss by more than `1e-4`.

All candidate identities used in the final battery satisfy that requirement.

## Persistent collision battery

A retained edit `R` is permanently installed first:

```math
W_R = W_0 + R.
```

Then candidate `C` is permanently added:

```math
W_{R+C}=W_0+R+C.
```

Primary damage is the increase in retained-skill mean squared error:

```math
D_{old}=L_{old}(W_{R+C})-L_{old}(W_R).
```

The collision threshold was frozen at

```text
D_old > 0.02.
```

The battery contains all six ordered pairs among the three different skills and 32 candidate identities per ordered pair:

```text
cases               192
train identities     96 pair cases
held-out identities  96 pair cases
```

Seeds `0..15` and `16..31` form closed identity pools: no edit identity appears on both sides of the train/test split, even in the retained role.

Held-out prevalence:

```text
collision fraction  0.458333
safe fraction       0.541667
```

So the result is not driven by a one-class or nearly trivial battery.

## Frozen predictors

The practical predictors never perform the joint `R+C` evaluation.

### Parameter overlap

Absolute cosine between the two low-rank parameter edits.

### Gradient alignment

Absolute cosine between candidate `C` and the exact BPTT gradient of retained-skill squared loss at `W_R`.

### Diagonal empirical Fisher

Per-example squared BPTT gradients are averaged parameter-wise. The candidate score is Fisher-weighted edit energy. This is an actual parameter-sensitivity statistic, not the activation-cosine proxy used as an attacker in T5.

### Static Jacobian overlap

Each edit's local recurrent Jacobian-change signature is measured on its own task trajectories and compared without transporting the change through later dynamics.

### Directional dynamic susceptibility

For retained-task trajectories under `W_R`, the persistent candidate acts as a first-order injection at every recurrent step:

```math
\delta h_{t+1}
=
D_t\left(W_R\delta h_t+C h_t\right),
```

where

```math
D_t=\operatorname{diag}(1-h_{t+1}^2).
```

The final hidden perturbation is contracted with retained-task loss susceptibility. The score keeps predicted harmful loss increase rather than cancelling positive and negative examples.

This score is directional: `C -> R` and `R -> C` ask different questions.

### Controls

- **shuffled transport:** candidate injections remain attached to their real states/times, but the sequence of state-dependent Jacobian damping matrices used to transport previous perturbations is deterministically permuted;
- **reversed direction:** asks whether retained edit `R` damages candidate skill `C` instead of the required `C -> R` direction;
- **oracle:** directly measures finite `R+C` damage and is used only as an upper bound / truth source.

## Primary held-out result

| predictor | damage correlation `r` | AUROC | balanced accuracy |
|---|---:|---:|---:|
| parameter cosine | -0.012359 | 0.551136 | 0.538462 |
| gradient alignment | 0.508182 | 0.698427 | 0.647727 |
| diagonal empirical Fisher | 0.166689 | 0.579983 | 0.565559 |
| static Jacobian overlap | -0.040823 | 0.542832 | 0.538462 |
| **directional dynamic susceptibility** | **0.769216** | **0.849213** | **0.774476** |
| shuffled transport | 0.774169 | **0.858392** | **0.784965** |
| reversed direction | -0.175669 | 0.402972 | 0.500000 |
| finite joint oracle | 1.000000 | 1.000000 | 0.965909 |

The directional score beats the strongest static attacker, gradient alignment, by

```text
0.849213 - 0.698427 = 0.150786 AUROC.
```

It also beats every other static attacker by more than the frozen `0.10` absolute margin.

The advantage over static attackers is positive in exactly

```text
4 / 6 ordered skill-pair families,
```

meeting that frozen requirement.

Reversing causal direction destroys the signal:

```text
correct direction AUROC   0.849213
reverse direction AUROC   0.402972
```

So compatibility is strongly asymmetric in this battery.

## Why the frozen gate fails

The required temporal-order control was:

```text
causal AUROC >= shuffled-order AUROC + 0.10.
```

Observed:

```text
causal           0.849213
shuffled         0.858392
required margin +0.100000
```

The control fails decisively. Shuffling is not merely within the allowed margin; it is slightly better.

Therefore the primary T6 gate is **FAIL**, despite the good absolute causal/directional prediction.

## Root-cause diagnostic

A separate diagnostic was run without changing the frozen world or predictor.

```text
corr(ordered, shuffled) held-out                  0.990072
mean relative ordered-shuffled score difference  0.059653
late-4 injection contribution fraction            0.663302
final-injection contribution fraction             0.321940
last-step-only AUROC                              0.708479
```

This explains the control failure.

A persistent matrix edit does not inject its effect once and then rely entirely on long transport. It acts **again at every recurrent step**. In this short recurrent task, late injections dominate the linearized retained-skill effect: the last four steps account for about two thirds of the absolute first-order contribution, and the final injection alone already contains meaningful predictive signal.

Consequently, permuting the transport of *earlier* perturbations changes the score only modestly. The shuffled and ordered scores remain almost the same ranking.

The correct conclusion is therefore not

> ordered recurrent propagation explains the advantage.

The narrower supported conclusion is

> **a direction-specific dynamic susceptibility of the retained computation to a persistent edit predicts finite cross-skill damage better than the tested static edit-similarity measures.**

The experiment does not establish that the fine temporal order of the tangent product is necessary for that advantage.

## Predeclared candidate-strength sweep

After the primary gate failed, the candidate edit alone was rescaled to `0.5x`, `1.0x`, and `1.5x`. Retained edits, tasks, data, collision threshold, and predictors remained unchanged. The `1.0x` row is the primary result and was not replaced.

| candidate strength | useful candidate fraction | collision fraction | causal `r` | causal AUROC | best static attacker | shuffled AUROC | reversed AUROC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5x | 1.000000 | 0.218750 | 0.712558 | 0.877460 | gradient 0.857143 | 0.882540 | 0.430476 |
| **1.0x** | **1.000000** | **0.458333** | **0.769216** | **0.849213** | **gradient 0.698427** | **0.858392** | **0.402972** |
| 1.5x | 1.000000 | 0.531250 | 0.804859 | 0.866667 | gradient 0.659695 | 0.877560 | 0.413072 |

The useful directional signal survives finite-strength changes. At `1.0x` and `1.5x` it substantially outperforms the best static attacker. At `0.5x`, gradient alignment nearly catches it, so the `+0.10` static margin would not hold there.

At every tested strength, however, shuffled transport remains slightly better than ordered transport. That makes the temporal-order failure robust rather than a threshold accident.

## What T6 changes in the project

T6 does **not** validate the originally strongest version of the causal collision-radar hypothesis.

It does establish a more useful separation:

```text
static parameter similarity          weak here
static Fisher / local Jacobian       weak here
correct damage direction             important
retained-task dynamic susceptibility important
fine ordering of intervening tangent transport
                                     not shown necessary
```

That distinction matters for the next engineering step. A practical continual-learning guard can still use the measured directional risk score, but it must be presented as an empirical consolidation heuristic whose advantage is currently tied to **which retained computation is susceptible**, not to a proven need for exact causal time ordering.

## Demo boundary

The next implementation step demonstrates the network learning continually with the same candidate generator and the same directional risk score:

```text
propose useful candidate
-> estimate risk to retained skills
-> accept or reject/defer
-> compare with accepting every useful candidate
```

That demo is valuable even though T6 is a frozen scientific FAIL. It tests whether the surviving signal can improve an actual sequence of consolidation decisions.

A successful demo must not be described as retroactively passing T6.
