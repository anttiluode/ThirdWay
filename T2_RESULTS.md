# Gate T2 — Delayed credit on a moving road

## Question

What happens when a computational highway moves through parameter coordinates **while the consequence of an earlier action is still in flight**?

T0 established that a delayed scalar consequence can learn a context-dependent router if the substrate keeps route-local eligibility. T1 established that route identity can continue while coordinates rotate. T2 fuses those two claims literally.

## Construction

The same two computational identities from T0 are used. Their current physical basis rotates every event by

```text
dtheta = 0.32 rad/event
```

while scalar reward arrives after a random delay of `3..10` events.

A local policy-score event is stored as a vector in the **current physical basis**:

```math
e_c^{\rm phys} \leftarrow e_c^{\rm phys} + E(t)s_c.
```

The later scalar reward contains no route label.

At update time, modal credit is recovered by projection onto the current basis:

```math
\hat e_c(t)=E(t)^\top e_c^{\rm phys}(t).
```

Two systems differ only in what happens to the local eligibility vector while the road moves.

### Covariant credit

Transport the eligibility with the road:

```math
e^{\rm phys}(t+1)=\rho L_t e^{\rm phys}(t),
```

where

```math
L_t=E(t+1)E(t)^\top.
```

### Stale-coordinate credit

Decay the eligibility but leave it in the old coordinates:

```math
e^{\rm phys}(t+1)=\rho e^{\rm phys}(t).
```

The delayed reward then projects yesterday's physical direction onto today's modes.

An ideal modal trace is maintained only as a diagnostic shadow; it is never used by the learner.

## Fixed 8-seed battery

Each seed runs 5000 events; late metrics use the final 1000 events.

| credit rule | late route accuracy | late scalar reward | task accuracy | worst-seed route accuracy |
|---|---:|---:|---:|---:|
| **covariantly transported** | **0.952750** | **0.955109** | **0.977875** | **0.932000** |
| stale physical coordinates | **0.184500** | **0.196868** | **0.598500** | **0.054000** |

For the transported system, the maximum disagreement between projected physical eligibility and the ideal modal shadow over all seeds and all events is

```text
1.110e-15
```

The stale trace reaches modal-credit mismatches above `3`, because the old coordinate vector is increasingly projected onto the wrong current highway.

## Interpretation

This is the first literal fusion result in ThirdWay:

```text
T0: the reward does not need the route address if local material keeps it.
T1: the route address is a continuation, not a fixed coordinate.
T2: therefore the local causal address itself must move with the continuation.
```

Or compactly,

```math
\boxed{\text{credit identity must be covariant with computational identity}.}
```

A trace can be perfectly persistent in time and still become wrong if the computation has moved away from the coordinates in which the trace was stored.

This introduces a distinct failure mode from ordinary forgetting:

> **stale credit** — the memory survives, but it points at yesterday's computation.

That is especially relevant to a self-rewriting system. If an expert/mode changes its internal geometry while delayed feedback is outstanding, a fixed-coordinate credit mechanism can reinforce a different mode than the one that caused the consequence.

## Limits

- The moving basis is constructed and smooth.
- The transport operator is known exactly in this gate.
- The computational modes are still seeded.
- There is no splitting/merger of highways.
- Eligibility is still a simple decaying trace rather than an adaptive resonant pole.

The next hard version should **infer the transport from observations** instead of receiving `L_t`, then test what happens near ambiguous crossings, mode birth/death, or branching.
