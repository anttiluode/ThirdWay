# Gate T4 — learning compatibility is causal, not geometric

T4 is the first ThirdWay gate about **whether two individually useful pieces of learning can coexist**.

The motivation comes directly from two older repos:

- **Kompressori**: experience-induced operator changes can be low-rank, and two low-rank changes can interfere when their response geometries overlap.
- **CausalHorizon**: for time-ordered finite-rank edits, the exact directed interaction coordinate is

```math
\Omega_{ji}=V_j^*\Phi(\tau_j,\tau_i+1)U_i.
```

T4 asks whether this propagated overlap is a better compatibility coordinate than static parameter/subspace overlap.

## Exact transport laboratory

The base dynamics is a 32-state cyclic shift. Two local rank-1 edits occur at different times:

```math
A_{\tau_i}\mapsto A+0.2\,e_a e_a^T,
```

```math
A_{\tau_j}\mapsto A+0.2\,e_b e_b^T.
```

Between the edits the substrate transports state around the ring.

For every first-event site and several event gaps, T4 constructs three deliberately balanced cases:

1. **transported collision** — the second edit is spatially different from the first, but lies exactly where the first edit is propagated;
2. **static confound** — the two edits use the same raw coordinate, but transport moves the first edit away before the second occurs;
3. **neutral** — neither static nor propagated overlap.

There are 192 cases total, 64 of each type.

## Result

A collision is defined by exact finite non-additivity of the two edits:

```math
\Delta\Phi_{ij}\ne\Delta\Phi_i+\Delta\Phi_j.
```

For this orthogonal rank-1 construction,

```math
\|\Delta\Phi_{ij}-\Delta\Phi_i-\Delta\Phi_j\|_F
=|0.2\cdot0.2\cdot\Omega_{ji}|.
```

The direct matrix products match that prediction to a worst error below `7e-18`.

| predictor | collision accuracy | precision | recall |
|---|---:|---:|---:|
| **propagated causal overlap** `|Omega_ji|` | **1.000** | **1.000** | **1.000** |
| static overlap `|V_j^T U_i|` | **0.333** | **0.000** | **0.000** |

The static rule fails in both possible directions:

- **100% of transported collisions** have zero static overlap and are therefore missed;
- **100% of same-coordinate static-overlap pairs** are actually harmless in this timing geometry and would be rejected unnecessarily.

The smallest concrete counterexample is almost absurdly simple:

```text
first edit at site 0
8 transport steps
second edit at site 8

static overlap       = 0
causal overlap       = 1
finite cross term    = 0.04
```

while

```text
first edit at site 0
8 transport steps
second edit at site 0

static overlap       = 1
causal overlap       = 0
finite cross term    = 0
```

So "nearby parameters" and even "same parameter direction" are not the right compatibility question once learning changes an operator embedded in dynamics.

## What this changes

The old continual-learning picture often asks whether two updates are orthogonal in parameter or gradient space.

ThirdWay now has a different candidate question:

> **Does the consequence of edit `i`, after being propagated by the current substrate, enter the write/read coordinates of edit `j`?**

That is directional and time ordered.

```text
edit i
   ↓
current operator propagates its effect
   ↓
does that propagated effect reach edit j's receptive/write subspace?
   ↓
yes -> collision risk
no  -> independent write may be safe
```

This is the first mechanism in the repo that addresses the compatibility of **learning events**, not merely the identity or credit of computational routes.

## Connection back to the genetic/evolutionary picture

The GA-inspired picture now becomes more specific.

A system can maintain several computational approaches and locally generate candidate changes. But selection should not retain every individually successful mutation independently.

Before two successful changes are consolidated together, the substrate can ask whether they are causally orthogonal:

```math
\Omega_{ji}\approx0
```

or whether one change will alter the dynamical coordinates in which the other change acts.

That gives a possible route from

```text
variation -> local success -> retain
```

to

```text
variation
   -> local success
   -> compatibility with retained causal highways
   -> retain separately / coordinate / reject
```

The additional compatibility stage is what the older genetic-algorithm metaphor was missing.

## Boundary

T4 is intentionally exact and linear.

It does **not** yet prove that `Omega_ji` estimated from a nonlinear neural system predicts real learning interference better than gradients, Fisher overlap, Jacobian overlap, or finite candidate evaluation.

The ring transport is chosen precisely because it makes the distinction between static geometry and propagated causal geometry unambiguous.

The next serious gate should make `Omega` estimated/noisy and put it against those boring attackers on an adaptive nonlinear model.
