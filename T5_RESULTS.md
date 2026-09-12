# Gate T5 — nonlinear collision radar

T4 established the exact linear statement that static parameter overlap can be the wrong compatibility coordinate. T5 asks whether any of that survives when the substrate is **state-dependent and nonlinear** and the causal score is only a first-order estimate.

The answer in this deliberately small recurrent laboratory is yes.

## Nonlinear recurrent substrate

The state evolves as

```math
x_{t+1}=\tanh(Ax_t+b_t),
```

with a 32-state directed transport operator plus weak self-retention,

```math
A=0.85P+0.08I,
```

and a deterministic nonzero drive `b_t`. The system therefore does not live at the zero-state linearization.

Two finite rank-1 edits occur at ordered times,

```math
E_i=0.8\,e_a e_a^T,
\qquad
E_j=0.8\,e_b e_b^T.
```

Each candidate edit has a measurable isolated state effect; the smallest isolated effect anywhere in the 192-case battery is `0.001208` in L2 norm.

## What counts as a collision

The collision target is not raw output non-additivity after an arbitrary long future. It is more local and operational:

> **How much does the first edit change the marginal action of the second edit when the second edit is actually applied?**

Let

```math
m_j(x)=f_{A+E_j}(x)-f_A(x).
```

If `x_j` is the unedited state at the second event and `x_j^{(i)}` is the state after edit `i`, the measured finite nonlinear collision is

```math
\boxed{
C_{i\to j}=\|m_j(x_j^{(i)})-m_j(x_j)\|_2.
}
```

A pair is classified as a material collision when

```math
C_{i\to j}>0.005.
```

The threshold is fixed before the train/test split is scored.

## Four pre-commit predictors

T5 compares four quantities available without running the *joint* `i+j` intervention.

### 1. Parameter cosine

```math
s_{\rm param}=\frac{|\langle E_i,E_j\rangle_F|}{\|E_i\|_F\|E_j\|_F}.
```

### 2. Fisher-ish activation proxy

This is deliberately named a **proxy**, not an actual Fisher matrix. It asks whether the isolated state changes produced by the two edits point in the same raw state direction.

```math
s_{\rm act}=|\cos(\delta x_i,\delta x_j)|.
```

### 3. Static local Jacobian-change cosine

Each edit changes the one-step recurrent Jacobian at its own event time. T5 compares those local Jacobian changes directly, without transporting either one through the intervening dynamics.

### 4. Propagated causal susceptibility

First measure the isolated state injection caused by edit `i`,

```math
r_i=x_{\tau_i+1}^{(i)}-x_{\tau_i+1}.
```

Then propagate that perturbation through the **unedited** recurrent tangent,

```math
\widehat{\delta x}_{\tau_j}
=J_{\tau_j-1}\cdots J_{\tau_i+1}r_i.
```

Finally differentiate the *marginal action* of edit `j`,

```math
G_j=\frac{\partial m_j}{\partial x}\bigg|_{x_{\tau_j}},
```

and score

```math
\boxed{
s_{\rm causal}=\|G_j\widehat{\delta x}_{\tau_j}\|_2.
}
```

This is the nonlinear descendant of the T4/CausalHorizon idea. It is no longer an exact finite-event identity. It is a first-order forecast of a finite nonlinear interaction.

## Hard cases

For four event gaps and 16 first-edit sites, T5 builds three pair types:

1. **transported** — the second edit is placed where the first edit's effect is carried by the recurrent dynamics;
2. **static confound** — both edits use the same raw parameter coordinate even though the first effect has moved away by the time the second acts;
3. **neutral** — neither construction is deliberately aligned.

That gives

```text
4 gaps x 16 sources x 3 pair types = 192 cases.
```

The first eight source sites are used only to fit each predictor's scalar decision threshold. Sources 8–15 are held out completely for the final collision classification.

```text
train cases = 96
held-out cases = 96
```

## Result

Continuous correlation with the measured finite nonlinear collision strength:

| predictor | Pearson r |
|---|---:|
| parameter cosine | **-0.411329** |
| activation / Fisher-ish proxy | **-0.411329** |
| static Jacobian-change cosine | **-0.410830** |
| **propagated causal susceptibility** | **0.994736** |

The negative correlations are useful: in this adversarial geometry, the most obvious static similarity is systematically pointing toward the harmless same-coordinate confound.

Thresholds are learned on the first half of source locations and then frozen.

Held-out collision classification:

| predictor | accuracy | precision | recall |
|---|---:|---:|---:|
| parameter cosine | 0.604167 | 0.000000 | 0.000000 |
| activation / Fisher-ish proxy | 0.604167 | 0.000000 | 0.000000 |
| static Jacobian-change cosine | 0.604167 | 0.000000 | 0.000000 |
| **propagated causal susceptibility** | **1.000000** | **1.000000** | **1.000000** |

The causal threshold selected on the training half is `0.004661`.

The three static predictors choose the conservative training solution: call no pair dangerous. That is rational under the misleading raw geometry, but it misses every held-out collision.

## What T5 adds

T4 said:

```text
static overlap != causal overlap
```

for an exact linear transport system.

T5 says something stronger but still narrow:

```text
single-edit measurement
    +
first-order transport through the current nonlinear trajectory
    +
local susceptibility of the later edit
        ↓
can predict finite nonlinear edit interaction
without running the joint edit pair.
```

The important object is not merely where two updates live. It is

> **where the first update's consequence will be when the second update tries to act, and how sensitive the second update is to that arriving consequence.**

That gives the GA-inspired retention loop a more specific compatibility stage:

```text
variation
  -> local consequence
  -> candidate low-rank edit
  -> propagate its effect through the current operator
  -> estimate collision with retained / pending edits
  -> retain independently, coordinate, defer, or reject
```

## Boundary

T5 is still a constructed recurrent transport laboratory.

It does **not** yet establish:

- persistent edits to a trained neural network;
- spontaneously discovered computational highways;
- superiority to true Fisher information, EWC, gradient projection, GEM/OGD, or full finite candidate evaluation;
- that the first-order radar remains accurate under strong chaotic amplification or long horizons;
- that a biological system computes these Jacobians explicitly.

The activation cosine is only a Fisher-ish attacker, not a Fisher matrix.

The next serious gate should move from time-local edit events to **persistent task-specific low-rank learning changes in a shared recurrent model**, then ask whether a measured/estimated causal radar predicts old-skill damage and loss of switching ability better than standard static compatibility signals.
