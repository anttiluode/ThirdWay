# T7B — signed residual-carrying compatibility memory

## Frozen result: FAIL

T7B asked whether the failure of T7A came from resetting compatibility after every write. It kept the same seed model, datasets, 12 target-only candidate matrices, schedule, scale bank and candidate directions, but carried a signed memory of collateral hidden-trajectory drift for each protected skill.

The answer is **mixed but negative under the predeclared gate**.

Signed route memory is less conservative and substantially better than an unsigned cumulative-debt attacker under the same route-space budget, but it does not improve aggregate held-out accuracy beyond the starting network, does not preserve the frozen switching bound, and produces **zero actual residual-repair events**.

The allowed conclusion is therefore:

> **A signed memory of collateral route displacement is more useful than accumulating unsigned displacement magnitude in this frozen sequence, but T7B does not show that later learning repairs earlier structural debt or that bounded route displacement is sufficient for sequence-level continual learning.**

---

## Frozen setup

T7B reuses the T7A world without changing:

```text
seed model                   23
proposal seed                101
calibration seed             103
evaluation seed              107
examples per skill           128
residual-signature examples  48
schedule                     A B C B A C A B C B A C
candidate seeds              32..43
scale bank                   1, .75, .5, .25, .125
minimum target improvement   1e-4
```

Every committed T7B edit is exactly

```math
\alpha\,\Delta W
```

for the original candidate matrix and one frozen scale `alpha`. No direction is rotated, regenerated or compensated.

---

## What the residual remembers

For protected skill `k`, T7B stores the exact recurrent hidden-trajectory signature on the first 48 calibration examples,

```math
H_k(W)=\operatorname{vec}(h_{1:T}^{(1)},\ldots,h_{1:T}^{(48)}).
```

Relative to the latest intentionally accepted version of that skill, the signed collateral residual is

```math
R_k(W)=H_k(W)-A_k.
```

For a candidate step `W -> W'`,

```math
r_k=H_k(W')-H_k(W),
```

so the carried response-space update is exactly

```math
R_k' = R_k + r_k.
```

This telescopes as a finite identity in measured response space; T7B does not claim the nonlinear RNN itself has become linear.

When skill `k` is intentionally updated as the target, its anchor moves to that accepted computation and its own residual resets to zero. Only collateral drift remains as debt.

---

## Untuned common budget

The route-space budget is derived before evaluating either cumulative controller by replaying the already-frozen T6 reject-only policy on the same 12 proposals.

Peak residual RMS by skill:

```text
A  0.025222037651
B  0.000000000000
C  0.030291886681
```

The common budget is therefore

```text
B = 0.030291886681226846
```

Both the unsigned scalar-debt attacker and signed residual controller use exactly this same `B`.

---

## Policies

### Scalar cumulative debt

For every protected skill,

```math
d_k \leftarrow d_k + \|r_k\|_{RMS}.
```

It remembers magnitude only. Its debt cannot decrease through cancellation.

Selected scales:

```text
0.50, 0.25, 0.75, 0,
0.125, 0, 0, 0.50,
0.25, 0.25, 0.50, 0.25
```

Nonzero writes: **9/12**.

### Signed residual memory

For every protected skill,

```math
R_k \leftarrow R_k + r_k,
```

and a candidate is legal only if

```math
\|R_k+r_k\|_{RMS}\le B.
```

Selected scales:

```text
0.50, 0.75, 0.50, 0.125,
0.75, 0.50, 0, 0.50,
0.50, 0.125, 0.75, 0.25
```

Nonzero writes: **11/12**.

---

## Final behavior

| learner | A | B | C | mean | old A/B mean | switch penalty | writes |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0.859375 | 0.664062 | 0.632812 | **0.718750** | 0.761719 | - | 0 |
| reject-only T6 | **0.843750** | 0.656250 | 0.648438 | 0.716146 | **0.750000** | **0.009663** | 1 |
| T7A per-write safe-step | 0.742188 | 0.656250 | **0.679688** | 0.692708 | 0.699219 | 0.01784 | 12 |
| scalar cumulative debt | 0.789062 | 0.648438 | 0.664062 | 0.700521 | 0.718750 | 0.018865 | 9 |
| **signed residual memory** | 0.789062 | **0.687500** | **0.679688** | **0.718750** | **0.738281** | 0.018258 | **11** |

Signed residual memory clearly improves over the matched scalar-debt attacker:

```text
writes        11 vs 9
mean          0.718750 vs 0.700521
old A/B       0.738281 vs 0.718750
switch        0.018258 vs 0.018865
```

It also stays within the predeclared old-skill safety floor relative to reject-only:

```text
required old A/B >= 0.750000 - 0.020000 = 0.730000
observed             0.738281
```

So the signed geometry carries useful information that the scalar magnitude budget discards.

---

## But the frozen gate fails

### 1. Switching remains too disrupted

Required:

```text
signed switch <= reject-only switch + 0.005
              <= 0.014662649370
```

Observed:

```text
0.018257609576
```

**FAIL.**

A bounded hidden-trajectory displacement does not, by itself, protect the carried-state switching behavior measured by the T6/T7 demos.

### 2. Aggregate learning does not improve beyond the reference learners

Required:

```text
signed mean >= reject-only mean + 0.01
            >= 0.726145833333

and

signed mean >= base mean + 0.005
            >= 0.723750000000
```

Observed:

```text
signed mean = 0.718750000000
```

The result lands **exactly at the starting aggregate mean**. Skill B and C improve relative to base, but Skill A loses enough to cancel the gain.

**FAIL.**

### 3. No actual repair event occurs

A repair event was frozen as an accepted update for which a protected skill's carried residual norm actually decreases:

```math
\|R_k+r_k\| < \|R_k\|-10^{-8}.
```

Observed repair events:

```text
0
```

**FAIL.**

The signed controller gets an advantage over scalar debt because vector addition is less pessimistic than adding norms: directions can be partly opposed or orthogonal so the resultant grows more slowly than the scalar sum. But in this proposal stream, no accepted later update actually pulls a protected route closer to its anchor than it was before that update.

That distinction matters. T7B shows **geometric slack**, not demonstrated structural repair.

---

## Final carried states

Signed residual RMS:

```text
A  0.007289295687
B  0.029362095636
C  0.000000000000
```

Unsigned scalar debt:

```text
A  0.007323743422
B  0.026816079635
C  0.000000000000
```

Every accepted signed write remained within the common budget and every committed matrix stayed on its original candidate direction.

---

## Frozen criteria

| criterion | result |
|---|---|
| identical 12 candidate identities | PASS |
| no direction rotation | PASS |
| at least 3 signed writes | PASS — 11 |
| old A/B within 0.02 of reject-only | PASS — 0.738281 |
| switch within reject-only + 0.005 | **FAIL** |
| mean >= reject-only + 0.01 | **FAIL** |
| mean >= base + 0.005 | **FAIL** |
| at least 2 more writes than scalar debt | PASS — 11 vs 9 |
| old A/B no worse than scalar debt - 0.01 | PASS |
| mean >= scalar debt + 0.005 | PASS |
| at least 2 genuine repair events | **FAIL — 0** |
| every accepted residual remains within budget | PASS |

Overall:

```text
T7B = FAIL
```

---

## What this says about the original intuition

The useful part of the intuition survives, but in a narrower form.

The search process does not need to explicitly remember its own history. Its accepted edits leave a physical consequence in the shared recurrent substrate. A linear signed response-space state can summarize part of that consequence:

```text
search / proposal history
        -> persistent operator edits
        -> altered route trajectories
        -> signed residual state
```

In that precise sense, the linear tracker can remember the **wake left by earlier search**.

But this run does not support the stronger story that later candidate writes naturally repair that wake. The proposal stream contains no accepted event that actually decreases an existing protected residual norm, and keeping route displacement bounded is not enough to keep switching behavior safe.

The next mechanism, if pursued, must therefore test something stronger than passive residual bookkeeping: either a compensation direction explicitly chosen against the endangered response, a compatibility state that includes switching/carry-state coordinates, or a search process that can deliberately seek residual-cancelling proposals.
