# T6 Design — Persistent Skill Compatibility

Date: 2026-09-12
Status: proposed implementation design

## Purpose

T5 showed that a first-order propagated causal susceptibility can predict finite interaction between two time-local low-rank edits in a driven nonlinear recurrent system, while several static similarity signals fail.

T6 asks the next harder question:

> **Can the same causal idea predict whether persistent, individually useful learning edits can coexist inside one shared recurrent model?**

This gate changes the object from transient events to continual learning. Candidate edits are no longer injected for one time step and discarded. They are proposed as permanent low-rank changes to a shared model. The system must decide whether to consolidate a candidate without destroying previously retained skills or the ability to switch among them.

The target is not a new optimizer. The target is a **pre-commit compatibility decision rule**.

## Scientific claim T6 is allowed to earn

If the gate succeeds, the narrow claim is:

> **In this shared nonlinear recurrent model, propagated causal compatibility predicts persistent cross-skill damage from low-rank candidate edits better than static parameter, gradient, empirical-Fisher, and local Jacobian overlap signals.**

T6 must not claim general superiority over continual-learning methods, biological plausibility, or spontaneous discovery of computational highways.

## Core design constraint

Every candidate edit used in the collision battery must be **individually useful** for its target skill before compatibility is evaluated.

This avoids a trivial failure mode in which the radar merely detects bad edits. The question is specifically whether two locally successful changes can coexist.

For an old retained edit `R` and a new candidate `C`, the gate separates two questions:

1. **utility:** does `C` improve its target skill when applied alone?
2. **compatibility:** after `C` is permanently added to the current shared model, how much does it damage retained skills or switching behavior?

Only candidates passing the utility threshold enter the compatibility comparison.

## Model

Use one small recurrent tanh network with shared recurrent weights:

```math
h_{t+1}=\tanh(Wh_t + Bx_t + Cc + b),
```

where `c` is an explicit one-hot task context. T6 is not testing task inference.

A single shared linear readout produces the binary answer:

```math
\hat y = r^\top h_T.
```

The recurrent matrix `W` is the only object that receives persistent low-rank edits.

First implementation constants are fixed by the spec:

```text
hidden dimension: 24
sequence length: 16
skills: 3
candidate edit rank: 1
```

The gate remains pure NumPy.

## Skills

All three tasks use the same input channels, binary target distribution and output scale. Only the temporal computation differs.

### Skill A — recent cue

At a marked late position, one signed cue determines the target.

### Skill B — delayed cue

The target is determined by a signed cue at a marked early position, with nuisance events after it.

### Skill C — temporal relation

Two marked signed events occur; the target is their ordered relation, e.g. whether their signed product matches the presented order bit. Neither single event is sufficient.

Task markers and context are balanced. Nuisance inputs share the same distribution across skills so that simple input magnitude does not reveal the task.

## Deterministic seed model

Use one fixed RNG seed to construct `W0`, `B`, `C` and `b`.

`W0` is scaled to a stable spectral radius below 1.0. `B`, `C` and `b` remain fixed for all of T6.

Generate a fixed training batch for the three skills, run it through `W0`, and fit the **single shared readout `r` by ridge regression** on the concatenated terminal hidden states. No recurrent weights are trained at this stage.

The resulting base model must satisfy all of:

```text
accuracy on each skill > 0.60
accuracy on each skill < 0.95
```

If the fixed initial seed misses this range, the implementation may deterministically scan a predeclared seed list `0..31` and choose the first seed satisfying it. This selection uses only base-model skill accuracy and is frozen before candidate edits are generated.

## Candidate persistent edits

For each skill and each candidate seed, sample a deterministic rank-1 direction

```math
D = uv^\top,
```

with unit-norm `u` and `v`.

Test the fixed signed amplitude grid

```text
{-0.30, -0.20, -0.10, +0.10, +0.20, +0.30}
```

and choose the amplitude that most improves the target skill on a candidate-generation batch.

A candidate is admitted only if it reduces target-skill loss by at least **5% relative** to `W0` on that batch and also improves target-skill accuracy by at least **0.02 absolute**.

Candidate generation is forbidden from reading any other skill's loss, any compatibility predictor, or any pairwise score.

Generate candidates until each skill has at least 12 admitted edits or a fixed maximum of 256 proposed directions is exhausted. Failure to obtain 12 admitted edits for a skill is a failed world configuration, not a radar result.

## Persistent consolidation experiment

The battery operates on an ordered pair:

```text
retained edit R
candidate edit C
```

with different target skills.

The retained model is

```math
W_R = W_0 + R,
```

and the consolidated model is

```math
W_{R+C}=W_0+R+C.
```

Both `R` and `C` must have passed the individual-utility rule from the same base model.

Each ordered pair is evaluated on a separate fixed evaluation batch never used for candidate generation.

## Primary outcome — retained-skill damage

For retained skill `s_R`, define raw damage

```math
\Delta L_{old}
=
L_{s_R}(W_{R+C})-L_{s_R}(W_R).
```

Define scale-normalized damage

```math
D_{old}
=
\frac{\Delta L_{old}}{\max(L_{s_R}(W_R),0.05)}.
```

The primary continuous target is `D_old`.

A **material collision** is fixed by the spec as

```math
\boxed{D_{old}>0.20}.
```

This threshold is never fit to the held-out battery.

## Secondary outcomes

### Computational leakage

For each skill, concatenate hidden trajectories across a fixed probe batch and form a normalized hidden-state Gram matrix. Report how much the retained and new skill Gram matrices become more similar after consolidation.

This is diagnostic only; it does not define the collision label.

### Switching cost

Run a fixed stream of short task blocks in the order

```text
A -> B -> C -> A -> C -> B
```

using explicit context. Report the change in error on the first two examples after each context switch. This detects edits that preserve isolated skill accuracy but damage rapid reuse of the shared substrate.

Again, this is diagnostic rather than the primary collision label.

## Pre-commit predictors

Every predictor is computed at `W_R` without permanently committing `C` and without evaluating the joint model `W_{R+C}`.

### Attacker 1 — parameter cosine

```math
s_{param}=|\cos(R,C)|.
```

Flatten the edit matrices.

### Attacker 2 — gradient conflict

Compute the full gradient of retained-skill loss with respect to `W_R` using explicit NumPy forward sensitivities through the recurrence. This attacker may use exact chain-rule information; it is deliberately allowed to be strong.

Score the candidate against the retained-skill gradient as

```math
s_{grad}
=
\frac{|\langle \nabla_W L_{s_R},C\rangle|}
{\|\nabla_W L_{s_R}\|_F\,\|C\|_F}.
```

The candidate mechanism does not use this gradient.

### Attacker 3 — diagonal empirical Fisher importance

On a fixed retained-skill probe batch, compute per-example gradients `g_q` of the binary negative-log-likelihood with respect to `W_R` and form

```math
F_{diag}=\frac1Q\sum_q g_q\odot g_q.
```

Score candidate risk as Fisher-weighted edit energy:

```math
s_{Fisher}(C\to R)
=
\frac{\sum F_{diag}\odot C^2}{\|C\|_F^2}.
```

This is explicitly a **diagonal empirical Fisher**, not a cosine proxy and not the full Fisher information matrix.

### Attacker 4 — static Jacobian-change overlap

For `R` and `C` separately, measure the change each edit produces in the one-step recurrent Jacobian over its own skill probe trajectories. Concatenate these Jacobian-change tensors over time and examples and compute their absolute cosine without temporal transport:

```math
s_{Jstatic}=|\cos(\Delta J_R,\Delta J_C)|.
```

### Candidate mechanism — propagated causal compatibility

First measure the hidden-state response field induced by temporarily applying `C` on its own target-skill trajectories around `W_R`.

Propagate each resulting perturbation through the **retained model's unedited tangent dynamics**:

```math
\widehat{\delta h}_{t+1}
=J_t(W_R)\widehat{\delta h}_t.
```

On retained-skill probe trajectories, compute retained-loss susceptibility to hidden-state perturbation,

```math
G_{R,t}=\frac{\partial L_{s_R}}{\partial h_t}.
```

The causal compatibility score is the mean propagated contraction over probe examples and time:

```math
\boxed{
s_{causal}(C\to R)
=
\operatorname{mean}_{q,t}
\left|G_{R,t}^{(q)}\widehat{\delta h}_{C,t}^{(q)}\right|.
}
```

The score is directional. T6 never symmetrizes it by default.

### Oracle — joint finite evaluation

Temporarily apply both edits and directly evaluate `D_old`.

This is the expensive upper bound and the source of the ground-truth collision label. It is not included among practical pre-commit predictors when ranking usefulness.

## Train / held-out split

The unit of holdout is the **candidate edit identity**, not an individual pair row.

Within each skill, admitted edits are sorted by their deterministic candidate seed. Even-indexed edit identities form the training pool; odd-indexed identities form the held-out pool.

Final evaluation pairs are built only from held-out retained edits and held-out candidate edits. No held-out edit identity appears in predictor-threshold fitting.

Predictor thresholds for balanced-accuracy reporting are fit only on training-pool ordered pairs.

Continuous AUROC is evaluated directly on held-out pairs and is the primary comparison metric because it does not depend on threshold tuning.

## Required battery structure

The held-out battery must contain all three qualitative categories:

1. **compatible pairs** — both edits remain useful together;
2. **destructive pairs** — `C` is individually useful but causes `D_old > 0.20`;
3. **static confounds** — at least five held-out cases in which parameter cosine and collision disagree qualitatively: high static overlap with no collision or low static overlap with collision.

A valid gate requires:

```text
held-out collision prevalence between 0.20 and 0.80
at least 5 static confounds
at least 4 of the 6 ordered skill pairs represented by both collision classes
```

If this structure is absent, the world is rejected as uninformative before predictor performance is interpreted.

## Success criteria

T6 passes only if all of the following hold on held-out edit identities:

1. every edit in the battery passed the individual-utility requirement;
2. the required battery structure above is satisfied;
3. causal score Pearson correlation with continuous `D_old` is `r > 0.70`;
4. causal held-out AUROC exceeds **each** static attacker (`parameter`, `gradient`, `Fisher`, `Jstatic`) by at least `0.10` absolute;
5. the train-fit causal threshold yields held-out balanced accuracy above `0.70`;
6. causal AUROC exceeds the best static attacker in at least **4 of the 6 ordered skill-pair families** where both collision classes are present;
7. shuffled-time and reversed-direction causal controls each lose at least `0.10` AUROC relative to the correct causal score;
8. all earlier T0–T5 tests continue to pass on Python 3.11 and 3.12.

The oracle is expected to be perfect because it directly measures the target; it is reported only as an upper-bound sanity check.

These are frozen design targets. If the first honest implementation misses them, record the negative result rather than tuning the thresholds until it passes.

## Negative controls

### Shuffle causal transport

Randomly permute the order of the intervening recurrent Jacobians while keeping the same matrices and marginal magnitudes.

Prediction: AUROC should fall if ordered propagation is carrying real information.

### Reverse direction

Use the causal score `R -> C` when predicting damage `C -> R`.

Prediction: asymmetric pairs should expose the mistake.

### Zero-horizon control

Replace propagated susceptibility with the same-time contraction before tangent transport.

This asks whether local sensitivity alone explains the result.

### Candidate-strength sweep

After the primary gate is frozen, rerun evaluation at candidate amplitude multipliers

```text
0.5x, 1.0x, 1.5x
```

without regenerating directions.

This is reported as a robustness curve, not used to decide the primary pass. The expected pattern is graceful degradation of first-order prediction as edits become stronger.

## Implementation boundaries

T6 remains pure NumPy.

Expected files:

```text
experiments/gate_t6_persistent_skill_compatibility.py
tests/test_gate_t6.py
T6_RESULTS.md
README.md update
THEORY.md update
.github/workflows/ci.yml update
```

Keep model/world helpers inside the T6 module unless reuse becomes real. Do not create a framework for hypothetical future gates.

## TDD sequence

Implementation follows the repository's existing gate style and the Superpowers TDD rule.

Recommended red-green order:

1. RED: module / `run()` contract absent;
2. GREEN: deterministic world and summary object exist;
3. RED: base model must satisfy the fixed 0.60–0.95 per-skill accuracy range;
4. GREEN: deterministic seed scan and shared ridge readout;
5. RED: useful-candidate invariant required;
6. GREEN: target-only candidate generator admits at least 12 edits per skill;
7. RED: held-out battery must contain compatible, destructive and static-confound pairs;
8. GREEN: build ordered pairs on disjoint edit-identity pools;
9. RED: define and verify gradient, empirical-Fisher and static-Jacobian attackers;
10. GREEN: implement attackers without causal transport;
11. RED: causal predictor must beat static attackers on held-out edit identities;
12. GREEN: implement propagated causal score;
13. RED: shuffled-time and reversed-direction controls must lose predictive value;
14. GREEN: finish controls and reporting;
15. regression: T0–T6 plus CI on Python 3.11 and 3.12.

Tests should assert scientific invariants and broad performance margins, not exact floating-point table values where avoidable.

## Failure interpretation

T6 has several useful negative outcomes.

### Static attackers win

Then T4/T5 were special transport constructions. Stop escalating the causal-compatibility claim and keep them as narrow counterexamples.

### Causal score works only at tiny edit amplitudes

Then it is a local screening tool, not a robust consolidation rule. Record the useful radius.

### Causal score predicts isolated old-skill damage but not switching cost

Then compatibility of stored skills and compatibility of routing are separate problems. Do not merge them conceptually.

### No individually useful edits coexist

Then the candidate generator or shared model is too constrained. Change the world/model before drawing conclusions about the radar.

### Almost every edit coexists

Then the problem is too easy. Increase shared-substrate pressure rather than inventing stronger metrics.

## Handoff to later gates

T6 does not yet combine every ThirdWay mechanism.

If T6 succeeds, the cleaner sequence is:

```text
T6 persistent compatibility
    -> T7 active measurement for write resolution
    -> T8 resonant/covariant credit during persistent rewrite
    -> later: discovered computational approaches
```

The eventual object would then have four independent safeguards:

```text
what computation is this?            identity / identifiability
which one should act now?            routing
which past action deserves credit?   covariant delayed credit
can this useful rewrite coexist?     causal compatibility
```

That is the smallest coherent route toward the stronger target:

> **a self-rewriting spectral router over discovered computational highways.**
