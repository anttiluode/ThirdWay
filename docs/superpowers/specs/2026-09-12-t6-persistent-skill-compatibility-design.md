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

> **In this shared nonlinear recurrent model, propagated causal compatibility predicts persistent cross-skill damage from low-rank candidate edits better than static parameter, gradient, Fisher-style, and local Jacobian overlap signals.**

T6 must not claim general superiority over continual-learning methods, biological plausibility, or spontaneous discovery of computational highways.

## Core design constraint

Every candidate edit used in the collision battery must be **individually useful** for its target skill before compatibility is evaluated.

This avoids a trivial failure mode in which the radar merely detects bad edits. The question is specifically whether two locally successful changes can coexist.

For an old retained edit `R` and a new candidate `C`, the gate therefore separates two questions:

1. **utility:** does `C` improve its target skill when applied alone?
2. **compatibility:** after `C` is permanently added to the current shared model, how much does it damage retained skills or switching behavior?

Only candidates passing the utility threshold enter the compatibility comparison.

## Model

Use one small recurrent tanh network with shared recurrent weights:

```math
h_{t+1}=\tanh(Wh_t + Bx_t + b),
```

with one fixed readout vector or a small shared readout matrix.

The recurrent matrix `W` is the object that receives persistent low-rank edits.

The model should stay small enough that all attackers and oracle measurements can be computed directly in NumPy. No autograd is required for the candidate mechanism. If finite-difference gradients are used as an attacker, label them as such.

Recommended first implementation size:

```text
hidden dimension: 24–40
sequence length: 12–24
skills: 3
candidate edit rank: 1 or 2
```

The exact values are implementation parameters, not claims.

## Skills

T6 needs multiple tasks that use the same recurrent substrate but require different temporal computations. They should not differ only by output label.

Recommended three-skill family:

### Skill A — recent cue

Output depends on a recent input feature.

### Skill B — delayed cue

Output depends on a feature seen several steps earlier.

### Skill C — relational / order cue

Output depends on the relation or temporal order between two events rather than either event alone.

The tasks share input statistics and output scale so that compatibility cannot be solved by separating them through obvious input magnitude or output channels.

The model is evaluated under explicit task context. T6 is not testing task inference.

## Seed model

T6 should start from a base recurrent system that has nontrivial but imperfect performance on all three skills.

Two acceptable construction paths are:

1. **closed-form / random-feature seed:** hold `W` fixed initially and fit only a linear readout so all tasks are above chance but leave room for recurrent improvement;
2. **small deterministic search seed:** select one random `W` from a fixed seed bank whose shared readout reaches a required baseline range.

Do not spend T6 building a large trained network. The purpose is persistent edit compatibility, not representation learning.

The chosen seed procedure must be deterministic and reproduced in tests.

## Candidate persistent edits

For each skill, generate a small family of rank-1 or rank-2 recurrent edits:

```math
\Delta W = U M V^\top.
```

A candidate may be obtained by finite local probes around the current model:

```text
propose low-rank direction
-> apply temporarily
-> measure target-skill improvement
-> scale / line-search over a small fixed amplitude set
-> keep only individually useful candidates
```

This is intentionally simple. T6 does not need to solve how candidate edits are generated optimally.

The candidate-generation procedure must not inspect cross-skill compatibility scores. Otherwise the compatibility predictor would be baked into the proposals.

## Persistent consolidation experiment

The battery operates on an ordered pair:

```text
retained edit R
candidate edit C
```

The current model is

```math
W_R = W_0 + R.
```

The candidate model is

```math
W_{R+C}=W_0 + R + C.
```

Both `R` and `C` must be individually useful on their own target skills from the same base model.

For each pair, measure the finite persistent damage caused by adding `C` after `R`.

## Outcome measures

T6 uses three damage measures. A pair may collide through any of them.

### 1. Retained-skill damage

```math
D_{old} = \max_{s\in S_R}
\left[\operatorname{loss}_s(W_{R+C})-\operatorname{loss}_s(W_R)\right].
```

This is the primary target.

### 2. Computational leakage

Measure how much the hidden trajectory for one skill becomes more similar to another skill's trajectory after consolidation.

Use a simple, reproducible diagnostic such as normalized cross-skill hidden-state Gram overlap or principal-subspace overlap.

This is diagnostic, not the primary collision label.

### 3. Switching cost

Run short alternating blocks of tasks A/B/C and measure the transient error immediately after a context switch.

This detects edits that preserve isolated task accuracy but damage fast routing/use of the shared substrate.

The primary binary collision label is derived from retained-skill damage with a fixed threshold chosen before test scoring. Leakage and switching cost are reported continuously.

## Pre-commit predictors

Every predictor must be computed without committing both edits permanently.

### Attacker 1 — parameter cosine

```math
s_{param}=|\cos(R,C)|.
```

For low-rank edits, flatten the matrices.

### Attacker 2 — gradient cosine

Estimate each skill's local loss gradient with respect to `W` at `W_R` using deterministic finite differences or analytic forward sensitivities if convenient.

Score the alignment between the candidate edit and retained-skill gradients, and/or gradient-gradient cosine between skill objectives.

The exact chosen scalar must be documented before the held-out battery is evaluated.

### Attacker 3 — true Fisher-style overlap

Unlike T5's activation proxy, T6 should build an actual empirical Fisher-style matrix or diagonal approximation from per-example output sensitivities under the retained model.

For tractability, use either:

```text
diagonal empirical Fisher
```

or a small low-rank empirical Fisher estimated from a fixed probe batch.

Report exactly which approximation is used. Do not call a cosine proxy "Fisher".

### Attacker 4 — static Jacobian/subspace overlap

Measure the local recurrent Jacobian change caused by each persistent edit on its own skill trajectories, but compare those changes without transporting them through the later retained dynamics.

This is the natural static analogue of the T5 attacker.

### Candidate mechanism — propagated causal compatibility

Measure the isolated state-response field induced by candidate `C` on its own target trajectories.

Propagate that perturbation through the current retained dynamics `W_R` to the times / states at which retained skills are sensitive.

Then contract it with a retained-skill susceptibility operator.

A generic score is

```math
s_{causal}(C\to R)
=
\sum_{t}\left\|G_{R,t}\,\widehat{\delta h}_{C,t}\right\|,
```

where

```math
\widehat{\delta h}_{C,t+1}=J_t(W_R)\widehat{\delta h}_{C,t}
```

and `G_{R,t}` measures how changes in the hidden state at time `t` affect retained-skill loss or output.

The score is directional. In general,

```math
s_{causal}(C\to R) \ne s_{causal}(R\to C).
```

T6 should preserve that direction rather than symmetrizing by default.

### Oracle — joint finite evaluation

Temporarily apply both edits and directly evaluate all relevant tasks.

This is not a practical pre-commit predictor; it is the expensive upper-bound attacker and source of the true collision label.

## Train / held-out split

Thresholds and scalar-combination choices must be frozen before final scoring.

Use a split that prevents trivial memorization of particular edit sites or task pairs.

Recommended split:

```text
train: half of candidate-generation seeds / low-rank source directions
held-out: disjoint remaining seeds / directions
```

Do not randomly split pair rows if the same underlying edit appears in both train and test.

The unit of holdout is the **candidate edit identity**, not an individual pair.

## Required battery structure

The final held-out set must contain all three qualitative categories:

1. **compatible pairs** — both edits remain useful together;
2. **destructive pairs** — candidate improves its own skill but harms a retained skill;
3. **static confounds** — high static overlap but little actual damage, or low static overlap with large actual damage.

A useful gate requires nontrivial counts in all three categories. If the generator produces only one category, that configuration is invalid and should not be reported as a success.

## Success criteria

T6 passes only if all of the following hold on held-out edit identities:

1. every evaluated candidate passes the individual-utility requirement before compatibility scoring;
2. at least 20% of held-out pairs collide and at least 20% remain compatible;
3. the propagated causal score has positive continuous correlation with finite retained-skill damage, target `r > 0.70`;
4. its held-out AUROC or balanced accuracy exceeds each static attacker by at least `0.10` absolute;
5. the result is not carried by a single task pair; causal advantage must be positive on at least two of the three retained/new skill pair families;
6. oracle finite joint evaluation remains best or tied-best, as expected;
7. all earlier T0–T5 tests continue to pass.

These thresholds are design targets. If the first honest implementation misses them, record the negative result rather than tuning until it passes.

## Important negative controls

### Shuffle causal transport

Randomly permute the time order or Jacobian sequence used by the causal radar while preserving its marginal magnitudes.

Prediction: performance should collapse if ordered propagation is the source of the signal.

### Reverse direction

Score `R -> C` when the task asks for `C -> R`.

Prediction: asymmetric cases should expose the error. If direction never matters, the claimed causal interpretation is too strong.

### Zero-horizon control

Replace propagated susceptibility with a same-time local susceptibility.

This asks whether simple local sensitivity already explains the result.

### Candidate-strength sweep

Evaluate at least three finite edit amplitudes.

Prediction: first-order causal scores should degrade gracefully as edits become stronger. A catastrophic breakdown is scientifically useful and should be reported.

## Implementation boundaries

T6 remains pure NumPy unless a dependency is scientifically necessary.

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
2. GREEN: deterministic base world and summary object exist;
3. RED: useful-candidate invariant required;
4. GREEN: candidate generator produces individually useful persistent edits;
5. RED: battery must contain compatible and destructive held-out pairs;
6. GREEN: construct the balanced pair battery without using predictor labels;
7. RED: causal predictor must beat static attackers on held-out edit identities;
8. GREEN: implement propagated score and attackers;
9. RED: shuffled/reversed causal controls must lose predictive value;
10. GREEN: finish controls and reporting;
11. regression: T0–T6 plus CI on Python 3.11 and 3.12.

Tests should assert scientific invariants and broad performance margins, not exact floating-point table values where avoidable.

## Failure interpretation

T6 has several useful negative outcomes.

### Static attackers win

Then T4/T5 were special transport constructions. Stop escalating the causal-compatibility claim and keep them as narrow counterexamples.

### Causal score works only at tiny edit amplitudes

Then it is a local screening tool, not a robust consolidation rule. Record the radius of usefulness.

### Causal score predicts isolated old-skill damage but not switching cost

Then compatibility of stored skills and compatibility of routing are separate problems. Do not merge them conceptually.

### No individually useful edits coexist

Then the candidate generator or shared model is too constrained. Change the world/model before drawing conclusions about the radar.

### Almost every edit coexists

Then the problem is too easy. Increase shared-substrate pressure rather than inventing stronger metrics.

## Handoff to later gates

T6 does not yet combine every ThirdWay mechanism.

If T6 succeeds, the next useful fusion is not immediately "discover all highways." The cleaner sequence is:

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
