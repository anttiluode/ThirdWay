# ThirdWay

**Three sols. Two unique views. This is fusion.**

ThirdWay is the separate fusion arm between:

- [GeometricNeuronV25GeneticAlgoAndFanningNeuron](https://github.com/anttiluode/GeometricNeuronV25GeneticAlgoAndFanningNeuron): separated causal routes, delayed scalar consequence, adaptive eligibility timescales and resonant temporal modes.
- [GAx](https://github.com/anttiluode/GAx): evolutionary operators, modal purification, separated computational approaches, context-dependent gain and covariant computational subspaces.

The rule is strict: **ThirdWay inherits questions, not conclusions.** Every fusion claim gets its own executable gate and attackers.

The current object is:

> **a self-rewriting collection of covariant computational highways.**

A highway has at least five separable properties:

```text
computation       — what counterfactual operation it performs
identity          — how that operation continues while coordinates move
routing           — when context amplifies it
credit            — how delayed consequence finds it again
compatibility     — whether rewriting it collides with other retained routes
```

See [`THEORY.md`](THEORY.md) for the formal object and [`RETROSPECTIVE.md`](RETROSPECTIVE.md) for the handoff from the older repos.

---

## T0 — the reward has no address, the substrate does

Delayed reward is only a scalar. Route-local eligibility keeps the missing causal address in the substrate.

| learner | late route accuracy | late reward | task accuracy |
|---|---:|---:|---:|
| **route-local eligibility** | **0.952750** | **0.955109** | **0.977875** |
| pooled/current-only | 0.512125 | 0.517902 | 0.759125 |
| shuffled route identity | 0.043000 | 0.049467 | 0.524750 |

> **The reward need not carry the address if the material has kept it locally.**

[`T0_RESULTS.md`](T0_RESULTS.md)

---

## T1 — the road moves, identity continues

Two computational modes rotate through coordinates. Fixed-coordinate identity fails; continuation follows the moving computation.

| observer | identity accuracy | contextual route reward |
|---|---:|---:|
| fixed initial coordinates | 0.519271 | 0.038542 |
| **continuation tracker** | **1.000000** | **1.000000** |

```math
\boxed{L_tE_k(t)=E_k(t+1)}
```

> **Identity is continuation, not location.**

[`T1_RESULTS.md`](T1_RESULTS.md)

---

## T2 — credit must move with the road

Reward is delayed while the computational basis moves. A persistent trace in stale coordinates becomes wrong credit.

| credit rule | late route accuracy | late reward | task accuracy |
|---|---:|---:|---:|
| **covariantly transported** | **0.952750** | **0.955109** | **0.977875** |
| stale coordinates | 0.184500 | 0.196868 | 0.598500 |

```math
\boxed{\textbf{credit identity must be covariant with computational identity}.}
```

[`T2_RESULTS.md`](T2_RESULTS.md)

---

## T3 — credit cannot be finer than identifiable computation

Near degeneracy makes individual spectral axes unstable. Ambiguity blocks keep the coarser identifiable subspace; a second operator is queried only when the primary observation is mathematically unable to identify the finer routes.

| tracker | late route accuracy | semantic fidelity |
|---|---:|---:|
| vector overlap | 0.5115 | 0.4565 |
| **ambiguity blocks** | **0.9525** | **0.9967** |
| oracle | 0.9525 | 1.0000 |

In the exact hidden-gauge attack:

| tracker | late route accuracy | semantic fidelity | extra query fraction |
|---|---:|---:|---:|
| ambiguity blocks only | 0.3884 | 0.0210 | 0 |
| **active operator family** | **0.9453** | **0.9992** | **0.2002** |
| oracle | 0.9525 | 1.0000 | 0 |

```math
\boxed{\textbf{resolution of credit} \le \textbf{resolution of identifiable computation}.}
```

[`T3_RESULTS.md`](T3_RESULTS.md)

---

## T4 — compatibility of learning is causal, not geometric

Kompressori's low-rank experience-induced operator changes meet CausalHorizon's exact propagated interaction coordinate

```math
\Omega_{ji}=V_j^*\Phi(\tau_j,\tau_i+1)U_i.
```

Across 192 exact linear cases:

| collision predictor | accuracy | precision | recall |
|---|---:|---:|---:|
| **propagated causal overlap** | **1.000** | **1.000** | **1.000** |
| static overlap | 0.333 | 0.000 | 0.000 |

> **Compatibility is about where an edit's consequence arrives, not merely where its parameters live.**

[`T4_RESULTS.md`](T4_RESULTS.md)

---

## T5 — nonlinear collision radar

T5 moves the T4 question into a driven `tanh` recurrent system with finite rank-1 edits. A single-edit first-order causal forecast predicts the finite nonlinear interaction on held-out sites.

| predictor | correlation with finite collision |
|---|---:|
| parameter cosine | -0.411329 |
| activation / Fisher-ish proxy | -0.411329 |
| static Jacobian-change cosine | -0.410830 |
| **propagated causal susceptibility** | **0.994736** |

Held-out collision classification:

| predictor | accuracy | precision | recall |
|---|---:|---:|---:|
| three static attackers | 0.604167 | 0.000000 | 0.000000 |
| **propagated causal susceptibility** | **1.000000** | **1.000000** | **1.000000** |

This is still a constructed transport RNN, not yet persistent continual learning.

[`T5_RESULTS.md`](T5_RESULTS.md)

---

## T6 — persistent learning in one shared recurrent network

T6 is the first gate here with permanent task-learning edits in one shared 24-state nonlinear recurrent substrate.

Candidate rank-1 edits are generated using only their own target skill. The held-out battery uses disjoint edit identities and measures actual old-skill damage after both edits become permanent.

### Scientific gate: mixed result, frozen FAIL

| predictor | held-out AUROC |
|---|---:|
| parameter cosine | 0.551136 |
| empirical Fisher | 0.579983 |
| static Jacobian overlap | 0.542832 |
| gradient alignment | 0.698427 |
| **directional dynamic susceptibility** | **0.849213** |
| shuffled tangent order | **0.858392** |
| reversed direction | 0.402972 |
| finite oracle | 1.000000 |

The directional score has `r=0.769216` with finite old-skill damage, beats every tested static attacker by the frozen `+0.10` margin at the primary strength, and wins in `4/6` ordered skill-pair families.

But the strongest hypothesis fails: shuffling detailed tangent time order does not hurt prediction. Ordered and shuffled scores correlate `0.990072` because a **persistent** edit injects a perturbation again at every recurrent step; the last four injections contribute about `66.3%` of the first-order signal.

So T6 supports the narrower claim:

> **Direction-specific dynamic susceptibility of a retained computation predicts persistent cross-skill damage better than the tested static similarity measures.**

It does **not** show that fine tangent time ordering is necessary for that advantage.

[`T6_RESULTS.md`](T6_RESULTS.md)

### Practical demo: dynamic susceptibility as a safety filter

The frozen 12-event schedule is

```text
A -> B -> C -> B -> A -> C -> A -> B -> C -> B -> A -> C
```

Both learners receive the **same candidate matrices**. Accept-all commits every locally useful proposal. The corrected guard protects every non-target skill already present in the baseline RNN from the first event onward; it commits only if the proposal remains useful in its current state and predicted risk to every protected skill stays below the frozen T6 threshold.

Starting accuracy:

```text
A=.859 B=.664 C=.633    mean=.719
```

After 12 persistent proposals:

| learner | A | B | C | mean | minimum | A/B old-at-final mean | switch penalty |
|---|---:|---:|---:|---:|---:|---:|---:|
| **guarded** | **0.844** | **0.656** | 0.648 | **0.716** | **0.648** | **0.750** | **0.00966** |
| accept-all | 0.664 | 0.641 | **0.672** | 0.659 | 0.641 | 0.652 | 0.02105 |

The corrected guard accepts only `1/12` updates and rejects `11/12`: nine for predicted dynamic risk and two because the proposal is no longer locally useful after the learning trajectories diverge.

This is a real safety effect, but **not yet a useful continual learner**. It substantially reduces forgetting and switching disruption, yet its final mean `0.716` remains slightly below the starting mean `0.719` because the binary veto refuses almost everything.

That exposes the next problem directly:

> **protective stagnation: compatibility rejection protects the repertoire by refusing useful learning.**

[`T6_DEMO_RESULTS.md`](T6_DEMO_RESULTS.md)

Run the live event trace:

```bash
python demo_t6_continual_learning.py
```

---

## T7A — compatible partial writes

T7A kept the T6 world, candidate stream, risk threshold and candidate directions fixed. It changed only one thing: instead of binary accept/reject, each useful direction was tried at the frozen amplitudes

```text
1.00, 0.75, 0.50, 0.25, 0.125
```

and the largest target-useful fraction below every protected-skill risk threshold was committed.

### Frozen result: FAIL

T7A completely escaped the T6 freeze: it found a nonzero fraction for **12/12** proposals. But per-write safety did not compose across the sequence.

| learner | A | B | C | mean | A/B old-at-final mean | switch penalty | writes |
|---|---:|---:|---:|---:|---:|---:|---:|
| accept-all | 0.664 | 0.641 | 0.672 | 0.659 | 0.652 | 0.02105 | 12 |
| reject-only T6 | **0.844** | **0.656** | 0.648 | **0.716** | **0.750** | **0.00966** | 1 |
| T7A safe-step | 0.742 | **0.656** | **0.680** | 0.693 | 0.699 | 0.01784 | 12 |
| finite per-step control | 0.812 | 0.641 | 0.664 | 0.706 | 0.727 | 0.01931 | 12 |

The frozen T7A safety floor required old A/B `>= 0.730`; safe-step reached only `0.699`. Its switch penalty also exceeded the allowed T6 margin. The learning criterion failed too: final mean `0.693` did not beat reject-only `0.716` or the starting mean `0.719`.

A post-failure diagnostic showed why:

```text
positive finite harm steps                 12 / 12
sum of positive per-step max harm          0.380687
corr(predicted risk, finite step harm)     0.644540
final calibration loss drift A             +0.061845
final calibration loss drift B             +0.077985
final calibration loss drift C             -0.022412
```

Every accepted partial write was below the frozen *predicted* risk ceiling, yet every one caused a positive finite loss increment to at least one protected skill. Small residual harm accumulated.

The finite per-step control is important here. It directly checks actual finite damage before each one step, yet it also accepts all twelve and ends below the starting aggregate mean. It is therefore an information upper bound for a **single local decision**, not a globally optimizing sequence oracle.

The new executable principle is:

> **Per-write compatibility does not compose into sequence-level compatibility.**

Or in the residual language suggested by the SighImageSuper connection:

> **Shrinking a useful operator change can reduce its harmful residual, but does not remove it; repeated residuals can accumulate into forgetting.**

[`T7A_RESULTS.md`](T7A_RESULTS.md)

Run:

```bash
python -m experiments.t7_compatible_partial_writes
python -m experiments.t7a_accumulation_diagnostic
```

---

## T7B — signed residual-carrying route memory

T7B asks whether T7A failed because compatibility was reset after every accepted write. It keeps the exact same 12 proposal matrices and fixed scale bank, but carries the collateral change in each protected skill's recurrent hidden trajectory as a signed vector.

For protected skill `k`,

```math
R_k=H_k(W)-A_k,
```

where `H_k` is the flattened hidden trajectory on 48 fixed calibration examples and `A_k` is the latest intentionally accepted version of that skill. A new write contributes

```math
r_k=H_k(W')-H_k(W),
```

so in measured response space

```math
R_k' = R_k+r_k.
```

The common RMS residual budget is **not tuned on T7B**. It is the largest collateral route displacement produced by the already-frozen T6 reject-only learner:

```text
B = 0.030291886681
```

A matched attacker accumulates only unsigned step magnitudes. Both policies receive the same candidates and the same budget.

### Frozen result: FAIL, with a real signed-memory advantage

| learner | A | B | C | mean | A/B old mean | switch penalty | writes |
|---|---:|---:|---:|---:|---:|---:|---:|
| reject-only T6 | **0.844** | 0.656 | 0.648 | 0.716 | **0.750** | **0.00966** | 1 |
| scalar cumulative debt | 0.789 | 0.648 | 0.664 | 0.701 | 0.719 | 0.01887 | 9 |
| **signed residual memory** | 0.789 | **0.688** | **0.680** | **0.719** | **0.738** | 0.01826 | **11** |

Signed memory is materially less conservative than unsigned debt: it accepts `11/12` rather than `9/12`, raises mean accuracy from `0.70052` to `0.71875`, and raises old A/B from `0.71875` to `0.73828`, while every accepted route residual remains inside the identical budget.

But the stronger hypothesis fails in three ways:

```text
switching target     FAIL   0.01826 > 0.01466 allowed
learning target      FAIL   final mean 0.71875 == starting mean
repair events        FAIL   0 actual residual-norm decreases
```

That last line matters. Vector addition gives **geometric slack** relative to adding unsigned norms, but this frozen proposal stream does not contain an accepted write that actually pulls a protected route closer to its anchor.

So the supported statement is narrower than the original intuition:

> **The shared substrate physically retains the wake of earlier accepted search, and a signed linear response-space memory uses that wake better than unsigned cumulative debt. T7B does not show that later learning naturally repairs the wake.**

[`T7B_RESULTS.md`](T7B_RESULTS.md)

Run:

```bash
python -m experiments.t7b_residual_carrying_memory
```

---

## What moved forward from the older repos

[`RETROSPECTIVE.md`](RETROSPECTIVE.md) contains the detailed map. The compact form is:

```text
SighImageSuper   -> persistence hierarchy; a surviving mode leaves a transient remainder to interrogate
MovingProblem    -> identity must survive moving coordinates
AlgoSchalgo      -> ambiguity sets the legal resolution of identity/credit
Child + V24      -> buy another observation when a proposed write needs finer evidence
IttnasNoruen     -> finite useful changes may need nonlinear compensation
Kompressori      -> experience-induced operator changes can be low-rank and collide
CausalHorizon    -> propagated causal overlap is the exact linear interaction coordinate
Operaattori      -> persistent structure can compile the slow operator being rewritten
V25              -> local eligibility can itself acquire adaptive temporal modes
GAx              -> maintain several approaches; context changes their modal gain
```

The genetic-algorithm picture has changed from

```text
variation -> selection -> retention
```

to

```text
maintain alternative computations
-> context favors some
-> consequence returns to identifiable continuations
-> propose persistent operator change
-> test compatibility with retained computation
-> consolidate, shrink, compensate or defer
```

Call the extra stage **compatibility selection**.

---

## Next hard boundary — repair, not just bookkeeping

T6 showed that a binary guard can protect knowledge by nearly stopping learning. T7A showed that per-write safe amplitudes can still accumulate damage. T7B showed that remembering the **signed** route wake is better than remembering only unsigned damage magnitude, but passive bookkeeping produced no actual repair event and did not protect switching.

The next architectural question is therefore no longer merely

```text
What residual has accumulated?
```

It is

```text
Can the learner deliberately choose a useful change that removes part of the
measured harmful residual — including the carried-state/switching coordinates
that hidden-trajectory distance alone failed to protect?
```

Two stronger mechanisms are now motivated but not established:

```text
1. measured compensation
   -> keep the useful component of Delta W while adding a component chosen
      specifically to cancel the endangered retained response

2. residual-seeking search
   -> let proposal generation prefer useful candidates whose signed route wake
      reduces existing collateral residual instead of waiting for accidental cancellation
```

Only if one of those produces genuine repair should residual saturation be considered a signal for new-highway growth.

After that, return to active sensing for write resolution, V25 adaptive resonant credit, and finally discovered rather than seeded computational approaches.

The strong phrase remains a target, not a result:

> **self-rewriting spectral router over discovered computational highways**

---

## Run

```bash
pytest -q
python -m experiments.gate_t0_fusion
python -m experiments.gate_t1_moving_road
python -m experiments.gate_t2_credit_on_moving_road
python -m experiments.gate_t3_identifiability_credit
python -m experiments.gate_t4_causal_collision
python -m experiments.gate_t5_nonlinear_collision_radar
python demo_t6_continual_learning.py
python -m experiments.t7_compatible_partial_writes
python -m experiments.t7a_accumulation_diagnostic
python -m experiments.t7b_residual_carrying_memory
```

The raw `python -m experiments.gate_t6_persistent_skill_compatibility` command intentionally exits nonzero because the frozen scientific T6 criterion remains a recorded negative result. CI verifies that exact negative result separately rather than hiding it. T7A and T7B are likewise kept as explicit frozen negative results rather than relaxed into passes.
