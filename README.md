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

### Practical demo: use the surviving signal as a consolidation guard

The frozen 12-event schedule is

```text
A -> B -> C -> B -> A -> C -> A -> B -> C -> B -> A -> C
```

Both learners receive the **same candidate matrices**. Accept-all commits every locally useful proposal. The guarded learner commits only if the proposal remains useful in its current state and predicted dynamic risk to retained other skills stays below the threshold learned from the frozen T6 training split.

Starting accuracy:

```text
A=.859 B=.664 C=.633    mean=.719
```

After 12 persistent proposals:

| learner | A | B | C | mean | A/B old-at-final mean | switch penalty |
|---|---:|---:|---:|---:|---:|---:|
| **guarded** | **0.859** | 0.625 | 0.664 | **0.716** | **0.742** | **0.01256** |
| accept-all | 0.664 | **0.641** | **0.672** | 0.659 | 0.652 | 0.02105 |

The guard accepts `5/12` updates and rejects `7/12`. Six are rejected for predicted dynamic risk; one proposal becomes locally useless after the two learning trajectories have diverged.

This is a real protective effect, not a solved continual learner. The guard preserves A and still improves C, but B regresses and the guarded model's aggregate mean does not exceed its starting mean.

That exposes the next problem:

> **compatibility rejection protects the repertoire, but wastes useful updates.**

[`T6_DEMO_RESULTS.md`](T6_DEMO_RESULTS.md)

Run the live event trace:

```bash
python demo_t6_continual_learning.py
```

---

## What moved forward from the older repos

[`RETROSPECTIVE.md`](RETROSPECTIVE.md) contains the detailed map. The compact form is:

```text
SighImageSuper   -> memory is a recoverable distinction
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

## Next hard boundary — T7 compatible partial writes

T6's reject-only guard is useful but wasteful. T7 should ask a more constructive question:

> A full candidate is locally useful but unsafe. How much of it can become permanent without destroying retained computation?

Freeze a descending scale bank such as

```text
1.00, 0.75, 0.50, 0.25, 0.125
```

and choose the largest fraction that remains target-useful while falling below the retained-skill risk threshold.

If every scaled version remains unsafe, add the IttnasNoruen lesson: measure a small compensation direction that bends the finite update back into the acceptable region.

The same 12-event proposal stream should compare:

```text
accept-all
reject-only guard       <- T6 demo
safe-step guard         <- T7
finite oracle upper bound
```

The target is no longer merely **less forgetting**. It is:

> **keep learning while respecting compatibility constraints.**

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
```

The raw `python -m experiments.gate_t6_persistent_skill_compatibility` command intentionally exits nonzero because the frozen scientific T6 criterion remains a recorded negative result. CI verifies that exact negative result separately rather than hiding it.
