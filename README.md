# ThirdWay

**Three sols. Two unique views. This is fusion.**

ThirdWay is the separate fusion arm between:

- [GeometricNeuronV25GeneticAlgoAndFanningNeuron](https://github.com/anttiluode/GeometricNeuronV25GeneticAlgoAndFanningNeuron): separated causal routes, delayed scalar consequence, adaptive eligibility timescales and resonant temporal modes.
- [GAx](https://github.com/anttiluode/GAx): positive/evolutionary operators, modal purification, separated computational approaches, context-dependent gain and covariant computational subspaces.

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

Two seeded, counterfactually different computations share one router. Reward is a delayed scalar with no `(context, route)` label.

| learner | late route accuracy | late reward | task accuracy |
|---|---:|---:|---:|
| **route-local eligibility** | **0.952750** | **0.955109** | **0.977875** |
| pooled eligibility | 0.512125 | 0.517902 | 0.759125 |
| current-only | 0.512125 | 0.517902 | 0.759125 |
| shuffled route identity | 0.043000 | 0.049467 | 0.524750 |

> **The reward need not carry the address if the material has kept it locally.**

[`T0_RESULTS.md`](T0_RESULTS.md)

---

## T1 — the road moves, identity continues

Two computational modes rotate through coordinates. A fixed-coordinate identity fails; a continuation tracker follows the moving road.

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

Reward is delayed 3–10 events while the computational basis rotates by `0.32 rad/event`.

| credit rule | late route accuracy | late reward | task accuracy |
|---|---:|---:|---:|
| **covariantly transported** | **0.952750** | **0.955109** | **0.977875** |
| stale coordinates | 0.184500 | 0.196868 | 0.598500 |

```math
\boxed{\textbf{credit identity must be covariant with computational identity}.}
```

> **stale credit** — the memory survives, but now points at yesterday's computation.

[`T2_RESULTS.md`](T2_RESULTS.md)

---

## T3 — credit cannot be finer than identifiable computation

When two moving spectral axes become nearly degenerate, ordinary vector tracking becomes unstable. An ambiguity-block tracker follows the coarser identifiable subspace instead.

| tracker | late route accuracy | semantic fidelity |
|---|---:|---:|
| vector overlap | 0.5115 | 0.4565 |
| **ambiguity blocks** | **0.9525** | **0.9967** |
| oracle | 0.9525 | 1.0000 |

Then the hidden axes rotate by 90 degrees *inside an exactly degenerate subspace*. The primary operator is mathematically blind to that rotation. A second diagnostic operator is allowed only while the primary operator cannot identify the individual routes.

| tracker | late route accuracy | semantic fidelity | extra query fraction |
|---|---:|---:|---:|
| ambiguity blocks only | 0.3884 | 0.0210 | 0 |
| **active operator family** | **0.9453** | **0.9992** | **0.2002** |
| oracle | 0.9525 | 1.0000 | 0 |

```math
\boxed{\textbf{resolution of credit} \le \textbf{resolution of identifiable computation}.}
```

If the required write is finer than the current evidence, **ask another question before writing**.

[`T3_RESULTS.md`](T3_RESULTS.md)

---

## T4 — compatibility of learning is causal, not geometric

Kompressori found low-rank experience-induced operator changes and interference between overlapping changes. CausalHorizon supplied the exact time-ordered interaction coordinate

```math
\Omega_{ji}=V_j^*\Phi(\tau_j,\tau_i+1)U_i.
```

Across 192 balanced exact linear cases:

| collision predictor | accuracy | precision | recall |
|---|---:|---:|---:|
| **propagated causal overlap** `|Omega_ji|` | **1.000** | **1.000** | **1.000** |
| static overlap `|V_j^T U_i|` | 0.333 | 0.000 | 0.000 |

Same raw coordinate can be harmless after transport; different raw coordinates can collide exactly.

> **Compatibility is about where an edit's consequence arrives, not merely where its parameters live.**

[`T4_RESULTS.md`](T4_RESULTS.md)

---

## T5 — nonlinear collision radar

T4's `Omega` result was exact and linear. T5 moves the same question into a driven `tanh` recurrent system with finite rank-1 edits.

The target is now the measured finite nonlinear change in the **marginal action of the later edit**. The causal predictor does not run the joint pair. It measures one edit in isolation, transports that perturbation through the unedited recurrent Jacobians, then asks how susceptible the later edit is to the arriving perturbation.

```math
\widehat{\delta x}_{j}
=J_{j-1}\cdots J_{i+1}r_i,
\qquad
s_{\rm causal}=\|G_j\widehat{\delta x}_{j}\|.
```

There are 192 cases. Predictor thresholds are fit on source sites 0–7 and frozen before evaluation on held-out sites 8–15.

Correlation with finite nonlinear collision strength:

| predictor | Pearson r |
|---|---:|
| parameter cosine | -0.411329 |
| activation / Fisher-ish proxy | -0.411329 |
| static Jacobian-change cosine | -0.410830 |
| **propagated causal susceptibility** | **0.994736** |

Held-out collision classification:

| predictor | accuracy | precision | recall |
|---|---:|---:|---:|
| parameter cosine | 0.604167 | 0.000000 | 0.000000 |
| activation / Fisher-ish proxy | 0.604167 | 0.000000 | 0.000000 |
| static Jacobian-change cosine | 0.604167 | 0.000000 | 0.000000 |
| **propagated causal susceptibility** | **1.000000** | **1.000000** | **1.000000** |

So the T4 idea survives a first nonlinear attack:

> **A single-edit, first-order causal forecast can predict a finite nonlinear interaction on unseen sites even when raw/static similarity points the wrong way.**

This is still a constructed transport RNN, not yet continual learning in a trained network. The activation attacker is only Fisher-ish, not a Fisher matrix.

[`T5_RESULTS.md`](T5_RESULTS.md)

---

## What moved forward from the older repos

[`RETROSPECTIVE.md`](RETROSPECTIVE.md) contains the detailed map. The compact form is:

```text
SighImageSuper   -> memory is a recoverable distinction
MovingProblem    -> identity must survive moving coordinates
AlgoSchalgo      -> ambiguity sets the legal resolution of identity/credit
Child + V24      -> buy another observation when a proposed write needs finer evidence
IttnasNoruen     -> preserve counterfactual computational access, not only old outputs
Kompressori      -> experience-induced operator changes can be low-rank and collide
CausalHorizon    -> propagated causal overlap is the right exact linear interaction coordinate
Operaattori      -> persistent structure can compile the slow operator being rewritten
V25              -> local eligibility can itself acquire adaptive temporal modes
GAx              -> maintain several approaches; context changes their modal gain
```

The genetic-algorithm picture has therefore changed from

```text
variation -> selection -> retention
```

to

```text
maintain alternative computational approaches
-> context amplifies some
-> consequence returns to identifiable continuations
-> propose local operator changes
-> test causal compatibility with retained changes
-> consolidate without destroying the repertoire
```

Call the extra stage **compatibility selection**.

---

## Next hard boundary

T5 still uses time-local edit events in a constructed recurrent transport system.

The next serious experiment is **persistent skill edits**:

1. put several task-specific computational highways in one shared recurrent model;
2. derive finite low-rank candidate weight changes from actual task episodes;
3. commit each candidate persistently rather than for one event;
4. measure old-skill damage, cross-mode leakage and switching cost;
5. compare parameter/gradient cosine, a real Fisher-style attacker, static Jacobian overlap, propagated causal radar and expensive finite joint evaluation.

If propagated causal geometry still wins there, ThirdWay stops being mainly an explanatory fusion and starts becoming a practical continual-learning mechanism.

Only after that should V25's adaptive resonant credit be folded back into the same changing substrate, and only later should the seeded computational approaches be removed.

The strong phrase remains a target:

> **self-rewriting spectral router over discovered computational highways**

---

## Run

```bash
python -m experiments.gate_t0_fusion
python -m experiments.gate_t1_moving_road
python -m experiments.gate_t2_credit_on_moving_road
python -m experiments.gate_t3_identifiability_credit
python -m experiments.gate_t4_causal_collision
python -m experiments.gate_t5_nonlinear_collision_radar
pytest -q
```
