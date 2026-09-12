# ThirdWay

**Three sols. Two unique views. This is fusion.**

ThirdWay is the separate fusion arm between:

- [GeometricNeuronV25GeneticAlgoAndFanningNeuron](https://github.com/anttiluode/GeometricNeuronV25GeneticAlgoAndFanningNeuron): separated causal routes, delayed scalar consequence, adaptive eligibility timescales and resonant temporal modes.
- [GAx](https://github.com/anttiluode/GAx): positive/evolutionary operators, modal purification, separated computational approaches, context-dependent gain and covariant computational subspaces.

The rule is strict: **ThirdWay inherits questions, not conclusions.** Every fusion claim gets its own executable gate and attackers.

The current object is:

> **a self-rewriting collection of covariant computational highways.**

A highway is not just a weight vector. It has at least five separable properties:

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

Result:

> **The reward need not carry the address if the material has kept it locally.**

[`T0_RESULTS.md`](T0_RESULTS.md)

---

## T1 — the road moves, identity continues

Two computational modes rotate through coordinates. A fixed-coordinate identity fails; a continuation tracker follows the moving road.

| observer | identity accuracy | contextual route reward |
|---|---:|---:|
| fixed initial coordinates | 0.519271 | 0.038542 |
| **continuation tracker** | **1.000000** | **1.000000** |

The exact cocycle residual is below `3e-16`.

Result:

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

The transported physical eligibility matches an ideal modal shadow to about `1.1e-15`.

Result:

```math
\boxed{\textbf{credit identity must be covariant with computational identity}.}
```

This names a distinct failure:

> **stale credit** — the memory survives, but now points at yesterday's computation.

[`T2_RESULTS.md`](T2_RESULTS.md)

---

## T3 — credit cannot be finer than identifiable computation

T2 knew the continuation transport. T3 removes that privilege.

### Noisy crossing

When two moving spectral axes become nearly degenerate, ordinary vector tracking becomes unstable. An ambiguity-block tracker instead tracks the coarser identifiable subspace and carries its previous semantic basis through the block.

| tracker | late route accuracy | semantic fidelity |
|---|---:|---:|
| vector overlap | 0.5115 | 0.4565 |
| **ambiguity blocks** | **0.9525** | **0.9967** |
| oracle | 0.9525 | 1.0000 |

### Exact hidden-gauge attacker

Then the hidden axes rotate by 90 degrees *inside an exactly degenerate subspace*. The primary operator is mathematically blind to that rotation.

A second diagnostic operator is allowed only while the primary operator cannot identify the individual routes.

| tracker | late route accuracy | semantic fidelity | extra query fraction |
|---|---:|---:|---:|
| ambiguity blocks only | 0.3884 | 0.0210 | 0 |
| **active operator family** | **0.9453** | **0.9992** | **0.2002** |
| oracle | 0.9525 | 1.0000 | 0 |

Result:

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

T4 creates a deliberately hostile transport example. A first low-rank edit is moved by the substrate before a second edit occurs.

Across 192 balanced cases:

| collision predictor | accuracy | precision | recall |
|---|---:|---:|---:|
| **propagated causal overlap** `|Omega_ji|` | **1.000** | **1.000** | **1.000** |
| static overlap `|V_j^T U_i|` | 0.333 | 0.000 | 0.000 |

The static rule fails both ways:

```text
same raw coordinate
    -> static overlap = 1
    -> causal overlap = 0
    -> finite cross term = 0
```

while

```text
different raw coordinate
    -> static overlap = 0
    -> causal overlap = 1
    -> finite cross term = 0.04
```

The exact cross-term prediction matches direct matrix products to below `7e-18`.

Result:

> **Two pieces of learning are compatible according to how their effects propagate through the current operator, not merely according to where their parameters sit.**

[`T4_RESULTS.md`](T4_RESULTS.md)

---

## What moved forward from the older repos

[`RETROSPECTIVE.md`](RETROSPECTIVE.md) contains the full map. The compact version is:

```text
SighImageSuper
    -> memory = recoverable distinction, not merely persistent state

MovingProblem
    -> moving coordinates are not the problem; unjustified identity is

AlgoSchalgo
    -> ambiguity blocks become the legal granularity of credit

Child
    -> delayed audit + active sensing + provenance + representation drift become one loop

GeometricNeuronV24
    -> ask for another measurement when ambiguity would change which write is justified

IttnasNoruen
    -> protect computational continuations/counterfactual signatures, not only stored outputs

Kompressori
    -> learn/store low-rank operator changes rather than trying to preserve the entire response operator

CausalHorizon
    -> use directed propagated overlap to predict whether low-rank edits collide

Operaattori
    -> persistent structure can eventually compile the slow operator being rewritten

BlackBoxLab
    -> preserve useful alternative computations rather than enforcing diversity through destructive depletion
```

Several generic old questions can now retire. We no longer need new repos merely to ask whether history matters, whether delayed reward can work, whether active sensing helps, or whether coordinates may drift.

The harder object is now:

```text
maintain several ways of computing
        +
know which distinctions are currently identifiable
        +
route among them
        +
return delayed consequence to the continuing route
        +
rewrite the operator without colliding with other retained routes
        +
keep enough alternative structure for tomorrow's different question
```

That is much closer to the original genetic-algorithm intuition than “a neuron runs a GA.”

The evolutionary principle has moved upward:

```text
maintain alternatives
-> expose them to context
-> amplify some
-> credit identifiable continuations
-> propose local operator changes
-> test compatibility with retained causal highways
-> consolidate without destroying the repertoire
```

---

## Next

T4 is exact and linear on purpose. The next serious gate should **estimate** causal overlap in a nonlinear adaptive model and attack it with the boring alternatives:

- gradient/parameter cosine;
- Fisher-like overlap;
- static Jacobian/subspace overlap;
- finite joint candidate evaluation;
- propagated causal-overlap estimate.

The target is not another pretty theorem. It is a practical **collision radar for learning**: decide before committing two updates whether they will destroy one another's computational route.

After that, V25's adaptive resonant credit can be brought back into the same moving/rewriting system, and only later should we remove the seeded direct/relational approaches and ask whether computational highways emerge spontaneously.

The strong phrase remains a target, not a current claim:

> **self-rewriting spectral router over discovered computational highways**

---

## Run

```bash
python -m experiments.gate_t0_fusion
python -m experiments.gate_t1_moving_road
python -m experiments.gate_t2_credit_on_moving_road
python -m experiments.gate_t3_identifiability_credit
python -m experiments.gate_t4_causal_collision
pytest -q
```
