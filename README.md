# ThirdWay

**Three sols. Two unique views. This is fusion.**

ThirdWay is the deliberately separate fusion arm between:

- [GeometricNeuronV25GeneticAlgoAndFanningNeuron](https://github.com/anttiluode/GeometricNeuronV25GeneticAlgoAndFanningNeuron): separated causal routes, delayed scalar consequence, adaptive eligibility timescales and resonant temporal modes.
- [GAx](https://github.com/anttiluode/GAx): positive/evolutionary operators, modal purification, separated computational approaches, context-dependent gain, and covariant computational subspaces.

The rule here is strict: **ThirdWay does not inherit a conclusion merely because both parent arms make it sound plausible. It builds the smallest executable object that needs both ideas and attacks it with source-destroying controls.**

The broader older-repo map is now explicit in [`RETROSPECTIVE.md`](RETROSPECTIVE.md).

## Working object

The current hypothesis is not “GA + neuron”. It is:

> **A self-rewriting collection of covariant computational highways.**

A highway has four separable roles:

1. **computation** — a counterfactually identifiable way of solving;
2. **routing** — context changes its relative gain;
3. **credit** — delayed global consequence is reassigned through route-local history rather than carrying its own route label;
4. **identity resolution** — the system tracks only the finest computational distinction its current evidence can actually support.

A useful sketch is

```text
context/input
    -> observable operator family
    -> identifiable route / ambiguity block
    -> context-dependent modal gain
    -> computation/action
    -> delayed scalar consequence
    x covariant local eligibility/history
    -> slow rewrite of future gains / route geometry
```

The two timescales matter. Fast dynamics should route among approaches. Slow dynamics may rewrite the approaches or the operator that selects them. If every winner immediately rewrites everything, the system risks returning to simple mode purification.

## Gate T0 — the reward has no address, the substrate does

Two seeded, counterfactually distinct computations coexist:

```math
y_d=\tanh(4x_0), \qquad y_r=\tanh(4x_1x_2).
```

In context A the direct cue is reliable; in context B the relational cue is reliable. A positive diagonal gain operator defines the route probabilities. The selected route acts, but its scalar consequence is returned only after several unrelated events.

On the fixed 8-seed battery (`5000` events/seed, delays `3..10`):

| learner | late route accuracy | late scalar reward | task accuracy | worst-seed route accuracy |
|---|---:|---:|---:|---:|
| **route-local eligibility** | **0.952750** | **0.955109** | **0.977875** | **0.932000** |
| pooled eligibility | 0.512125 | 0.517902 | 0.759125 | 0.501000 |
| current-only | 0.512125 | 0.517902 | 0.759125 | 0.501000 |
| shuffled route identity | 0.043000 | 0.049467 | 0.524750 | 0.034000 |

T0 says:

> **Delayed consequence can rewrite context-dependent modal gain without carrying its own route label, provided the substrate preserves route-local causal history.**

See [`T0_RESULTS.md`](T0_RESULTS.md).

## Gate T1 — the road moves, identity continues

T1 attacks the assumption that a computational highway is a fixed vector in parameter coordinates.

Two persistent mode identities rotate continuously through a 2-D basis. At each step their observed vectors are randomly reordered, sign-flipped and noised. A fixed-coordinate matcher keeps comparing against initialization. A continuation tracker asks which yesterday-road each new candidate became.

The noiseless cocycle is

```math
L_t = E(t+1)E(t)^\top,
\qquad
\boxed{L_tE_k(t)=E_k(t+1)}.
```

Fixed 8-seed battery:

| observer | identity accuracy | contextual route reward |
|---|---:|---:|
| fixed initial coordinates | **0.519271** | **0.038542** |
| **continuation / covariant tracker** | **1.000000** | **1.000000** |

Worst-seed continuation accuracy is `1.000000`; maximum exact covariance residual is `2.937e-16`.

T1 says:

> **A highway can move completely away from its initial coordinates while retaining identity by dynamical continuation.**

See [`T1_RESULTS.md`](T1_RESULTS.md).

## Gate T2 — credit must move with the road

T2 lets the computational basis rotate **while scalar consequence is still in flight**.

A local eligibility event is stored in physical coordinates. Two systems receive the same delayed rewards:

- **covariant:** transport eligibility with the moving basis before reward arrives;
- **stale:** let eligibility persist in yesterday's coordinates.

The basis rotates by `0.32 rad/event`, while rewards arrive `3..10` events later.

| credit rule | late route accuracy | late scalar reward | task accuracy | worst-seed route accuracy |
|---|---:|---:|---:|---:|
| **covariantly transported** | **0.952750** | **0.955109** | **0.977875** | **0.932000** |
| stale coordinates | **0.184500** | **0.196868** | **0.598500** | **0.054000** |

For covariant transport, the maximum mismatch between the physical trace projected into the current basis and an ideal modal shadow is only `1.110e-15`.

T2 says:

```math
\boxed{\textbf{credit identity must be covariant with computational identity}.}
```

This gives ThirdWay a failure mode distinct from forgetting:

> **stale credit** — the memory survives, but it now points at yesterday's computation.

See [`T2_RESULTS.md`](T2_RESULTS.md).

## Gate T3 — credit cannot be finer than identifiable computation

T2 cheated in one important way: the continuation transport was known.

T3 removes that privilege and connects the fusion line to **MovingProblem**, **AlgoSchalgo**, **GeometricNeuronV24**, and **Child**.

### T3A — noisy crossing

Four spectral modes move through changing coordinates. The two useful computational modes repeatedly become nearly degenerate. An ordinary vector-overlap tracker tries to keep naming both axes separately. The ambiguity-block tracker changes the tracked object itself: when the gap becomes too small, it tracks the whole identifiable subspace and carries the previous semantic basis through that subspace with a Procrustes lift.

Eight-seed battery:

| tracker | late route accuracy | late reward | task accuracy | late semantic fidelity |
|---|---:|---:|---:|---:|
| vector overlap | **0.5115** | 0.5027 | 0.7515 | 0.4565 |
| **ambiguity blocks** | **0.9525** | **0.9541** | **0.9774** | **0.9967** |
| oracle frame | 0.9525 | 0.9541 | 0.9774 | 1.0000 |

The block is active for about 18.8% of the trajectory.

So:

```math
\boxed{\text{when the vector is not identifiable, stop making vector-level claims}.}
```

### T3B — exact hidden-gauge attack

Then the easy story is deliberately broken.

The first two eigenvalues become **exactly equal**, and the hidden semantic axes rotate by 90 degrees inside that degenerate subspace. The primary operator is invariant to the rotation. A passive tracker therefore has no legitimate information about the individual axes.

A second diagnostic operator shares the same computational frame but has a nondegenerate signature. The active arm queries it only while the primary eigengap says individual identity is unsupported.

| tracker | late route accuracy | late reward | task accuracy | late semantic fidelity | extra query fraction |
|---|---:|---:|---:|---:|---:|
| overlap | 0.6804 | 0.6788 | 0.8396 | 0.5101 | 0 |
| ambiguity blocks only | **0.3884** | 0.3782 | 0.6893 | **0.0210** | 0 |
| **active operator family** | **0.9453** | **0.9419** | **0.9713** | **0.9992** | **0.2002** |
| oracle frame | 0.9525 | 0.9541 | 0.9774 | 1.0000 | 0 |

This earns the strongest rule in the repo so far:

```math
\boxed{\textbf{resolution of credit} \;\le\; \textbf{resolution of identifiable computation}.}
```

If a finer write matters and the present observation cannot support it, **ask another question before writing**.

See [`T3_RESULTS.md`](T3_RESULTS.md).

## What the older repos look like from here

The detailed map is in [`RETROSPECTIVE.md`](RETROSPECTIVE.md). The shortest version is:

```text
SighImageSuper   -> a distinction is usable memory only if a future question can recover it
MovingProblem    -> moving coordinates are fine; unjustified identity is the danger
AlgoSchalgo      -> ambiguity blocks define the legal granularity of credit
Child            -> delayed audit + active sensing + provenance + representation drift become one loop
V24              -> measure when an ambiguity would change which write is justified
IttnasNoruen     -> preserve counterfactual computational signatures, not just a bag of old outputs
Kompressori      -> the full operator is high-rank but experience-induced ΔJ can be low-rank
CausalHorizon    -> propagated causal overlap, not distance, should predict update collision
Operaattori      -> slow structural variables can physically compile the future operator
BlackBoxLab      -> preserve useful computational alternatives, not diversity by destructive depletion
```

This lets several old generic questions retire. We no longer need separate projects asking merely whether history matters, whether delayed reward can work, whether active sensing helps, or whether a representation may move. The hard problem has become **maintaining and rewriting multiple identifiable computational continuations without collapsing them into one**.

## Next gate — collision-aware operator rewrite

The most useful next step is no longer another frequency toy.

Kompressori found that the response operator can be high-rank while an experience-induced change `ΔJ` is low-rank, and that separated updates add while overlapping updates interfere. CausalHorizon gives an exact linear measure of directed propagated event overlap,

```math
\Omega_{ji}=V_j^*\Phi(\tau_j,\tau_i+1)U_i.
```

So **T4** should give each computational highway a low-rank operator edit and compare:

1. naive addition of successful edits;
2. parameter-space orthogonalization;
3. static subspace-overlap guarding;
4. **directed propagated causal-overlap guarding**;
5. oracle joint finite evaluation.

The question is no longer only “which route deserves reward?” It is:

> **Can two individually useful pieces of learning coexist dynamically, and can we predict that before committing them?**

That would move the line from routing and credit assignment into **compatibility of learning itself**.

## Run

```bash
python -m experiments.gate_t0_fusion
python -m experiments.gate_t1_moving_road
python -m experiments.gate_t2_credit_on_moving_road
python -m experiments.gate_t3_identifiability_credit
pytest -q
```

See [`THEORY.md`](THEORY.md) for the formal object and [`RETROSPECTIVE.md`](RETROSPECTIVE.md) for the old-repo handoff.