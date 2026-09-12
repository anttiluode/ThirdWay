# ThirdWay

**Three sols. Two unique views. This is fusion.**

ThirdWay is the deliberately separate fusion arm between:

- [GeometricNeuronV25GeneticAlgoAndFanningNeuron](https://github.com/anttiluode/GeometricNeuronV25GeneticAlgoAndFanningNeuron): separated causal routes, delayed scalar consequence, adaptive eligibility timescales and resonant temporal modes.
- [GAx](https://github.com/anttiluode/GAx): positive/evolutionary operators, modal purification, separated computational approaches, context-dependent gain, and covariant computational subspaces.

The rule here is strict: **ThirdWay does not inherit a conclusion merely because both parent arms make it sound plausible. It builds the smallest executable object that needs both ideas and attacks it with source-destroying controls.**

## Working object

The current hypothesis is not “GA + neuron”. It is:

> **A self-rewriting collection of covariant computational highways.**

A highway has three separable roles:

1. **computation** — a counterfactually identifiable way of solving;
2. **routing** — context changes its relative gain;
3. **credit** — delayed global consequence is reassigned through route-local history rather than carrying its own route label.

A useful mathematical sketch is

```text
context/input
    -> context-dependent positive operator
    -> relative gain over separated computational modes
    -> computation/action
    -> delayed scalar consequence
    x route-local eligibility/history
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

T2 is the first literal fusion gate.

T0 gave us local delayed credit. T1 gave us moving computational identity. T2 lets the computational basis rotate **while scalar consequence is still in flight**.

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

This gives ThirdWay a new failure mode:

> **stale credit** — the memory survives, but it now points at yesterday's computation.

See [`T2_RESULTS.md`](T2_RESULTS.md).

## Where we are

We now have three increasingly strong statements in one executable line:

```text
T0: local material can hold the causal address that reward lacks.
T1: computational identity is continuation, not frozen coordinates.
T2: therefore causal address must move with the continuing computation.
```

This is already more than juxtaposing the parent repos. T2 is a result that only appears after their two abstractions are combined.

We still do **not** yet have the strong object “self-rewriting spectral router over discovered computational highways.” The computations are seeded, the T2 transport is known exactly, and eligibility is still a simple decaying trace rather than an adaptive resonant pole.

## Next gates

**T3 — infer the moving road.** Remove privileged access to `L_t`. Estimate continuation/transport from noisy observations and test delayed credit near ambiguous crossings, mode birth/death, or branching.

**T4 — spectral local credit.** Replace scalar decays with damped resonant eligibility and make the useful pole change while the road also moves.

**T5 — two useful temporal modes on one route.** Attack purification directly: preserve two causal temporal highways without averaging or extinguishing one.

**T6 — discovered approaches.** Remove the hand-labelled direct/relational decomposition. Evolve generic tiny networks, cluster survivors only by interventions/Jacobians/hidden trajectories, and ask whether context can amplify the discovered families while delayed local consequence rewrites a moving router.

That is the first point at which ThirdWay would deserve the strong phrase:

> **self-rewriting spectral router over discovered computational highways**

Until then, the repo keeps the claims smaller than the idea.

## Run

```bash
python -m experiments.gate_t0_fusion
python -m experiments.gate_t1_moving_road
python -m experiments.gate_t2_credit_on_moving_road
pytest -q
```

See [`THEORY.md`](THEORY.md) for the formal object and gate sequence.
