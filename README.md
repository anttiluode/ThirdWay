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

T0 is intentionally small and partially seeded. It does **not** claim spontaneous discovery of experts yet.

Two counterfactually different computations coexist:

```math
y_d=\tanh(4x_0), \qquad y_r=\tanh(4x_1x_2).
```

In context A the direct cue is reliable; in context B the relational cue is reliable. A positive diagonal gain operator defines the route probabilities. The selected route acts, but its scalar consequence is returned only after several unrelated events.

The reward packet contains no `(context, route)` tag. Instead each route/context location keeps a decaying local policy-score eligibility trace. The delayed scalar multiplies those traces and rewrites the context-dependent gains.

On the fixed 8-seed battery (`5000` events/seed, delays `3..10`):

| learner | late route accuracy | late scalar reward | task accuracy | worst-seed route accuracy |
|---|---:|---:|---:|---:|
| **route-local eligibility** | **0.952750** | **0.955109** | **0.977875** | **0.932000** |
| pooled eligibility | 0.512125 | 0.517902 | 0.759125 | 0.501000 |
| current-only | 0.512125 | 0.517902 | 0.759125 | 0.501000 |
| shuffled route identity | 0.043000 | 0.049467 | 0.524750 | 0.034000 |

The learned local router is almost complementary across contexts:

```text
context A -> [direct 0.957808, relational 0.042192]
context B -> [direct 0.044969, relational 0.955031]
```

Pooling/current-only leave both contexts at exactly `[0.5, 0.5]`. Shuffling the route address learns the near-opposite router.

T0 establishes a narrow fusion result:

```text
separated computational modes
+ context-dependent multiplicative gain
+ delayed unlabelled consequence
+ route-local eligibility
----------------------------------------
= gains can be rewritten without putting the route address in the reward
```

See [`T0_RESULTS.md`](T0_RESULTS.md).

## Gate T1 — the road moves, identity continues

T1 attacks the next hidden assumption: that a computational highway is a fixed vector in parameter coordinates.

Two persistent mode identities rotate continuously through a 2-D basis. At each step their observed vectors are randomly reordered, sign-flipped and noised. A fixed-coordinate matcher keeps comparing against the initialization. A continuation tracker asks which yesterday-road each new candidate became.

The noiseless cocycle is

```math
L_t = E(t+1)E(t)^\top,
```

and therefore

```math
\boxed{L_tE_k(t)=E_k(t+1)}.
```

Fixed 8-seed battery:

| observer | identity accuracy | contextual route reward |
|---|---:|---:|
| fixed initial coordinates | **0.519271** | **0.038542** |
| **continuation / covariant tracker** | **1.000000** | **1.000000** |

Worst-seed continuation accuracy is `1.000000`; maximum exact covariance residual is `2.937e-16`.

So a road can move completely away from its initial coordinates, even trade coordinate positions with another road, while retaining its computational identity by continuation.

See [`T1_RESULTS.md`](T1_RESULTS.md).

## What we have — and what we do not

T0 gives the **credit half**:

> consequence can find a route because local material preserved the address.

T1 gives the **identity half**:

> the same route can move through coordinates if identity is defined by continuation rather than a frozen vector.

We still do **not** yet have the strong object “self-rewriting spectral router.” The computational families are seeded, T0 keeps their geometry fixed, and T1 does not yet make delayed consequence rewrite a moving road.

## Next gates

**T2 — consequence on a moving road.** Fuse T0 and T1 literally. Let a computational family drift while delayed scalar reward is still in flight. Compare credit attached to stale coordinates against credit transported with the covariant identity.

**T3 — spectral local credit.** Replace scalar decays with damped resonant eligibility and make the useful pole change. The route should adapt its temporal law without the reward carrying an address.

**T4 — two useful modes on one route.** Attack purification directly: one physical route must preserve two causal temporal modes without averaging or extinguishing one.

**T5 — discovered approaches.** Remove the hand-labelled direct/relational decomposition. Evolve generic tiny networks, cluster survivors only by interventions/Jacobians/hidden trajectories, and ask whether context can amplify the discovered computational families while delayed local consequence rewrites the router.

That is the first point at which ThirdWay would deserve the strong phrase:

> **self-rewriting spectral router over discovered computational highways**

Until then, the repo keeps the claims smaller than the idea.

## Run

```bash
python -m experiments.gate_t0_fusion
python -m experiments.gate_t1_moving_road
pytest -q
```

See [`THEORY.md`](THEORY.md) for the formal object and gate sequence.
