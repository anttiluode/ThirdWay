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

Attackers destroy the address in different ways:

- **pooled**: collapse all eligibility traces to one shared scalar;
- **shuffled**: keep temporal memory but swap route identity;
- **current-only**: discard the delayed causal trace entirely.

On the fixed 8-seed battery (`5000` events/seed, delays `3..10`):

| learner | late route accuracy | late scalar reward | task accuracy |
|---|---:|---:|---:|
| **route-local eligibility** | **0.8139** | **0.8130** | **0.9068** |
| pooled eligibility | 0.5121 | 0.5179 | 0.7591 |
| current-only | 0.5121 | 0.5179 | 0.7591 |
| shuffled route identity | 0.0091 | 0.0147 | 0.5074 |

The result is deliberately narrower than the phrase “self-rewriting spectral router.” What T0 establishes is:

```text
separated computational modes
+ context-dependent multiplicative gain
+ delayed unlabelled consequence
+ route-local eligibility
----------------------------------------
= gains can be rewritten without putting the route address in the reward
```

Pooling preserves some memory but destroys the computational address and therefore cannot learn the context-specific router. Shuffling the address is worse than forgetting it: it confidently reinforces the wrong highway.

## What T0 does *not* establish

- The two computational modes are seeded, not spontaneously discovered.
- Their internal geometry is fixed; only their routing gains change.
- The route-local traces are decaying rather than resonant/off-grid adaptive poles.
- “Covariant highway” is therefore still a target for later gates, not a result of T0.

## Next gates

**T1 — moving road.** Allow each computational family to drift in parameter coordinates while preserving its counterfactual identity. Test whether routing can track continuation of a mode rather than a fixed vector.

**T2 — spectral local credit.** Replace scalar decays with damped resonant eligibility and make the useful pole change. The route should adapt its temporal law without the reward carrying an address.

**T3 — two useful modes on one route.** Attack purification directly: one physical route must preserve two causal temporal modes without averaging or extinguishing one.

**T4 — discovered approaches.** Remove the hand-labelled direct/relational decomposition. Evolve generic tiny networks, cluster survivors only by interventions/Jacobians/hidden trajectories, and ask whether context can amplify the discovered computational families while delayed local consequence rewrites the router.

That is the first point at which ThirdWay would deserve the strong phrase:

> **self-rewriting spectral router**

Until then, the repo keeps the claims smaller than the idea.

## Run

```bash
python -m experiments.gate_t0_fusion
pytest -q
```
