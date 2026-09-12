# Gate T0 — Delayed scalar consequence, local route address

## Question

Can a context-dependent router learn from a delayed **scalar** consequence when the reward packet contains no route/context label, provided the computational highways keep their own local eligibility histories?

This is the first fusion gate. The two computations are deliberately seeded so T0 tests **routing + credit assignment**, not spontaneous expert discovery.

## Computational highways

Two fixed counterfactually distinct mechanisms are available:

```math
y_d=\tanh(4x_0),\qquad y_r=\tanh(4x_1x_2).
```

Context A makes `x0` causal. Context B makes `x1*x2` causal. The irrelevant cue is random, so selecting the wrong mechanism is genuinely harmful.

For each context, route logits define a normalized positive gain over the two modes. An 8% exploration floor prevents an early stochastic winner from becoming irreversible.

Each chosen route emits a local policy-score event. The event enters a decaying `(context, route)` eligibility location. Its scalar consequence returns after a uniformly random delay of 3–10 later events. The reward queue stores only the scalar.

The update is

```math
\Delta a_{c,k}\propto m(t)e_{c,k}(t),
```

where the address `(c,k)` is present in the substrate, not in `m(t)`.

## Controls

- `local`: keep the full `(context, route)` eligibility matrix.
- `pooled`: replace the matrix by its mean before every delayed update.
- `current`: discard delayed causal history.
- `shuffled`: preserve temporal memory but swap the route address before updating.

## Fixed battery

Eight seeds, 5000 events per seed. Metrics are measured over the final 1000 events of each run.

| learner | late route accuracy | late scalar reward | task accuracy | worst-seed route accuracy |
|---|---:|---:|---:|---:|
| **route-local eligibility** | **0.952750** | **0.955109** | **0.977875** | **0.932000** |
| pooled eligibility | 0.512125 | 0.517902 | 0.759125 | 0.501000 |
| current-only | 0.512125 | 0.517902 | 0.759125 | 0.501000 |
| shuffled route identity | 0.043000 | 0.049467 | 0.524750 | 0.034000 |

Mean final route probabilities for `local`:

```text
context A -> [direct 0.957808, relational 0.042192]
context B -> [direct 0.044969, relational 0.955031]
```

For pooled/current-only the router remains exactly `[0.5, 0.5]` in both contexts. Shuffling produces the near-opposite router:

```text
context A -> [direct 0.042518, relational 0.957482]
context B -> [direct 0.955742, relational 0.044258]
```

## Interpretation

T0 supports a narrow statement:

> **Delayed global consequence can rewrite context-dependent modal gains without carrying its own route address, if the substrate preserves route-local causal history.**

The shuffled attacker is particularly useful. It has memory and receives the same scalar rewards, but systematically binds consequence to the wrong highway. The result is not merely failure to learn; it learns the wrong router.

So the useful resource is not generic memory:

```math
\boxed{\text{temporal persistence} + \text{computational address}}
```

Pooling preserves persistence while destroying address, and loses the effect.

## What this does not prove

T0 does **not** yet demonstrate a self-rewriting spectral router in the strong sense:

- the computational families are seeded;
- their geometry is fixed;
- eligibility is a scalar decay, not a resonant/off-grid adaptive pole;
- only routing gain is rewritten;
- there is no covariant continuation test under moving parameter coordinates.

Those are the next boundaries rather than conclusions smuggled into T0.
