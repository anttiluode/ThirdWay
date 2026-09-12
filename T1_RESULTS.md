# Gate T1 — The road moves, the identity continues

## Question

If a computational family moves through parameter coordinates, can a router preserve its identity by following dynamical continuation rather than matching a fixed vector?

This gate isolates the word **covariant** in “covariant computational highway.” It does not add new learning yet; it attacks the assumption that a route is a stationary direction in parameter space.

## Construction

Two orthogonal computational identities form the columns of a rotating basis

```math
E(t)=R(\theta_t).
```

At every step the two candidate vectors are:

- randomly reordered;
- randomly sign-flipped;
- perturbed by small Gaussian noise.

The hidden identities still continue smoothly.

The exact noiseless transport is

```math
L_t=E(t+1)E(t)^\top,
```

so

```math
\boxed{L_tE_k(t)=E_k(t+1)}.
```

Two identity mechanisms are compared.

### Fixed-coordinate matcher

Compare every new candidate to the original `E_k(0)` forever.

### Continuation matcher

Compare each new candidate to the previous tracked `E_k(t)`, assign the maximum-total-absolute-cosine matching, sign-align it, and use that as the next local identity.

A random context then asks for computational identity 0 or 1. The router gets `+1` for selecting the candidate belonging to the requested continuing identity and `-1` for selecting the other one.

## Fixed 8-seed battery

Each seed runs 240 moving-basis steps. The rotation increment is sampled per seed from `0.025..0.045` radians and observation noise is `0.015`.

| observer | identity accuracy | contextual route reward |
|---|---:|---:|
| fixed initial coordinates | **0.519271** | **0.038542** |
| **continuation / covariant tracker** | **1.000000** | **1.000000** |

Worst-seed continuation identity accuracy:

```text
1.000000
```

Maximum exact cocycle covariance residual over all seeds:

```text
2.937e-16
```

## Interpretation

A computational highway cannot in general be identified with one frozen parameter vector.

The fixed matcher is initially correct, but as the basis rotates it eventually calls one road the other road. Across repeated rotation it falls to chance-level identity even though the underlying computational continuations never cross or disappear.

The continuation tracker preserves identity because it asks a different question:

```text
not:  which initial vector does this look like?

but:  which yesterday-road did this become?
```

That is the operational meaning of

```math
\boxed{L_tE_k(t)\approx E_k(t+1)}.
```

The coordinates can change while the dynamical identity remains continuous.

## Limits

T1 is intentionally geometric and constructed:

- the two moving modes are supplied rather than discovered from a generic network;
- the transport is smooth and low-dimensional;
- there is no extinction, splitting, merger, or ambiguous crossing;
- T1 does not yet couple moving identity back into T0's delayed local-credit rule.

The next serious fusion step is therefore not merely “more rotation.” It is to let the route move **while delayed consequence is trying to rewrite it**. That will test whether the credit trace follows continuation identity or becomes stale in the old coordinates.
