# ThirdWay theory — covariant computational highways

ThirdWay starts from a separation of roles that the parent arms reached independently.

GAx asks how multiple computational approaches can remain dynamically distinct and be selected by context. V25 asks how a mixed future consequence can find and rewrite the local causal route that produced it.

The fusion object is therefore not “a neuron running a GA.” It is a dynamical system with **separated computational continuations, context-dependent gain, local causal history, and slow operator rewrite**.

## 1. Computational mode

A computational mode is not merely a parameter cluster or an output cluster.

Two states can produce the same observed answer and still implement different computations. A mode is distinguished counterfactually: there exists an intervention `I` for which their response, Jacobian, hidden trajectory, ablation sensitivity, or pathway usage differs systematically.

Write the computational families as moving subspaces

```math
E_1(t),E_2(t),\ldots,E_K(t).
```

For a self-changing operator `L_t`, a useful highway should approximately continue as

```math
\boxed{L_tE_k(t)\approx E_k(t+1).}
```

This is the covariant-road idea. Identity is continuation through time, not fixed coordinates.

Useful diagnostics are

```math
g_k(t)=\|P_k(t+1)L_tP_k(t)\|
```

for within-mode gain and

```math
\ell_{jk}(t)=\|P_j(t+1)L_tP_k(t)\|,\qquad j\ne k
```

for cross-mode leakage.

A router should be able to change the `g_k` strongly with context while keeping leakage low enough that distinct approaches do not collapse into a lethal average.

## 2. Fast routing

Let `q_t` be positive amplitudes over currently available computational modes. A context `c_t` induces a positive gain operator

```math
D(c_t)=\operatorname{diag}(e^{a_1(c_t)},\ldots,e^{a_K(c_t)}).
```

A minimal routing step is projective normalization:

```math
\boxed{q_{t+1}=\frac{D(c_t)q_t}{\mathbf 1^\top D(c_t)q_t}.}
```

This is the cleanest mathematical meaning of “spectral router” in the early gates: context changes relative modal gain and the dynamics perform the selection.

It does not require a separate classifier whose output is an expert index.

## 3. Local causal history

A routed computation acts on the world. A later modulation `m_t` returns, but the modulation need not contain the route address.

Each highway retains its own eligibility state. In the simplest T0 form,

```math
e_{c,k}(t+1)=\rho e_{c,k}(t)+u_{c,k}(t),
```

where `u_{c,k}` is the local event generated when context `c` used route `k`.

Then a delayed global scalar can rewrite route gain through

```math
\boxed{a_k(c)\leftarrow a_k(c)+\eta m_t e_{c,k}(t).}
```

The important point is architectural:

```text
reward: scalar
address: local material state
```

The reward does not carry `(c,k)` because the substrate has preserved it.

## 4. Covariant credit

If the computational basis itself moves, a persistent eligibility trace can still become wrong.

Let `E(t)` be the current mode basis and

```math
L_tE(t)\approx E(t+1).
```

A physical trace storing yesterday's route must therefore move too:

```math
\boxed{e^{\rm phys}(t+1)=\rho L_t e^{\rm phys}(t)+u(t).}
```

Modal credit at the current time is recovered by projection:

```math
\hat e(t)=E(t)^\top e^{\rm phys}(t).
```

T2 shows why this is not cosmetic. If the memory simply decays in yesterday's coordinates, then delayed reward can reinforce a different current mode even though the memory itself never vanished.

This gives the compact rule

```math
\boxed{\text{credit identity must be covariant with computational identity}.}
```

## 5. Spectral local credit

V25 suggests replacing a single decay by resonant temporal states,

```math
z_{k,j}(t+1)=\rho_j e^{i\omega_j}z_{k,j}(t)+u_k(t).
```

The coordinates `(rho, omega)` describe causal lifetime and temporal carrier. A later scalar consequence can interact with the local phase/history without transporting an explicit route label.

A stronger future ThirdWay gate will let the pole itself move,

```math
(\rho_k(t),\omega_k(t))\rightarrow(\rho_k(t+1),\omega_k(t+1)),
```

while the computational road is moving too.

That would make credit assignment itself a slowly rewritten dynamical object nested inside a moving computational mode.

## 6. Two kinds of rewrite

The fusion requires keeping two operations distinct.

### Gain rewrite

```math
a_k(c)\rightarrow a_k'(c)
```

means **use this existing approach more or less in this situation**.

### Geometry rewrite

```math
E_k(t)\rightarrow E_k(t+1)
```

means **change how the approach computes**.

The working hypothesis is a timescale separation:

```math
\boxed{\text{fast: route among approaches}}
```

and

```math
\boxed{\text{slow: rewrite approaches/router while preserving continuation identity}.}
```

Without that separation, repeated success can simply purify one mode and erase the repertoire.

## 7. Failure modes

ThirdWay now distinguishes at least five different collapses.

**Extinction** — one computational family loses essentially all amplitude.

**Mixing** — multiple families remain populated but their internal computations blend into an invalid average.

**Geometry drift** — a family still exists, but moves so far that the old routing structure no longer tracks its continuation.

**Stale credit** — the causal memory survives, but remains attached to old coordinates and therefore reinforces the wrong current computation.

**Spectral aliasing** — two causal histories become indistinguishable because the available temporal basis cannot resolve them.

The first four are conceptually different: a system can remember an event, retain both modes, and still learn incorrectly because the address of the memory failed to move with the mode.

## 8. Gate sequence

### T0 — local address, global consequence

Seed two distinct computations. Learn context-dependent modal gains from delayed scalar consequence. Destroy route address with pooled/shuffled controls.

Status: **implemented and passing**. Route-local late routing accuracy `0.952750`; pooled/current-only `0.512125`; route-shuffled `0.043000`.

### T1 — moving road

Rotate the computational basis through parameter coordinates and compare frozen-coordinate identity with continuation identity.

Status: **implemented and passing**. Fixed-coordinate identity `0.519271`; covariant continuation `1.000000`; exact cocycle residual below `3e-16`.

### T2 — delayed credit on a moving road

Store eligibility in physical coordinates while the basis rotates during the reward delay. Compare transported credit with stale-coordinate credit.

Status: **implemented and passing**. Covariant late routing `0.952750`; stale-coordinate routing `0.184500`; transported physical credit matches the ideal modal shadow to about `1.1e-15`.

This is the first gate that requires both parent abstractions at once.

### T3 — infer the road

Remove privileged access to `L_t`. Infer mode continuation/transport from noisy observations and test delayed credit near ambiguous crossings, branching, birth/death, and temporary occlusion.

### T4 — resonant moving credit

Give each moving route damped resonant eligibility states. Hidden causal frequencies change. Ask whether the temporal pole can adapt while credit remains attached to the continuing computational identity.

### T5 — separation against purification

Place two useful temporal modes on the same physical route. Test whether adaptation can preserve both rather than converge to one pole or average them.

### T6 — discovered computational approaches

Remove labelled direct/relational pathways. Evolve/train generic tiny networks on an ambiguous world. Cluster survivors only by interventions, Jacobians, hidden trajectories and ablations. Then test contextual modal gain and delayed local credit over the discovered moving families.

That is the first gate capable of supporting the strong claim:

```math
\boxed{\textbf{self-rewriting spectral router over discovered computational highways}.}
```

Until then, ThirdWay uses smaller names for smaller results.
