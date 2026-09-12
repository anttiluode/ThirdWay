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

## 4. Spectral local credit

V25 suggests replacing a single decay by resonant temporal states,

```math
z_{k,j}(t+1)=\rho_j e^{i\omega_j}z_{k,j}(t)+u_k(t).
```

The coordinates `(rho, omega)` describe causal lifetime and temporal carrier. A later scalar consequence can interact with the real/complex phase of a local trace without transporting an explicit route label.

A stronger future ThirdWay gate will let the pole itself move:

```math
(\rho_k(t),\omega_k(t))\rightarrow(\rho_k(t+1),\omega_k(t+1)).
```

That would make credit assignment itself a slowly rewritten dynamical object.

## 5. Two kinds of rewrite

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

## 6. Failure modes

ThirdWay will explicitly distinguish at least three collapses.

**Extinction** — one computational family loses essentially all amplitude.

**Mixing** — multiple families remain populated but their internal computations blend into an invalid average.

**Geometry drift** — a family still exists, but moves so far that the old routing/credit structure no longer tracks its continuation.

A fourth failure becomes relevant once temporal poles can adapt:

**spectral aliasing** — two causal histories become indistinguishable because the available temporal basis cannot resolve them.

## 7. Gate sequence

### T0 — local address, global consequence

Seed two distinct computations. Learn context-dependent modal gains from delayed scalar consequence. Destroy route address with pooled/shuffled controls.

Status: implemented.

### T1 — moving road

Construct two computational families whose parameter coordinates rotate/drift while their counterfactual functions remain identifiable. Compare fixed-vector routing with subspace-continuation routing.

Target result:

```math
\text{fixed coordinate identity fails},\qquad
L_tE_k(t)\approx E_k(t+1)\text{ remains trackable}.
```

### T2 — resonant credit

Give each route a bank of damped resonant eligibility states. Hidden causal delays/frequencies change. Ask whether route-local spectral states rewrite gains better than pooled frequency banks.

### T3 — separation against purification

Place two useful temporal modes on the same physical route. Test whether adaptation can preserve both rather than converge to one pole or average them.

### T4 — discovered computational approaches

Remove labelled direct/relational pathways. Evolve/train generic tiny networks on an ambiguous world. Cluster survivors only by interventions, Jacobians, hidden trajectories and ablations. Then test contextual modal gain and delayed local credit over the discovered families.

That is the first gate capable of supporting the strong claim:

```math
\boxed{\textbf{self-rewriting spectral router over discovered computational highways}.}
```

Until then, ThirdWay uses smaller names for smaller results.
