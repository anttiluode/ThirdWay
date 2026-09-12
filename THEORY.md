# ThirdWay theory — covariant computational highways

ThirdWay starts from two independently developed questions.

**GAx** asks how several computational approaches can remain dynamically distinct, how context changes their gain, and how an evolving operator changes the meaning of a “mode.”

**GeometricNeuronV25** asks how mixed delayed consequence can find and rewrite the local causal route that produced it, including adaptive eligibility lifetimes and resonant temporal modes.

The fusion object is therefore not “a neuron running a GA.” It is:

> **a self-rewriting collection of identifiable, covariant computational highways.**

The central difficulty is no longer merely learning a good answer. It is preserving several useful ways of computing while the substrate, representation, credit process and operator all change.

---

## 1. Computational identity is counterfactual and covariant

A computational mode is not merely a parameter cluster or output cluster.

Two systems can give the same observed answer and still implement different computations. A mode is distinguished by interventions, Jacobians, hidden trajectories, ablations or pathway usage.

Write the computational families as moving subspaces

```math
E_1(t),E_2(t),\ldots,E_K(t).
```

For a self-changing operator `L_t`, continuation means approximately

```math
\boxed{L_tE_k(t)\approx E_k(t+1).}
```

Identity is therefore **continuation through dynamics**, not a frozen coordinate.

Useful diagnostics are within-mode gain

```math
g_k(t)=\|P_k(t+1)L_tP_k(t)\|
```

and cross-mode leakage

```math
\ell_{jk}(t)=\|P_j(t+1)L_tP_k(t)\|,\qquad j\ne k.
```

A router should change the `g_k` strongly with context while keeping leakage low enough that distinct approaches do not collapse into a lethal average.

---

## 2. Fast routing is modal gain, not necessarily an explicit expert index

Let `q_t` be positive amplitudes over available computational modes. A context `c_t` induces a positive gain operator

```math
D(c_t)=\operatorname{diag}(e^{a_1(c_t)},\ldots,e^{a_K(c_t)}).
```

A minimal routing step is projective normalization:

```math
\boxed{q_{t+1}=\frac{D(c_t)q_t}{\mathbf 1^\top D(c_t)q_t}.}
```

This is the early meaning of **spectral router** in ThirdWay: context changes relative growth rates and the dynamics perform the selection.

GAx supplies the evolutionary interpretation. A lifted mutation-selection system has

```math
q_{g+1}=L_gq_g,
```

with the observed normalized population as a projective shadow. Repeated selection purifies modes; preserving a repertoire means preventing every useful alternative from acquiring a strongly negative long-term growth rate.

---

## 3. Reward can stay scalar if causal address stays local

A routed computation acts. A later modulation `m_t` returns, but the modulation need not contain a route label.

In the simplest form, each route keeps an eligibility trace

```math
e_{c,k}(t+1)=\rho e_{c,k}(t)+u_{c,k}(t).
```

Delayed consequence rewrites modal gain through

```math
\boxed{a_k(c)\leftarrow a_k(c)+\eta m_t e_{c,k}(t).}
```

Architecturally:

```text
reward: scalar
address: local material state
```

T0 establishes this in the seeded two-route laboratory.

---

## 4. Credit identity must move with computational identity

If the computational basis moves while reward is in flight, a persistent trace can survive and still become wrong.

A physical eligibility vector must be transported with the continuing road:

```math
\boxed{e^{\rm phys}(t+1)=\rho L_t e^{\rm phys}(t)+u(t).}
```

T2 names the failure when this does not happen:

> **stale credit** — the event is remembered, but the memory now points at yesterday's computation.

The compact rule is

```math
\boxed{\textbf{credit identity must be covariant with computational identity}.}
```

---

## 5. Identifiability sets the legal resolution of credit

Covariance is not enough if the current observation cannot identify individual modes.

When several axes become nearly degenerate, the stable observable object may be only a block

```math
\mathcal B(t)=\operatorname{span}\{E_i(t):i\in S\}.
```

ThirdWay therefore imposes

```math
\boxed{\textbf{resolution of credit}\le\textbf{resolution of identifiable computation}.}
```

There are three operational cases:

```text
individual route identifiable
    -> route-specific credit is legal

only a block identifiable
    -> preserve/transport the block, do not invent axis labels

fine write required but fine identity unobservable
    -> acquire another measurement/intervention before writing
```

T3 connects MovingProblem, AlgoSchalgo, V24 and Child here. In an exact hidden-gauge interval the primary operator genuinely cannot reveal the internal semantic rotation; a selectively queried second operator restores the needed distinction.

---

## 6. Temporal credit can itself be a searchable spectrum

V25 replaces a single decay with damped resonant states

```math
z_{k,j}(t+1)=\rho_j e^{i\omega_j}z_{k,j}(t)+u_k(t).
```

The pair `(rho, omega)` describes causal lifetime and temporal carrier.

V25 F9 goes further: the pole itself can evolve toward hidden causal dynamics rather than merely selecting from a fixed bank.

ThirdWay has not yet fused that adaptive pole into the moving-road/collision machinery. The ordering now matters:

```text
first: know what computational distinction is identifiable
then: keep credit attached to that continuation
then: refine the temporal carrier used to hold the causal history
```

Otherwise one can build an exquisitely tuned temporal trace attached to the wrong or unidentifiable computation.

---

## 7. Slow learning should be represented as an operator edit

Kompressori measured an important asymmetry:

```text
full finite-time response operator J      high-rank
experience-induced change Delta J          much lower-rank
```

So a route-local slow rewrite can be modeled as

```math
\Delta L_i=U_iM_iV_i^*.
```

The question is no longer only whether edit `i` is useful by itself. It is whether several useful edits can coexist.

CausalHorizon gives the exact time-ordered linear interaction coordinate

```math
\boxed{\Omega_{ji}=V_j^*\Phi(\tau_j,\tau_i+1)U_i}. 
```

This motivates a fourth stage in the genetic/evolutionary cartoon:

```text
variation
 -> local success
 -> compatibility selection
 -> retention
```

**Compatibility selection** asks whether the consequence of one retained change propagates into the coordinates in which another change acts.

T4 proves the need for that distinction in an exact linear transport laboratory: raw parameter overlap has zero precision and recall while propagated causal overlap classifies every collision.

---

## 8. T5: nonlinear causal susceptibility

T5 asks whether the T4 idea survives when exact linear event algebra is gone.

The substrate is

```math
x_{t+1}=\tanh(Ax_t+b_t).
```

For a later edit `j`, define its marginal action

```math
m_j(x)=f_{A+E_j}(x)-f_A(x).
```

The finite nonlinear collision caused by an earlier edit `i` is

```math
C_{i\to j}=\|m_j(x_j^{(i)})-m_j(x_j)\|.
```

The radar does **not** run the joint pair. It measures the isolated effect of `i`, propagates that perturbation through the unedited tangent trajectory,

```math
\widehat{\delta x}_j=J_{j-1}\cdots J_{i+1}r_i,
```

then applies the susceptibility of the later edit

```math
G_j=\frac{\partial m_j}{\partial x}\bigg|_{x_j}.
```

The score is

```math
\boxed{s_{\rm causal}=\|G_j\widehat{\delta x}_j\|.}
```

Across 192 finite cases, with thresholds fit on source sites 0–7 and tested on held-out sites 8–15:

```text
correlation with finite collision
parameter cosine                 -0.411329
activation / Fisher-ish proxy    -0.411329
static Jacobian cosine           -0.410830
propagated causal susceptibility  0.994736

held-out collision recall
parameter cosine                  0.000000
activation / Fisher-ish proxy     0.000000
static Jacobian cosine            0.000000
propagated causal susceptibility  1.000000
```

The causal radar also reaches 1.000 held-out accuracy and precision in this construction.

This is stronger than T4 but still narrow. It shows that a **single-edit first-order causal forecast can predict a finite nonlinear interaction on unseen locations**. It does not yet establish continual learning in a trained network.

---

## 9. Two timescales of rewrite

ThirdWay keeps two operations distinct.

### Gain rewrite

```math
a_k(c)\rightarrow a_k'(c)
```

means **use this existing approach more or less in this situation**.

### Geometry/operator rewrite

```math
E_k(t)\rightarrow E_k(t+1)
```

or

```math
L_{t+1}=L_t+\Delta L_t
```

means **change how the approach itself computes or propagates**.

The working hypothesis is

```math
\boxed{\text{fast: route among approaches}}
```

and

```math
\boxed{\text{slow: rewrite approaches/operator while preserving identities and compatibility}.}
```

Without that separation, every successful context can pull all highways together and return the system to mode purification.

---

## 10. Failure modes

ThirdWay now distinguishes at least eight collapses.

**Extinction** — one computational family loses essentially all amplitude.

**Mixing** — multiple families remain but blend into an invalid average.

**Geometry drift** — a family survives but the router no longer recognizes its continuation.

**Stale credit** — memory survives in obsolete coordinates and reinforces the wrong current computation.

**Over-specific credit** — the learner writes at a finer route resolution than observation justifies.

**Update collision** — two locally effective operator edits interfere after propagation through the substrate.

**Spectral aliasing** — different causal histories collapse onto indistinguishable temporal modes.

**Purification collapse** — repeated success drives alternatives so far down that later contexts can no longer recover them cheaply.

---

## 11. Gate ladder

### T0 — local address, global consequence

**Implemented.** Route-local routing `0.952750`; pooled/current-only `0.512125`; route-shuffled `0.043000`.

### T1 — moving road

**Implemented.** Fixed-coordinate identity `0.519271`; continuation identity `1.000000`.

### T2 — delayed credit on a moving road

**Implemented.** Covariant routing `0.952750`; stale-coordinate routing `0.184500`.

### T3 — identifiability-aware credit

**Implemented.** Ambiguity blocks preserve `0.9525` routing through noisy crossings; a selective second operator restores `0.9453` routing in an exact hidden-gauge attack.

### T4 — exact causal compatibility

**Implemented.** Propagated overlap classifies all exact linear collisions; static overlap has zero precision and recall.

### T5 — nonlinear collision radar

**Implemented.** Propagated causal susceptibility has `r=0.994736` with finite nonlinear collision strength and 1.000 held-out accuracy/precision/recall; three static attackers have zero held-out recall.

### T6 — persistent skill edits

Next target. Move from time-local edit events to persistent task-specific low-rank changes in a shared recurrent model. Compare:

- parameter and gradient cosine;
- a genuine Fisher-style attacker;
- static Jacobian/subspace overlap;
- propagated causal radar;
- oracle finite joint evaluation.

Measure old-skill damage, cross-mode leakage and switching cost.

### T7 — active sensing for write resolution

Do not supply the diagnostic operator family. Let the system choose which costly intervention will reduce *write-relevant* identity ambiguity.

### T8 — resonant moving credit

Combine covariant computational identity with V25's adaptive temporal poles. Test whether computational identity and temporal-credit identity can both move without aliasing.

### T9 — discovered computational approaches

Remove the seeded direct/relational decomposition. Evolve/train generic small networks, identify computational families only by interventions/Jacobians/hidden trajectories, then test routing, delayed credit, active ambiguity resolution and compatible persistent rewrite.

Only that final stage could support the strong target phrase:

```math
\boxed{\textbf{self-rewriting spectral router over discovered computational highways}.}
```

Until then, ThirdWay keeps smaller names for smaller results.
