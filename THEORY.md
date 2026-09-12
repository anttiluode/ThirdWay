# ThirdWay theory — covariant computational highways

ThirdWay starts from a separation of roles that the parent arms reached independently.

GAx asks how multiple computational approaches can remain dynamically distinct and be selected by context. V25 asks how a mixed future consequence can find and rewrite the local causal route that produced it.

The fusion object is therefore not “a neuron running a GA.” It is a dynamical system with **separated computational continuations, context-dependent gain, local causal history, changing observational resolution, and slow operator rewrite**.

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

A router should change the `g_k` strongly with context while keeping leakage low enough that distinct approaches do not collapse into a lethal average.

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

Architecturally:

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

T2 shows why this is not cosmetic. If the memory simply decays in yesterday's coordinates, delayed reward can reinforce a different current mode even though the memory itself never vanished.

```math
\boxed{\text{credit identity must be covariant with computational identity}.}
```

## 5. Identifiability sets the legal resolution of credit

T2 still assumed that the continuation transform was known.

T3 removes that privilege.

Suppose the current observation operator has a nearly degenerate cluster. Individual vectors inside the cluster can be ill-conditioned even when the invariant subspace is stable. Then the currently justified object is a block

```math
\mathcal B(t)=\operatorname{span}\{E_i(t):i\in S\},
```

not the individual axes inside it.

This leads to the strongest constraint in ThirdWay so far:

```math
\boxed{\textbf{resolution of credit} \;\le\; \textbf{resolution of identifiable computation}.}
```

The rule has three cases.

### Individual identity identifiable

Use route-local credit normally.

### Only a block/subspace identifiable

Track and transport the block. Do not manufacture axis-level certainty from an ill-conditioned decomposition.

### Fine identity needed but genuinely unobservable

Acquire another intervention/measurement whose response separates the candidate continuations.

In T3's hidden-gauge attacker the primary operator is exactly invariant to a 90-degree semantic rotation inside a degenerate block. No primary-only algorithm can recover the individual axes. A second operator with a different spectrum over the same latent frame restores the distinction.

So active sensing is no longer generic uncertainty reduction. It has a precise write-side trigger:

```text
ask another question when the current observational resolution
is too coarse for the specificity of the proposed write.
```

## 6. Spectral local credit

V25 suggests replacing a single decay by resonant temporal states,

```math
z_{k,j}(t+1)=\rho_j e^{i\omega_j}z_{k,j}(t)+u_k(t).
```

The coordinates `(rho, omega)` describe causal lifetime and temporal carrier. A later scalar consequence can interact with local phase/history without transporting an explicit route label.

A future ThirdWay gate can let the pole itself move,

```math
(\rho_k(t),\omega_k(t))\rightarrow(\rho_k(t+1),\omega_k(t+1)),
```

while the computational road moves too.

This remains important, but T3 changes the ordering of priorities: before refining the temporal carrier, the system must know at what **computational resolution** a write is justified.

## 7. Two kinds of rewrite

The fusion requires keeping two operations distinct.

### Gain rewrite

```math
a_k(c)\rightarrow a_k'(c)
```

means **use this existing approach more or less in this situation**.

### Geometry/operator rewrite

```math
E_k(t)\rightarrow E_k(t+1)
```

or more generally

```math
L_{t+1}=L_t+\Delta L_t
```

means **change how future computations propagate**.

The working hypothesis is a timescale separation:

```math
\boxed{\text{fast: route among approaches}}
```

and

```math
\boxed{\text{slow: rewrite approaches/router while preserving continuation identity}.}
```

Without that separation, repeated success can simply purify one mode and erase the repertoire.

## 8. The next rewrite object should be low-rank

Kompressori measured a useful asymmetry:

```text
full response operator J      high-rank
experience-induced change ΔJ  much lower-rank
```

This suggests representing a route-local slow edit as

```math
\boxed{\Delta L_i=U_iM_iV_i^*}. 
```

The question then becomes not merely whether an edit is locally useful, but whether several useful edits are dynamically compatible.

CausalHorizon supplies the exact finite-event overlap object

```math
\boxed{\Omega_{ji}=V_j^*\Phi(\tau_j,\tau_i+1)U_i}. 
```

If `Omega_ji` is small/zero, one edit has little/no propagated access to the update coordinates of another. If it is large, two individually good changes can interact strongly.

That gives a candidate write guard:

```text
local success
    -> low-rank ΔL proposal
    -> estimate directed propagated overlap with retained highways
    -> compatible: commit separately
    -> colliding: coordinate / postpone / jointly evaluate
```

This is more specific than generic parameter-space orthogonality.

## 9. Failure modes

ThirdWay now distinguishes at least six different collapses.

**Extinction** — one computational family loses essentially all amplitude.

**Mixing** — multiple families remain populated but their internal computations blend into an invalid average.

**Geometry drift** — a family still exists, but moves so far that the old routing structure no longer tracks its continuation.

**Stale credit** — causal memory survives, but remains attached to old coordinates and therefore reinforces the wrong current computation.

**Over-specific credit** — the learner assigns an individual route identity when the available evidence supports only a coarser block.

**Update collision** — two locally useful slow edits interfere because their propagated response geometries overlap.

**Spectral aliasing** remains a seventh future failure: distinct causal histories become indistinguishable because the available temporal basis cannot resolve them.

## 10. Gate sequence

### T0 — local address, global consequence

Seed two distinct computations. Learn context-dependent modal gains from delayed scalar consequence. Destroy route address with pooled/shuffled controls.

Status: **implemented and passing**. Route-local late routing accuracy `0.952750`; pooled/current-only `0.512125`; route-shuffled `0.043000`.

### T1 — moving road

Rotate the computational basis through parameter coordinates and compare frozen-coordinate identity with continuation identity.

Status: **implemented and passing**. Fixed-coordinate identity `0.519271`; covariant continuation `1.000000`; exact cocycle residual below `3e-16`.

### T2 — delayed credit on a moving road

Store eligibility in physical coordinates while the basis rotates during the reward delay. Compare transported credit with stale-coordinate credit.

Status: **implemented and passing**. Covariant late routing `0.952750`; stale-coordinate routing `0.184500`; transported physical credit matches the ideal modal shadow to about `1.1e-15`.

### T3 — identifiability-aware credit

Infer continuation from a noisy moving operator. Near crossings, switch from individual-vector tracking to ambiguity blocks. Then attack that solution with an exact hidden gauge and allow a second operator only during ambiguity.

Status: **implemented**. In the crossing world, vector-overlap routing falls to `0.5115` while ambiguity blocks retain `0.9525`, matching the oracle. In the exact hidden-gauge world, block-only routing falls to `0.3884`; selective second-operator queries recover `0.9453` versus oracle `0.9525`.

T3 establishes the operational rule that fine credit requires fine identifiability.

### T4 — collision-aware operator rewrite

Give separate highways low-rank `Delta L` proposals and compare compatibility guards:

- naive addition;
- parameter-space orthogonality;
- static update-subspace overlap;
- directed propagated overlap `Omega_ji`;
- oracle joint finite evaluation.

Target: predict which locally useful edits can coexist without destroying old computational modes or future switching ability.

### T5 — active measurement for write resolution

Do not supply the diagnostic operator family. Give the system several possible probes with costs and let it choose the one expected to reduce *write-relevant* identity ambiguity.

This is the direct successor of V24/Child active sensing.

### T6 — resonant moving credit

Combine covariant identity with V25's adaptive temporal poles. Test whether route identity and temporal-credit identity can both move without aliasing.

### T7 — discovered computational approaches

Remove labelled direct/relational pathways. Evolve/train generic tiny networks on an ambiguous world. Cluster survivors only by interventions, Jacobians, hidden trajectories and ablations. Then test contextual modal gain, delayed local credit, ambiguity-aware sensing and slow compatible rewrite over the discovered moving families.

That is the first gate capable of supporting the strong claim:

```math
\boxed{\textbf{self-rewriting spectral router over discovered computational highways}.}
```

Until then, ThirdWay uses smaller names for smaller results.