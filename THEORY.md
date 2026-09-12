# ThirdWay theory — covariant computational highways

ThirdWay starts from two independently developed questions.

**GAx** asks how several computational approaches can remain dynamically distinct, how context changes their gain, and how an evolving operator changes the meaning of a “mode.”

**GeometricNeuronV25** asks how mixed delayed consequence can find and rewrite the local causal route that produced it, including adaptive eligibility lifetimes and resonant temporal modes.

The fusion object is not “a neuron running a GA.” It is:

> **a self-rewriting collection of identifiable, covariant computational highways.**

The central difficulty is not merely learning a good answer. It is preserving several useful ways of computing while representation, credit, routing and the operator itself change.

---

## 1. Computational identity is counterfactual and covariant

A computational mode is not merely a parameter cluster or output cluster.

Two systems can give the same answer while implementing different computations. A mode is distinguished by interventions, Jacobians, hidden trajectories, ablations or pathway usage.

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

A router should change `g_k` strongly with context while keeping leakage low enough that distinct approaches do not collapse into a lethal average.

---

## 2. Fast routing is modal gain, not necessarily an explicit expert index

Let `q_t` be positive amplitudes over available computational modes. Context `c_t` induces a positive gain operator

```math
D(c_t)=\operatorname{diag}(e^{a_1(c_t)},\ldots,e^{a_K(c_t)}).
```

A minimal routing step is projective normalization:

```math
\boxed{q_{t+1}=\frac{D(c_t)q_t}{\mathbf 1^\top D(c_t)q_t}.}
```

This is the early meaning of **spectral router** in ThirdWay: context changes relative growth rates and the dynamics perform the selection.

GAx supplies the evolutionary interpretation. Repeated selection purifies modes; preserving a repertoire means preventing every useful alternative from acquiring a strongly negative long-term growth rate.

---

## 3. Reward can stay scalar if causal address stays local

A routed computation acts. A later modulation `m_t` returns, but the modulation need not contain a route label.

Each route can keep an eligibility trace

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

T2 names the failure:

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

Operationally:

```text
individual route identifiable
    -> route-specific credit is legal

only a block identifiable
    -> preserve/transport the block

fine write required but fine identity unobservable
    -> acquire another measurement before writing
```

T3 connects MovingProblem, AlgoSchalgo, V24 and Child here. In an exact hidden-gauge interval the primary operator genuinely cannot reveal the internal semantic rotation; a selectively queried second operator restores the needed distinction.

---

## 6. Temporal credit can itself be a searchable spectrum

V25 replaces a single decay with damped resonant states

```math
z_{k,j}(t+1)=\rho_j e^{i\omega_j}z_{k,j}(t)+u_k(t).
```

The pair `(rho, omega)` describes causal lifetime and temporal carrier.

V25 F9 goes further: the pole itself can evolve toward hidden causal dynamics instead of merely selecting from a fixed bank.

ThirdWay has not yet fused that adaptive pole into persistent compatibility learning. The ordering remains:

```text
identify the computation
-> keep credit attached to its continuation
-> refine the temporal carrier
-> only then use the returned consequence to rewrite persistent structure
```

Otherwise one can build an exquisitely tuned temporal trace attached to the wrong computation.

---

## 7. Slow learning is an operator edit, not merely a scalar weight score

Kompressori measured an important asymmetry:

```text
full finite-time response operator J      high-rank
experience-induced change Delta J          much lower-rank
```

So a route-local slow rewrite can be modeled as

```math
\Delta L_i=U_iM_iV_i^*.
```

The question is not only whether edit `i` works by itself. It is whether several locally useful edits can coexist.

CausalHorizon gives the exact time-ordered linear interaction coordinate

```math
\boxed{\Omega_{ji}=V_j^*\Phi(\tau_j,\tau_i+1)U_i}.
```

This motivates a fourth stage in the evolutionary cartoon:

```text
variation
 -> local success
 -> compatibility selection
 -> retention
```

**Compatibility selection** asks whether the consequence of one change enters the operating coordinates of another retained computation.

T4 proves that distinction in an exact linear transport laboratory: raw parameter overlap has zero precision and recall while propagated causal overlap classifies every collision.

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

An earlier edit changes the arriving state; the finite nonlinear interaction is

```math
C_{i\to j}=\|m_j(x_j^{(i)})-m_j(x_j)\|.
```

The radar measures the isolated effect of `i`, propagates it through the unedited tangent trajectory,

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

Across 192 finite cases:

```text
correlation with finite collision
parameter cosine                 -0.411329
activation / Fisher-ish proxy    -0.411329
static Jacobian cosine           -0.410830
propagated causal susceptibility  0.994736
```

The causal radar reaches 1.000 held-out accuracy/precision/recall in that constructed transport world while the three static attackers have zero held-out recall.

T5 therefore establishes that a **single-edit first-order causal forecast can predict a finite nonlinear interaction on unseen locations**. It does not yet establish persistent continual learning.

---

## 9. T6: persistent edits reveal a narrower compatibility object

T6 moves from time-local edit events to permanent rank-1 changes in one shared 24-state `tanh` RNN with one shared readout.

A retained edit `R` is installed first:

```math
W_R=W_0+R.
```

Then candidate `C` becomes permanent:

```math
W_{R+C}=W_0+R+C.
```

The primary finite damage is

```math
D_{old}=L_{old}(W_{R+C})-L_{old}(W_R).
```

The practical directional predictor asks how the persistent candidate changes the retained task's trajectory under the already-edited model:

```math
\delta h_{t+1}=D_t\left(W_R\delta h_t+Ch_t\right),
```

where

```math
D_t=\operatorname{diag}(1-h_{t+1}^2).
```

The final state perturbation is contracted with retained-task loss susceptibility.

### What survives

Held-out AUROC:

```text
parameter cosine                    0.551136
empirical Fisher                    0.579983
static Jacobian overlap             0.542832
gradient alignment                  0.698427
directional dynamic susceptibility  0.849213
reversed direction                  0.402972
finite oracle                       1.000000
```

The directional score has `r=0.769216` with finite old-skill damage and beats every tested static attacker by the frozen `+0.10` margin at primary strength.

Direction matters strongly:

```text
candidate -> retained   0.849213 AUROC
retained -> candidate   0.402972 AUROC
```

### What fails

The predeclared claim that *fine temporal tangent order* explains the advantage fails:

```text
ordered causal   0.849213
shuffled order   0.858392
```

Ordered and shuffled scores correlate `0.990072`.

The reason is structural. A persistent matrix edit does not act once. It injects a perturbation again at every recurrent step. The last four injections contribute about `66.3%` of the first-order effect, and the final injection alone about `32.2%`.

So T6 changes the compatibility object. The supported quantity is not yet

```text
exact long causal path of the edit
```

but rather

```text
which retained computation is dynamically susceptible to this persistent edit
```

The supported principle is therefore narrower:

```math
\boxed{\textbf{compatibility is directional dynamic susceptibility, not static similarity}.}
```

Fine tangent ordering may matter in other regimes, but T6 does not establish it here.

---

## 10. Practical consolidation: a useful safety filter can still be a bad learner

The T6 demo uses the surviving directional-risk score before each permanent write. Both learners receive the same 12 target-only candidate matrices.

The corrected guard protects every non-target skill already represented in the baseline calibration set from the first new write onward:

```text
accept-all:
    every locally useful candidate becomes permanent

guarded:
    candidate must remain useful in current state
    AND dynamic risk to every protected non-target skill must stay below threshold
```

The threshold comes only from the frozen T6 training split. Held-out cross-skill damage is not consulted before commitment.

Final behavior:

```text
                 A      B      C      mean   minimum   old A/B mean
base           .859   .664   .633    .719    .633          -
guarded        .844   .656   .648    .716    .648        .750
accept-all     .664   .641   .672    .659    .641        .652
```

Mean carried-state switch penalty also falls:

```text
guarded     .00966
accept-all  .02105
```

But the corrected guard commits only **1 of 12** proposals. Nine are rejected by predicted dynamic risk; two become locally useless after the trajectories diverge.

This is the first ThirdWay experiment where compatibility prediction actually controls persistent neural-network learning, not merely labels collisions after the fact. It is also a clean failure of binary veto as a complete continual-learning strategy:

```text
rejecting dangerous writes preserves old computation
but rejecting almost every write produces protective stagnation
```

The guard finishes much safer than accept-all but does not improve aggregate held-out accuracy beyond the initial network.

So a continual learner cannot stop at accept/reject.

---

## 11. T7 hypothesis: compatible partial writes

Suppose a candidate `Delta W` is useful on the target but unsafe at full amplitude.

Instead of

```text
unsafe -> reject
```

try a predeclared descending scale bank

```math
W' = W + \alpha\Delta W,
\qquad \alpha\in\{1,.75,.5,.25,.125\}.
```

Choose the largest `alpha` satisfying both

```math
\Delta L_{target}(\alpha)<0
```

and

```math
s_{risk,k}(\alpha)\le\tau
\quad\forall\text{ protected }k.
```

This turns compatibility prediction into **step-size control** rather than merely a veto. The first experiment should not introduce a second search direction: keep the proposal direction fixed and ask whether amplitude alone recovers useful learning while preserving the T6 safety advantage.

The T7A comparison should keep the same frozen 12-event proposal stream:

```text
accept-all
reject-only guard       <- corrected T6
safe-step guard         <- T7A
finite oracle           <- upper bound only
```

The target quantity is **useful learning retained per unit compatibility damage**.

Only if scalar safe steps fail should T7B import the finite-change lesson from IttnasNoruen. A first-order preserving direction can leave a curved acceptable region at finite amplitude; then a compensating component may be required:

```math
\Delta W_{safe}=\alpha\Delta W+\beta C_{comp}.
```

The compensation direction should be measured against the endangered retained response, not chosen from parameter-space orthogonality alone.

This is the direct path from a collision radar to an actual continual-learning mechanism.

---

## 12. Two timescales of rewrite remain distinct

ThirdWay still separates:

### Gain rewrite

```math
a_k(c)\rightarrow a_k'(c)
```

meaning **use this existing approach more or less in this situation**.

### Geometry/operator rewrite

```math
L_{t+1}=L_t+\Delta L_t
```

meaning **change how the approach itself computes or propagates**.

The working architecture is

```text
fast: route among approaches
slow: rewrite approaches/operator under compatibility constraints
```

Without that separation, repeated success can pull all approaches together and return the system to purification collapse.

---

## 13. Failure modes

ThirdWay now distinguishes at least ten collapses.

**Extinction** — one computational family loses essentially all amplitude.

**Mixing** — multiple families remain but blend into an invalid average.

**Geometry drift** — a family survives but the router no longer recognizes its continuation.

**Stale credit** — memory survives in obsolete coordinates and reinforces the wrong current computation.

**Over-specific credit** — the learner writes at a finer route resolution than observation justifies.

**Update collision** — two locally effective persistent operator edits interfere.

**Spectral aliasing** — different causal histories collapse onto indistinguishable temporal modes.

**Purification collapse** — repeated success suppresses alternative computations until later contexts cannot recover them cheaply.

**Stale compatibility model** — a risk estimate remains tied to an earlier network state after accepted edits have changed the substrate.

**Protective stagnation** — the guard preserves retained skill by rejecting so many useful candidates that learning stops.

T6 demonstrates the last danger directly: a good veto mechanism is not yet a good continual learner.

---

## 14. Gate ladder

### T0 — local address, global consequence

**Implemented.** Route-local routing `0.952750`; pooled/current-only `0.512125`; route-shuffled `0.043000`.

### T1 — moving road

**Implemented.** Fixed-coordinate identity `0.519271`; continuation identity `1.000000`.

### T2 — delayed credit on a moving road

**Implemented.** Covariant routing `0.952750`; stale-coordinate routing `0.184500`.

### T3 — identifiability-aware credit

**Implemented.** Ambiguity blocks preserve `0.9525` routing through noisy crossings; selective secondary observation restores `0.9453` routing in an exact hidden-gauge attack.

### T4 — exact causal compatibility

**Implemented.** Propagated overlap classifies all exact linear collisions; static overlap has zero precision and recall.

### T5 — nonlinear collision radar

**Implemented.** Propagated causal susceptibility has `r=0.994736` with finite nonlinear collision strength and 1.000 held-out classification in the constructed transport world.

### T6 — persistent skill edits

**Implemented, primary scientific criterion FAILS.** Directional dynamic susceptibility reaches `0.849213` held-out AUROC and beats the tested static attackers, but shuffled tangent order reaches `0.858392`; exact transport order is therefore not shown necessary.

**Corrected practical demo is a safety result, not yet continual learning.** Protecting every baseline non-target skill gives old A/B mean `.750` versus `.652` for accept-all and a lower switch penalty, but the binary guard accepts only `1/12` proposals and leaves aggregate mean essentially at baseline.

### T7 — compatible partial writes

**Next.** Convert the T6 veto into a safe-step controller. Keep the direction fixed and shrink unsafe useful edits to the largest compatible fraction. Only if that fails should a second compensation direction be introduced.

### T8 — active sensing for write resolution

Let the system choose which costly intervention reduces *write-relevant* identity ambiguity rather than supplying the diagnostic family.

### T9 — resonant moving credit

Combine covariant computational identity with V25 adaptive temporal poles. Test whether computational identity and temporal-credit identity can both move without aliasing.

### T10 — discovered computational approaches

Remove the seeded decomposition. Evolve/train generic small networks, identify computational families by intervention/Jacobian/trajectory behavior, then test routing, delayed credit, ambiguity resolution and compatible persistent rewrite.

Only that final stage could support the strong target phrase:

```math
\boxed{\textbf{self-rewriting spectral router over discovered computational highways}.}
```

Until then, ThirdWay keeps smaller names for smaller results.
