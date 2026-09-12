# ThirdWay retrospective — what the older repos become now

This note is not a claim that the earlier projects were secretly one theory all along. They were separate experiments with different privileges and failure modes.

The useful question in September 2026 is narrower:

> **Now that GAx, V25 and ThirdWay have exposed covariant computational routes, delayed local credit and identifiability-aware routing, which old problems can be reformulated more cleanly and which mechanisms are worth carrying forward?**

The answer is encouraging because several formerly separate problems collapse onto one common object:

```text
computational distinctions
        ↓
which distinctions remain identifiable?
        ↓
which route should dominate now?
        ↓
which delayed consequence belongs to which continuation?
        ↓
what low-dimensional part of the operator should actually change?
        ↓
which alternative routes must remain alive for a different future question?
```

## 1. SighImageSuper — stop equating memory with state

Sigh established three carriers of history: lingering activity, travelling activity and changed material. It also established that a distinction can physically exist yet remain invisible to one question/readout.

ThirdWay should keep that definition and discard the temptation to call every persistent mode a memory.

A computational highway is useful only insofar as some legal future interaction can still distinguish it from alternatives.

The new translation is:

```math
\text{memory capacity}
\approx
\text{number/geometry of computational distinctions still recoverable under future questions}.
```

T3 adds the complementary write rule: if the present observation cannot distinguish two routes, the learner must not invent route-specific credit merely because it wants a specific answer.

So Sigh's old question

```text
which histories can a later observer recover?
```

becomes part of ThirdWay's learning rule:

```text
which histories are specific enough to justify a write?
```

## 2. MovingProblem — representation drift was not the real bug

MovingProblem froze at Gate 4 after showing that a locally healthy frame can return with the wrong oriented sign after a closed path. That result killed the naive idea that local eigengaps and overlap are sufficient to preserve semantic identity.

ThirdWay T1/T2/T3 move beyond that dead end by separating three questions:

```text
coordinates moved                 — harmless by itself
continuation identity moved       — track covariantly
identity became ambiguous         — reduce resolution / ask another question
```

The important change is that we no longer need one globally fixed coordinate frame.

We need a continuation object whose granularity changes with identifiability.

That makes the old frozen repo useful again without adding another Gate 5 there.

## 3. AlgoSchalgo — ambiguity blocks become a learning primitive

AlgoSchalgo found that individual eigenvectors become ill-conditioned before the invariant subspace does. Its adaptive ambiguity blocks therefore tracked the coarser identifiable object rather than pretending every axis remained observable.

ThirdWay T3 promotes that from an estimation trick to a **credit constraint**:

```math
\boxed{\text{resolution of credit} \le \text{resolution of identifiable computation}.}
```

This is a substantial conceptual upgrade.

The block is no longer merely how we report uncertainty. It determines which updates are legal.

AlgoSchalgo HUNT3 also found that ambiguity belongs to an operator family: degeneracy in one operator can be resolved by another operator sharing the same latent frame. ThirdWay converts that into active sensing: query the second operator only when the first cannot support the needed write.

## 4. Child — several gates collapse into one control loop

Child accumulated what looked like separate problems:

- Gate 2: preserve temporary detail until future relevance is known;
- Gate 4: learn what deserves long memory;
- Gate 5: a policy can make itself blind to better off-policy evidence;
- Gate 6: learn when an extra measurement is worth its cost under delayed audit;
- Gate 8: evidence without causal provenance can become confident nonsense;
- planned Gate 9: representation drift.

ThirdWay suggests these are not six independent modules.

They can be organized around one loop:

```text
maintain a repertoire of computational routes
        ↓
route under current evidence
        ↓
if route identity is too ambiguous for the intended write:
        acquire another view
        ↓
act
        ↓
keep causally addressed eligibility while consequence is delayed
        ↓
update only the identifiable continuation that evidence supports
        ↓
retain unresolved alternatives when the evidence cannot decide
```

This gives Child Gate 9 a concrete successor rather than another standalone representation-drift toy.

## 5. GeometricNeuronV24 — active sensing gets a precise trigger

V24 showed that address choice changes observability and that a write can alter future sensing cost. Its next open problem was learning what deserves to become persistent state.

ThirdWay supplies a sharper trigger for active sensing:

> **Measure when the resolution of current evidence is lower than the resolution required by the proposed write.**

That is better than generic uncertainty sampling.

The system should not ask merely because confidence is low. It should ask when an ambiguity would cause two materially different updates.

This suggests a future measurement value:

```math
\text{value of probe}
\approx
\text{expected reduction in harmful credit ambiguity}
-
\text{probe cost}.
```

That is a stronger successor to both V24's information-gain lens and Child's learned sensing threshold.

## 6. IttnasNoruen — preservation should follow computational continuations, not a bag of outputs

IttnasNoruen exposed two limits of ordinary replay/guarding:

1. a finite reference bank can preserve every stored answer while unseen access routes deteriorate;
2. an infinitesimal/tangent preserving direction can still violate the finite nonlinear constraint after an actual step.

ThirdWay does not solve continual learning yet, but it suggests a better protected object.

Instead of guarding only

```text
old input -> old output
```

protect a small set of **counterfactual computational signatures** for each surviving highway:

```text
intervention / Jacobian / hidden trajectory / route usage
        ↓
computational identity signature
```

Then a proposed update is dangerous if it:

- extinguishes a highway,
- merges two previously distinct signatures,
- makes the old signature inaccessible under its old questions,
- or moves the highway beyond the router's ability to continue tracking it.

Finite candidate evaluation remains useful, but the reference set should be organized by route identity rather than by arbitrary historical examples.

## 7. Kompressori — do not rewrite the whole operator

Kompressori produced one of the most useful facts for the next ThirdWay stage:

```text
full response operator J          high-rank
experience-induced change ΔJ      low-rank
```

and then:

```text
far-separated ΔJ updates          almost exactly additive
near / overlapping updates        strongly interfere
```

This suggests that ThirdWay's slow rewrite should not begin as a dense new neural training rule.

Use a low-rank update object attached to a computational highway:

```math
L_{t+1}=L_t + U_k M_k V_k^*.
```

Then preserve multiple approaches by keeping their update subspaces separated when possible.

The old "lethal average" problem becomes an operator-update collision problem rather than a metaphor.

## 8. CausalHorizon — replace distance with directed causal overlap

CausalHorizon gives the exact linear finite-event object

```math
\Omega_{ji}=V_j^*\Phi(\tau_j,\tau_i+1)U_i.
```

That is almost exactly what ThirdWay needs after Kompressori.

Two low-rank edits should be independently writable when their *propagated causal overlap* is small, not merely when their parameter vectors are orthogonal or spatial locations are distant.

This suggests a future ThirdWay write guard:

```text
candidate update for highway i
candidate update for highway j
        ↓
estimate directed causal overlap Ω_ji
        ↓
small overlap  -> allow separate writes
large overlap  -> coordinate / postpone / allocate a new subspace
```

This is a much sharper mechanism than generic orthogonal-gradient projection.

## 9. Operaattori / OperaattoriJako — eventually make the highway physical

Operaattori established a clean compiler:

```text
intrinsic structure -> transport operator -> local nonlinear closure -> response
```

and showed that the derivative of response with respect to geometry depends strongly on operating state; transport and feedback can even reverse signs.

That means ThirdWay should not assume a computational highway is a static abstract vector forever.

A later physical version can let slow low-rank/operator updates be generated by structural variables whose effect is compiled through a substrate.

The important inheritance is not "dendrites are spectral routers." It is:

> **structure can be the persistent variable that compiles the operator through which future signals travel.**

ThirdWay currently works one abstraction above that level.

## 10. BlackBoxLab — specialization without a diversity mechanism becomes monoculture or damage

BlackBoxLab showed two useful warnings:

- identical learners can diverge into specialists through different sampling histories;
- naive shared depletion can spread them out while destroying predictive quality.

GAx now gives this a better vocabulary.

A population can contain multiple computational modes without requiring explicit named roles, but repeated selection can purify the repertoire down to one approach.

So the future diversity mechanism should not be "make occupied regions unpleasant." It should preserve **useful counterfactual computational alternatives**.

ThirdWay can eventually replace the crude refractory field with persistent local operator edits whose gains are context-dependent and whose interference is monitored causally.

## 11. What can now be retired

Several recurring questions no longer deserve standalone repos unless a new empirical boundary appears:

- **"Can history matter?"** — yes; the carrier and readout geometry are the real questions.
- **"Can representations move?"** — yes; continuation and identifiability are the real questions.
- **"Can delayed reward work?"** — yes in addressed toys; the real question is preserving/inferring the causal address.
- **"Can active sensing help?"** — yes; the useful question is when extra evidence changes which write is justified.
- **"Can multiple specialists exist?"** — yes; the hard question is preventing extinction/mixing while allowing them to evolve.
- **"Can a local event change the operator?"** — yes; the hard question is how multiple low-rank changes compose and interfere.

Those are useful closures. They let the next repo work begin higher up.

# The new object

The surviving architecture is now closer to:

```text
             CURRENT CONTEXT
                    |
                    v
        observable operator family
                    |
                    v
        identifiable route/subspace
           /                  \
          /                    \
 enough evidence?          insufficient
       |                        |
       |                        v
       |                 active intervention
       |                        |
       +-----------<------------+
                    |
                    v
       context-dependent modal gain
                    |
                    v
          computation / action
                    |
                    v
       delayed global consequence
                    x
       covariant local eligibility
                    |
                    v
       low-rank route-local ΔL proposal
                    |
                    v
     causal-overlap / collision guard
                    |
                    v
          slow operator rewrite
                    |
                    +----> changes future routing
```

This is materially better than the old "neuron runs a genetic algorithm" picture.

The genetic/evolutionary principle survives at a higher level:

```text
maintain alternatives
      -> expose them to context
      -> amplify some
      -> assign consequence to identifiable continuations
      -> retain useful changes
      -> keep enough spectral reserve that another context can still select differently
```

# The next new gate

After T3, the next high-value experiment should combine **Kompressori + CausalHorizon + ThirdWay** rather than adding another frequency toy.

Call it provisionally **T4 — collision-aware operator rewrite**.

Give two computational highways separate low-rank operator edits. Let context and delayed credit propose writes to each. Compare:

1. naive addition of all successful edits;
2. parameter-space orthogonalization;
3. static subspace-overlap guard;
4. **directed propagated causal-overlap guard** using an estimated `Omega_ji`;
5. oracle joint finite evaluation.

Then measure:

```text
old-route preservation
new-route progress
cross-mode leakage
switching cost
number of retained alternatives
```

If causal-overlap gating predicts when two locally useful edits can coexist better than ordinary geometric orthogonality, we will have moved beyond several older repos at once.

That would be new territory for this line: not merely finding routes, remembering them, or assigning them credit, but **learning when two pieces of learning are dynamically compatible before committing either one**.
