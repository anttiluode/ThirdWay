# Gate T3 — credit at the resolution of identifiable computation

T3 asks a question that several older repos approached from different sides:

> **What should a learner do when delayed consequence arrives, but the identity of the computation that caused it is only partially observable?**

The answer tested here is:

```text
if individual computational routes are identifiable
    -> keep individual route identity and credit

if only a subspace/block is identifiable
    -> track the block rather than inventing unstable vector labels

if the ambiguity is real rather than merely noisy
    -> acquire another observation before making a fine-grained write
```

This is the first ThirdWay gate where the *granularity of credit assignment* is explicitly constrained by the *granularity of computational identifiability*.

## Part A — noisy crossing

Four moving spectral modes define a changing representation. The first two are the two useful computations from the GAx bridge: a direct cue and a relational cue. Their eigenvalues repeatedly cross while all four axes move in ambient coordinates.

The learner receives only a noisy symmetric operator estimate. It must maintain semantic route identity while delayed scalar reward trains a context-dependent router.

Two trackers are compared:

- `overlap`: ordinary one-vector-at-a-time nearest-overlap tracking;
- `blocks`: when the eigengap becomes too small, treat the ambiguous cluster as one identifiable subspace and carry the previous semantic basis through it with an orthogonal Procrustes lift.

Eight-seed battery, 5000 events per seed, reward delays 3–10 events:

| tracker | late route accuracy | late reward | task accuracy | late semantic fidelity |
|---|---:|---:|---:|---:|
| vector overlap | **0.5115** | 0.5027 | 0.7515 | 0.4565 |
| **ambiguity blocks** | **0.9525** | **0.9541** | **0.9774** | **0.9967** |
| oracle frame | 0.9525 | 0.9541 | 0.9774 | 1.0000 |

The ambiguity block is active for about 18.8% of the trajectory.

So the useful lesson is not merely “better eigenvector tracking.” It is:

```math
\boxed{\text{when the vector is not identifiable, stop making vector-level claims.}}
```

The delayed-credit learner can remain essentially oracle-level if route identity is carried at the coarsest currently identifiable resolution.

## Part B — exact hidden gauge attack

Part A is not enough. A near-degenerate noisy crossing can still contain enough continuity information to transport a semantic basis.

The attacker therefore creates an interval in which the first two eigenvalues are **exactly equal**. During that interval the hidden semantic axes rotate by 90 degrees *inside the degenerate subspace*.

The primary operator is exactly invariant to that internal rotation.

No primary-operator-only algorithm is entitled to know which individual axis became which.

The passive ambiguity-block tracker fails, as it should:

| tracker | late route accuracy | late reward | task accuracy | late semantic fidelity | extra query fraction |
|---|---:|---:|---:|---:|---:|
| overlap | 0.6804 | 0.6788 | 0.8396 | 0.5101 | 0 |
| ambiguity blocks only | **0.3884** | 0.3782 | 0.6893 | **0.0210** | 0 |
| **active operator family** | **0.9453** | **0.9419** | **0.9713** | **0.9992** | **0.2002** |
| oracle frame | 0.9525 | 0.9541 | 0.9774 | 1.0000 | 0 |

The active arm is allowed one extra diagnostic operator sharing the same hidden computational frame but with a nondegenerate signature. It queries that operator **only when the primary eigengap says individual identity is not supported**.

About 20% of time steps require the extra question, and the result returns close to the oracle.

This is deliberately not magic recovery from an impossible observation. It is the opposite:

> **When the current evidence cannot support a fine identity, either keep the update coarse or buy another observation that can.**

## Connection to the old repos

- **MovingProblem Gate 4**: local health can stay green while semantic orientation fails globally. T3 adds an operational response: do not equate local vector continuity with justified identity.
- **AlgoSchalgo HUNT0/HUNT3**: ambiguity blocks and operator families become part of the *credit-assignment rule*, not merely a tracking score.
- **GeometricNeuronV24**: active measurement is now invoked specifically when a write would otherwise exceed observable resolution.
- **Child Gate 6 / Gate 8 / planned Gate 9**: delayed audit, causal provenance, active sensing and representation drift meet in one executable object.
- **SighImageSuper**: a distinction that no available question can reveal is not an available memory for the learner. T3 refuses to fabricate credit beneath that observational boundary.

## New principle

The result suggests a compact design constraint:

```math
\boxed{\textbf{resolution of credit} \;\le\; \textbf{resolution of identifiable computation}.}
```

Or operationally:

```text
identify -> route -> act -> delayed consequence

if identity stayed resolvable:
    rewrite that route

if identity collapsed to a block:
    preserve the block / defer fine write

if a fine write matters:
    ask another question
```

This is stronger than T2's stale-credit result. T2 assumed the continuation transport was known. T3 makes the system confront the case where the continuation itself is uncertain or genuinely unobservable.

## Boundary

T3 still seeds the two computational mechanisms. It does not yet discover them from a generic neural population.

The second operator is also supplied. The next problem is to choose or learn *which intervention/measurement* best resolves an ambiguity, rather than being handed the diagnostic family.

That next step should reuse V24/Child active sensing rather than invent another router-specific heuristic.
