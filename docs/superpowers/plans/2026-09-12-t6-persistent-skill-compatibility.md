# T6 Persistent Skill Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and demonstrate a small continual-learning recurrent network that proposes persistent low-rank edits, predicts cross-skill damage before consolidation, and retains or rejects edits using propagated causal compatibility.

**Architecture:** One deterministic 24-state tanh RNN and shared readout solve three context-labelled temporal tasks. Candidate recurrent edits are generated from target-skill information only, then scored by static attackers and a directional propagated-causal predictor against already retained skills. The same consolidation interface powers both the scientific T6 battery and the final interactive/text demo, so the demo cannot bypass the tested mechanism.

**Tech Stack:** Python 3.11/3.12, NumPy, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-12-t6-persistent-skill-compatibility-design.md`

## Global Constraints

- Keep T6 pure NumPy unless a scientifically necessary dependency is discovered.
- Fixed recurrent hidden size: 24.
- One shared readout; task context may enter the input but does not select separate readout heads.
- Three temporal skills: recent cue, delayed cue, relational/order cue.
- Persistent candidate edits are rank-1 in the first implementation.
- Candidate generation may inspect only target-skill data and target-skill utility, never cross-skill compatibility.
- Holdout unit is candidate edit identity, not pair rows.
- Implement parameter cosine, gradient alignment, diagonal empirical-Fisher overlap, static Jacobian overlap, directional propagated-causal compatibility, and finite joint oracle evaluation.
- Freeze predictor scalar definitions and classification procedure before held-out scoring.
- Honest failure is a valid result; do not tune the world after seeing held-out results merely to cross the gate thresholds.
- Preserve all T0–T5 behavior and run T0–T6 in CI on Python 3.11 and 3.12.
- The final demo must call the same candidate-generation, scoring, and consolidation functions used by the gate.

---

## File Structure

- Create `experiments/gate_t6_persistent_skill_compatibility.py`: complete scientific world, model, candidate generation, predictors, battery, summary, and executable gate.
- Create `tests/test_gate_t6.py`: TDD scientific invariants and broad performance requirements.
- Create `demo_t6_continual_learning.py`: human-readable continual-learning demonstration using the production T6 interfaces.
- Create `T6_RESULTS.md`: exact measured result receipt and limitations.
- Modify `README.md`: add T6 result and demo instructions only after measurement.
- Modify `THEORY.md`: update the compatibility section and gate sequence only after measurement.
- Modify `.github/workflows/ci.yml`: execute T6 gate and demo smoke test after unit tests.

---

### Task 1: Deterministic Recurrent World and Three Temporal Skills

**Files:**
- Create: `experiments/gate_t6_persistent_skill_compatibility.py`
- Create: `tests/test_gate_t6.py`

**Interfaces:**
- Produces: `WorldConfig`, `Dataset`, `make_dataset(seed: int, examples_per_skill: int) -> Dataset`, `make_seed_model(seed: int) -> Model`, `evaluate_skill(model: Model, dataset: Dataset, skill: int) -> float`, `evaluate_all(model: Model, dataset: Dataset) -> np.ndarray`.
- `Model` stores `W: np.ndarray`, `B: np.ndarray`, `b: np.ndarray`, and one shared readout `C: np.ndarray`.

- [ ] **Step 1: Write the failing deterministic-world tests**

```python
from experiments.gate_t6_persistent_skill_compatibility import (
    evaluate_all,
    make_dataset,
    make_seed_model,
)


def test_t6_world_is_deterministic_and_shared():
    d1 = make_dataset(seed=17, examples_per_skill=48)
    d2 = make_dataset(seed=17, examples_per_skill=48)
    assert np.array_equal(d1.inputs, d2.inputs)
    assert np.array_equal(d1.targets, d2.targets)
    assert set(np.unique(d1.skills)) == {0, 1, 2}

    m = make_seed_model(seed=23)
    assert m.W.shape == (24, 24)
    assert m.C.ndim == 1
    assert m.C.shape == (24,)


def test_t6_seed_model_is_nontrivial_but_imperfect_on_every_skill():
    data = make_dataset(seed=31, examples_per_skill=128)
    score = evaluate_all(make_seed_model(seed=23), data)
    assert score.shape == (3,)
    assert np.all(score > 0.52)
    assert np.all(score < 0.95)
```

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_gate_t6.py -q`

Expected: FAIL because the T6 module / interfaces do not yet exist.

- [ ] **Step 3: Implement the minimal deterministic world**

Implement three binary sequence tasks with matched input distribution and context encoded as a small component of the same input stream:

```python
# skill 0: sign of recent cue at T-2
# skill 1: sign of delayed cue at T-8
# skill 2: +1 iff cue A appeared before cue B, else -1
```

Use sequence length 16 and input dimension 6. The fixed model recurrence is:

```python
h = np.tanh(model.W @ h + model.B @ x_t + model.b)
y = float(model.C @ h)
```

Fit the single shared readout `C` deterministically by ridge regression over hidden states from the seeded recurrent body so all three tasks are above chance but imperfect. If seed `23` does not satisfy the fixed baseline band before held-out T6 scoring exists, choose the first seed in `range(100)` satisfying the band and encode that deterministic search in `make_seed_model`; do not revisit it after predictor results exist.

- [ ] **Step 4: Run GREEN**

Run: `pytest tests/test_gate_t6.py -q`

Expected: both tests PASS.

- [ ] **Step 5: Commit**

Commit message: `T6: add shared recurrent continual-learning world`

---

### Task 2: Individually Useful Persistent Rank-1 Candidate Edits

**Files:**
- Modify: `experiments/gate_t6_persistent_skill_compatibility.py`
- Modify: `tests/test_gate_t6.py`

**Interfaces:**
- Produces: `CandidateEdit(skill: int, seed: int, delta_w: np.ndarray, base_loss: float, edited_loss: float)`, `generate_candidates(model: Model, dataset: Dataset, skill: int, candidate_seeds: Iterable[int]) -> list[CandidateEdit]`, `apply_edit(model: Model, edit: CandidateEdit) -> Model`.

- [ ] **Step 1: Write RED for the utility invariant and target-only generation**

```python
def test_generated_candidates_are_rank1_and_individually_useful():
    data = make_dataset(seed=41, examples_per_skill=128)
    model = make_seed_model(seed=23)
    edits = generate_candidates(model, data, skill=1, candidate_seeds=range(12))
    assert len(edits) >= 4
    for edit in edits:
        assert np.linalg.matrix_rank(edit.delta_w, tol=1e-9) == 1
        assert edit.edited_loss < edit.base_loss - 1e-4
```

Add a second test that regenerating skill-1 candidates after permuting labels of skills 0 and 2 leaves the candidate matrices unchanged; this proves candidate generation does not inspect non-target losses.

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_gate_t6.py -q`

Expected: FAIL because candidate APIs do not exist.

- [ ] **Step 3: Implement minimal candidate search**

For each candidate seed, draw deterministic unit vectors `u, v`, form `delta = alpha * outer(u, v)`, and line-search a fixed amplitude set:

```python
AMPLITUDES = (-0.30, -0.20, -0.12, -0.07, 0.07, 0.12, 0.20, 0.30)
```

Evaluate only the target skill and retain the best amplitude if it improves target loss by at least `1e-4`. Candidate generation receives a target-skill dataset view so it has no API access to other skill losses.

- [ ] **Step 4: Run GREEN**

Run: `pytest tests/test_gate_t6.py -q`

Expected: utility and target-only tests PASS.

- [ ] **Step 5: Commit**

Commit message: `T6: generate useful persistent low-rank edits`

---

### Task 3: Persistent Pair Battery and Finite Damage Oracle

**Files:**
- Modify: `experiments/gate_t6_persistent_skill_compatibility.py`
- Modify: `tests/test_gate_t6.py`

**Interfaces:**
- Produces: `PairCase`, `build_pair_battery(...) -> list[PairCase]`, `measure_persistent_damage(base: Model, retained: CandidateEdit, candidate: CandidateEdit, data: Dataset) -> Damage`, where `Damage` includes `old_skill_damage`, `leakage_delta`, and `switch_cost_delta`.

- [ ] **Step 1: Write RED for pair validity and identity-level split**

```python
def test_pair_battery_uses_disjoint_edit_identities_and_has_both_outcomes():
    battery = build_default_pair_battery()
    train_ids = {case.candidate_id for case in battery if case.split == 'train'}
    test_ids = {case.candidate_id for case in battery if case.split == 'test'}
    assert train_ids.isdisjoint(test_ids)
    held = [c for c in battery if c.split == 'test']
    collided = [c for c in held if c.old_skill_damage > COLLISION_THRESHOLD]
    safe = [c for c in held if c.old_skill_damage <= COLLISION_THRESHOLD]
    assert len(collided) >= 0.20 * len(held)
    assert len(safe) >= 0.20 * len(held)
```

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_gate_t6.py::test_pair_battery_uses_disjoint_edit_identities_and_has_both_outcomes -q`

Expected: FAIL because persistent pair measurement is absent.

- [ ] **Step 3: Implement finite persistent damage measurement**

For every ordered retained/new skill pair with `retained.skill != candidate.skill`:

```python
m_r = apply_edit(base, retained)
m_rc = apply_edit(m_r, candidate)
old_damage = loss(m_rc, retained_skill) - loss(m_r, retained_skill)
```

Compute hidden-trajectory Gram overlap before/after as leakage diagnostic. Compute switch cost on deterministic alternating blocks `0,1,2,0,2,1,...` by evaluating the first two examples after each context switch versus steady examples later in the block.

Use candidate seeds `0..15` for training identities and `16..31` for held-out identities. A retained edit may appear across many cases inside one split, but no candidate identity may cross splits.

Set the collision threshold once from a fixed absolute loss increase of `0.02`. If the pre-predictor battery violates the required 20/20 outcome mix, record T6 as structurally invalid rather than searching the threshold after seeing predictor scores.

- [ ] **Step 4: Run GREEN**

Run: `pytest tests/test_gate_t6.py -q`

Expected: pair split and nontrivial-outcome tests PASS.

- [ ] **Step 5: Commit**

Commit message: `T6: measure persistent cross-skill edit damage`

---

### Task 4: Static Attackers and Directional Propagated-Causal Predictor

**Files:**
- Modify: `experiments/gate_t6_persistent_skill_compatibility.py`
- Modify: `tests/test_gate_t6.py`

**Interfaces:**
- Produces: `PredictorScores`, `score_pair(case: PairCase, ...) -> PredictorScores`, `fit_threshold(train_scores, train_truth)`, `summarize_predictors(...) -> PredictorSummary`.

- [ ] **Step 1: Write RED for all predictor fields and directional controls**

```python
def test_t6_predictors_include_real_static_attackers_and_directional_controls():
    result = run()
    assert set(result.predictors) == {
        'parameter', 'gradient', 'fisher', 'static_jacobian',
        'causal', 'causal_shuffled', 'causal_reversed', 'oracle'
    }
    assert result.predictors['causal'].test_auroc >= 0.5
```

Add tests asserting the empirical Fisher is built from per-example output/loss sensitivities, not activation cosine; expose `fisher_diagonal(...)` directly and assert its shape is `(24,24)` after reshaping the diagonal.

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_gate_t6.py -q`

Expected: FAIL because predictor implementations are absent.

- [ ] **Step 3: Implement static attackers**

Implement:

```text
parameter: abs cosine(flatten(retained.delta_w), flatten(candidate.delta_w))
gradient: abs cosine(flatten(candidate.delta_w), flatten(grad_old_loss_at_W_R))
fisher: sum(F_old * candidate.delta_w**2), normalized by candidate norm
static_jacobian: cosine of average local recurrent-Jacobian changes on their own task trajectories
```

Compute the empirical Fisher diagonal from per-example derivatives of squared-error loss with respect to every `W[p,q]` using analytic recurrent forward sensitivities. Because hidden size is 24 and sequences are short, exact forward sensitivity is tractable and avoids autograd.

- [ ] **Step 4: Implement directional propagated score**

For a candidate edit `C`, measure its isolated hidden-state injection along candidate-task trajectories under `W_R`. Propagate each perturbation through the retained model's ordered recurrent Jacobians. Contract transported perturbations with retained-task output/loss susceptibility and sum normalized magnitudes:

```python
causal_score = mean_t_examples(abs(g_old_t @ delta_h_transport_t))
```

Build controls with the identical marginals:

```text
causal_shuffled: deterministically permute Jacobian time order
causal_reversed: evaluate retained-edit consequence entering candidate-task susceptibility
```

The oracle score is the directly measured old-skill damage and is never used by the practical guard.

- [ ] **Step 5: Run GREEN on interfaces**

Run: `pytest tests/test_gate_t6.py -q`

Expected: all interface/control tests PASS even if the final scientific gate may still fail.

- [ ] **Step 6: Commit**

Commit message: `T6: add causal compatibility radar and attackers`

---

### Task 5: Freeze Evaluation and Run the Scientific Gate

**Files:**
- Modify: `experiments/gate_t6_persistent_skill_compatibility.py`
- Modify: `tests/test_gate_t6.py`
- Create: `T6_RESULTS.md` only after measurements exist.

**Interfaces:**
- Produces: `run() -> Summary`, CLI `python -m experiments.gate_t6_persistent_skill_compatibility` returning exit code 0 only when the frozen gate criteria pass.

- [ ] **Step 1: Write RED for the approved gate criteria**

```python
def test_t6_frozen_scientific_gate():
    s = run()
    assert s.all_candidates_individually_useful
    assert s.heldout_collision_fraction >= 0.20
    assert s.heldout_safe_fraction >= 0.20
    assert s.predictors['causal'].damage_correlation > 0.70
    for name in ('parameter', 'gradient', 'fisher', 'static_jacobian'):
        assert s.predictors['causal'].test_auroc >= s.predictors[name].test_auroc + 0.10
    assert s.skill_pair_families_with_positive_causal_advantage >= 4
    assert s.predictors['oracle'].test_auroc >= s.predictors['causal'].test_auroc - 1e-12
    assert s.predictors['causal'].test_auroc >= s.predictors['causal_shuffled'].test_auroc + 0.10
    assert s.predictors['causal'].test_auroc >= s.predictors['causal_reversed'].test_auroc + 0.10
```

- [ ] **Step 2: Run RED / honest scientific evaluation**

Run: `pytest tests/test_gate_t6.py::test_t6_frozen_scientific_gate -q`

Expected: either PASS, or a scientifically meaningful FAIL. Do not alter thresholds/world based on this output merely to force PASS.

- [ ] **Step 3: If the gate fails, characterize without p-hacking**

Run the fixed strength sweep `0.5x, 1.0x, 1.5x` around the already chosen candidate amplitudes and report where causal prediction degrades. Do not replace the primary 1.0x result.

- [ ] **Step 4: Write `T6_RESULTS.md` from measured output**

Include exact train/test counts, useful-candidate counts, collision fractions, AUROCs, correlations, six ordered task-pair results, controls, strength sweep, leakage and switching diagnostics, and explicit failure boundary.

- [ ] **Step 5: Commit**

Commit message: `T6: record persistent compatibility result`

---

### Task 6: Continual-Learning Consolidation Policy and Demonstration

**Files:**
- Modify: `experiments/gate_t6_persistent_skill_compatibility.py`
- Modify: `tests/test_gate_t6.py`
- Create: `demo_t6_continual_learning.py`

**Interfaces:**
- Produces: `ConsolidationDecision`, `consider_candidate(model, retained_edits, candidate, calibration) -> ConsolidationDecision`, `continual_demo(seed: int = 0) -> DemoSummary`.

- [ ] **Step 1: Write RED for a guard that truly controls consolidation**

```python
def test_causal_guard_changes_what_is_persistently_retained():
    demo = continual_demo(seed=0)
    assert demo.candidates_considered >= 6
    assert demo.accepted + demo.rejected == demo.candidates_considered
    assert demo.final_old_skill_accuracy_guarded >= demo.final_old_skill_accuracy_accept_all
```

Also assert the demo calls `consider_candidate`; do not duplicate the scoring logic inside the demo.

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_gate_t6.py -q`

Expected: FAIL because consolidation policy/demo do not exist.

- [ ] **Step 3: Implement the minimal guard**

Use the training split only to calibrate a causal-risk threshold. For each new target-skill candidate:

```text
candidate individually useful? no -> discard
causal risk to every retained skill below threshold? yes -> permanently add delta_w
otherwise -> reject/defer and keep current W
```

Compare against `accept_all`, which commits every individually useful candidate in the same order.

- [ ] **Step 4: Implement human-readable demo output**

`python demo_t6_continual_learning.py` prints a compact event stream such as:

```text
BASE  A=.71 B=.68 C=.66
learn B candidate #18  target +.09  risk(A)=.012  ACCEPT
STATE A=.70 B=.77 C=.65
learn C candidate #22  target +.08  risk(A)=.084  REJECT
STATE A=.70 B=.77 C=.65
...
FINAL guarded     A=... B=... C=... switch=...
FINAL accept-all  A=... B=... C=... switch=...
```

The output values come from live evaluation, never hard-coded receipt values.

- [ ] **Step 5: Run GREEN**

Run: `pytest tests/test_gate_t6.py -q && python demo_t6_continual_learning.py`

Expected: tests PASS; demo visibly shows candidates being accepted/rejected and reports guarded versus accept-all retention.

- [ ] **Step 6: Commit**

Commit message: `T6: demonstrate guarded continual learning`

---

### Task 7: Repository Integration and Full Regression

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `README.md`
- Modify: `THEORY.md`

**Interfaces:**
- CI executes all gates T0–T6 plus a demo smoke run.

- [ ] **Step 1: Update CI**

Append:

```yaml
- run: python -m experiments.gate_t6_persistent_skill_compatibility
- run: python demo_t6_continual_learning.py
```

- [ ] **Step 2: Update README/THEORY from actual T6 result**

Do not write success language before Task 5 measurements exist. State whether T6 passed or failed, preserve exact limitations, and distinguish the scientific gate from the demonstration policy.

- [ ] **Step 3: Run full verification**

Run locally or via CI equivalent:

```bash
pytest -q
python -m experiments.gate_t0_fusion
python -m experiments.gate_t1_moving_road
python -m experiments.gate_t2_credit_on_moving_road
python -m experiments.gate_t3_identifiability_credit
python -m experiments.gate_t4_causal_collision
python -m experiments.gate_t5_nonlinear_collision_radar
python -m experiments.gate_t6_persistent_skill_compatibility
python demo_t6_continual_learning.py
```

Expected: zero test failures. T6 gate exit is determined by the frozen scientific criteria; if T6 scientifically fails, CI should preserve that fact only after the README/results explicitly mark the negative gate and the workflow is adjusted to run the experiment as a recorded negative result rather than disguise it as success.

- [ ] **Step 4: Verify GitHub Actions on Python 3.11 and 3.12**

Read both matrix jobs. Do not call the branch green until both complete successfully.

- [ ] **Step 5: Final commit**

Commit message: `T6: integrate persistent continual-learning gate and demo`
