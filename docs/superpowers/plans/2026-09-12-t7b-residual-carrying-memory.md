# T7B Residual-Carrying Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and falsify a signed hidden-trajectory residual memory that carries collateral route drift across the frozen T7A continual-learning sequence.

**Architecture:** Reuse the existing T6 recurrent model, T6 reject-only guard, and T7A proposal/scale helpers. T7B measures a finite signed hidden-trajectory residual on 48 calibration examples per skill, derives one untuned residual budget from the already-frozen reject-only baseline, and compares a signed vector controller against a matched scalar cumulative-debt attacker on the same 12 proposal matrices.

**Tech Stack:** Python 3.11/3.12, NumPy, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-12-t7b-residual-carrying-memory-design.md`

## Global Constraints

- Reuse seed model `make_seed_model(seed=23)` and the existing T7A proposal/calibration/evaluation seeds.
- Reuse `DEMO_SCHEDULE = (0, 1, 2, 1, 0, 2) * 2` and the accept-all-anchored target-only candidate generator.
- Reuse `SCALE_BANK = (1.0, 0.75, 0.5, 0.25, 0.125)` and `MIN_TARGET_IMPROVEMENT = 1e-4`.
- Use exactly `48` calibration examples per skill for hidden-trajectory residual signatures.
- A committed candidate must remain exactly `alpha * candidate.delta_w`; no rotation, compensation, regeneration, or new direction is allowed.
- The common residual budget is derived only from the frozen T6 reject-only reference replay, never tuned from T7B outcomes.
- Freeze positive or negative T7B outcome as a regression receipt; do not relax success margins after observing results.

---

### Task 1: Hidden-Trajectory Residual Primitive and Untuned Budget

**Files:**
- Create: `experiments/t7b_residual_carrying_memory.py`
- Create: `tests/test_gate_t7b.py`

**Interfaces:**
- Consumes: `batch_hidden_trajectories`, `apply_edit`, `make_dataset`, `make_seed_model`, `CandidateEdit`, `consider_candidate`, `_next_useful_proposal`, T7A `_scaled_candidate` and frozen constants.
- Produces:
  - `RESIDUAL_EXAMPLES: int = 48`
  - `route_signature(model, calibration, skill, examples=RESIDUAL_EXAMPLES) -> np.ndarray`
  - `residual_rms(vector: np.ndarray) -> float`
  - `build_frozen_candidate_stream(seed: int = 0) -> tuple[CandidateEdit, ...]`
  - `reject_only_residual_budget(seed: int = 0) -> ResidualBudgetReference`
  - dataclass `ResidualBudgetReference(budget: float, candidate_seeds: tuple[int, ...], accepted: int, peak_by_skill: tuple[float, float, float])`

- [ ] **Step 1: Write the failing primitive tests**

```python
import numpy as np

from experiments.gate_t6_persistent_skill_compatibility import (
    apply_edit,
    make_dataset,
    make_seed_model,
)
from experiments.t7_compatible_partial_writes import t7a_demo
from experiments.t7b_residual_carrying_memory import (
    RESIDUAL_EXAMPLES,
    build_frozen_candidate_stream,
    reject_only_residual_budget,
    residual_rms,
    route_signature,
)


def test_t7b_route_residual_is_signed_and_telescopes_exactly():
    calibration = make_dataset(seed=103, examples_per_skill=128)
    base = make_seed_model(seed=23)
    stream = build_frozen_candidate_stream(seed=0)
    first = stream[0]
    second = stream[1]

    after_first = apply_edit(base, first)
    after_second = apply_edit(after_first, second)

    h0 = route_signature(base, calibration, skill=2)
    h1 = route_signature(after_first, calibration, skill=2)
    h2 = route_signature(after_second, calibration, skill=2)

    r1 = h1 - h0
    r2 = h2 - h1
    assert RESIDUAL_EXAMPLES == 48
    assert np.allclose(r1 + r2, h2 - h0, atol=1e-12, rtol=1e-12)
    assert residual_rms(r1) > 0.0


def test_t7b_stream_and_budget_are_frozen_from_existing_baseline():
    stream = build_frozen_candidate_stream(seed=0)
    t7a = t7a_demo(seed=0)
    reference = reject_only_residual_budget(seed=0)

    assert tuple(edit.seed for edit in stream) == t7a.candidate_seeds
    assert len(stream) == 12
    assert reference.candidate_seeds == t7a.candidate_seeds
    assert reference.accepted == t7a.reject_only_accepted == 1
    assert reference.budget > 1e-12
    assert reference.budget == max(reference.peak_by_skill)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
pytest tests/test_gate_t7b.py::test_t7b_route_residual_is_signed_and_telescopes_exactly \
       tests/test_gate_t7b.py::test_t7b_stream_and_budget_are_frozen_from_existing_baseline -v
```

Expected: import/module failure because `experiments.t7b_residual_carrying_memory` does not exist yet.

- [ ] **Step 3: Add a nonfunctional module shell if needed and rerun until RED is behavioral**

Create the exported names with deliberately wrong placeholder behavior only long enough for pytest to reach assertions, for example:

```python
RESIDUAL_EXAMPLES = 48


def residual_rms(vector):
    return 0.0
```

Rerun the focused tests. Expected: assertion failure such as `assert residual_rms(r1) > 0.0`, not an import error.

- [ ] **Step 4: Implement exact finite route signatures and candidate stream**

```python
RESIDUAL_EXAMPLES = 48


def route_signature(model, calibration, skill, examples=RESIDUAL_EXAMPLES):
    subset = calibration.for_skill(int(skill))
    n = min(int(examples), subset.targets.size)
    states = batch_hidden_trajectories(model, subset.inputs[:n])[:, 1:, :]
    return np.asarray(states, dtype=float).reshape(-1)


def residual_rms(vector):
    x = np.asarray(vector, dtype=float).reshape(-1)
    if x.size == 0:
        raise ValueError("residual vector must be non-empty")
    return float(np.linalg.norm(x) / np.sqrt(x.size))


def build_frozen_candidate_stream(seed=0):
    proposal = make_dataset(
        seed=DEMO_PROPOSAL_SEED + int(seed),
        examples_per_skill=DEMO_EXAMPLES_PER_SKILL,
    )
    anchor = make_seed_model(seed=23)
    next_seed = DEMO_FIRST_CANDIDATE_SEED + 1_000 * int(seed)
    edits = []
    for skill in DEMO_SCHEDULE:
        candidate, next_seed = _next_useful_proposal(
            anchor, proposal, int(skill), next_seed
        )
        edits.append(candidate)
        anchor = apply_edit(anchor, candidate)
    return tuple(edits)
```

- [ ] **Step 5: Implement the reject-only residual-envelope replay**

Use the existing `consider_candidate` and `calibrated_risk_threshold`. Maintain one route anchor per skill and one signed residual vector per skill. When reject-only accepts a candidate targeting `s`, update non-target residuals with exact finite signature differences, then move the target anchor to its new signature and reset its residual to zeros. Track peak RMS residual by skill and return `budget = max(peak_by_skill)`.

The concrete core should follow:

```python
@dataclass(frozen=True)
class ResidualBudgetReference:
    budget: float
    candidate_seeds: tuple[int, ...]
    accepted: int
    peak_by_skill: tuple[float, float, float]


def _initial_route_state(model, calibration):
    anchors = {k: route_signature(model, calibration, k) for k in range(3)}
    residuals = {k: np.zeros_like(anchors[k]) for k in range(3)}
    return anchors, residuals
```

For each accepted reject-only edit, compute `before_signature[k]`, `after_signature[k]`, and `step = after - before`; carry `residuals[k] += step` for `k != target`, then re-anchor target and zero its residual.

- [ ] **Step 6: Run focused tests and full existing T7A tests**

Run:

```bash
pytest tests/test_gate_t7b.py -v
pytest tests/test_gate_t7.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit Task 1**

Commit message:

```text
T7B: add signed route residual primitive
```

---

### Task 2: Scalar Debt Attacker and Signed Residual Controller

**Files:**
- Modify: `experiments/t7b_residual_carrying_memory.py`
- Modify: `tests/test_gate_t7b.py`

**Interfaces:**
- Consumes: Task 1 route signatures, budget, frozen candidate stream, T7A `_scaled_candidate` and scale bank.
- Produces:
  - `ResidualTrial`
  - `ResidualDecision`
  - `RepairEvent`
  - `consider_scalar_debt_candidate(...) -> ResidualDecision`
  - `consider_residual_candidate(...) -> ResidualDecision`

Dataclasses:

```python
@dataclass(frozen=True)
class ResidualTrial:
    scale: float
    target_improvement: float
    step_rms_by_skill: dict[int, float]
    resultant_rms_by_skill: dict[int, float]


@dataclass(frozen=True)
class ResidualDecision:
    accepted: bool
    reason: str
    scale: float
    target_improvement: float
    scaled_candidate: CandidateEdit
    step_vectors_by_skill: dict[int, np.ndarray]
    step_rms_by_skill: dict[int, float]
    resultant_rms_by_skill: dict[int, float]
    trials: tuple[ResidualTrial, ...]
```

- [ ] **Step 1: Write failing controller tests**

```python
from experiments.t7_compatible_partial_writes import SCALE_BANK
from experiments.t7b_residual_carrying_memory import (
    consider_residual_candidate,
    consider_scalar_debt_candidate,
    reject_only_residual_budget,
    route_signature,
)


def test_t7b_controller_never_rotates_candidate_and_respects_budget():
    proposal = make_dataset(seed=101, examples_per_skill=128)
    calibration = make_dataset(seed=103, examples_per_skill=128)
    model = make_seed_model(seed=23)
    stream = build_frozen_candidate_stream(seed=0)
    candidate = stream[0]
    reference = reject_only_residual_budget(seed=0)
    anchors = {k: route_signature(model, calibration, k) for k in range(3)}
    residuals = {k: np.zeros_like(anchors[k]) for k in range(3)}

    decision = consider_residual_candidate(
        model,
        candidate=candidate,
        target_data=proposal,
        calibration=calibration,
        residuals=residuals,
        budget=reference.budget,
    )

    assert decision.scale in (0.0,) + SCALE_BANK
    if decision.accepted:
        assert np.allclose(
            decision.scaled_candidate.delta_w,
            decision.scale * candidate.delta_w,
        )
        assert max(decision.resultant_rms_by_skill.values()) <= reference.budget + 1e-12


def test_scalar_debt_never_decreases_but_signed_resultant_can():
    # Replay decisions through the full demo helper added in Task 3; this test
    # starts RED against the shell and becomes behavioral once the demo exists.
    summary = t7b_demo(seed=0)
    assert all(after >= before - 1e-15 for before, after in summary.scalar_debt_transitions)
    assert summary.repair_events >= 1
```

- [ ] **Step 2: Run focused tests to verify RED**

Run:

```bash
pytest tests/test_gate_t7b.py -v
```

Expected: missing controller/demo symbols or deliberate behavioral failure.

- [ ] **Step 3: Implement shared scale trial measurement**

For each scale in descending `SCALE_BANK`:

```python
scaled = _scaled_candidate(model, candidate, target_data, scale)
trial_model = apply_edit(model, scaled)
protected = tuple(k for k in range(3) if k != int(candidate.skill))
step_vectors = {}
step_rms = {}
for k in protected:
    before = route_signature(model, calibration, k)
    after = route_signature(trial_model, calibration, k)
    step = after - before
    step_vectors[k] = step
    step_rms[k] = residual_rms(step)
```

Do not inspect evaluation data.

- [ ] **Step 4: Implement scalar cumulative-debt policy**

Accept the largest scale with target improvement `> 1e-4` and

```python
all(float(debts[k]) + step_rms[k] <= budget + 1e-12 for k in protected)
```

Return resultant scalar values in `resultant_rms_by_skill` as `debts[k] + step_rms[k]`. On rejection, return a zero-scaled candidate exactly as T7A does.

- [ ] **Step 5: Implement signed residual policy**

Accept the largest scale with target improvement `> 1e-4` and

```python
resultant = residuals[k] + step_vectors[k]
residual_rms(resultant) <= budget + 1e-12
```

for every protected skill. Preserve signed `step_vectors_by_skill` in the decision for commit-time state updates and repair-event accounting.

- [ ] **Step 6: Run focused tests**

Run:

```bash
pytest tests/test_gate_t7b.py -v
```

Expected: controller primitive tests PASS; the full-demo test may remain RED until Task 3.

- [ ] **Step 7: Commit Task 2**

Commit message:

```text
T7B: add scalar and signed residual controllers
```

---

### Task 3: Frozen Five-Learner Demo and Scientific Gate

**Files:**
- Modify: `experiments/t7b_residual_carrying_memory.py`
- Modify: `tests/test_gate_t7b.py`

**Interfaces:**
- Consumes: Tasks 1-2 and existing `t7a_demo`.
- Produces:
  - `T7BEvent`
  - `T7BSummary`
  - `t7b_demo(seed: int = 0) -> T7BSummary`
  - `_passes_t7b(summary: T7BSummary) -> bool`

`T7BSummary` must expose at minimum:

```python
candidate_seeds: tuple[int, ...]
base_accuracy: tuple[float, float, float]
reject_only_accuracy: tuple[float, float, float]
t7a_accuracy: tuple[float, float, float]
scalar_debt_accuracy: tuple[float, float, float]
residual_accuracy: tuple[float, float, float]
finite_control_accuracy: tuple[float, float, float]
reject_only_accepted: int
scalar_debt_nonzero_writes: int
residual_nonzero_writes: int
residual_scales: tuple[float, ...]
scalar_debt_scales: tuple[float, ...]
residual_budget: float
final_residual_rms: tuple[float, float, float]
final_scalar_debt: tuple[float, float, float]
repair_events: int
repair_event_records: tuple[RepairEvent, ...]
residual_old_skill_accuracy: float
scalar_debt_old_skill_accuracy: float
reject_only_old_skill_accuracy: float
residual_mean_accuracy: float
scalar_debt_mean_accuracy: float
reject_only_mean_accuracy: float
base_mean_accuracy: float
residual_switch_penalty: float
reject_only_switch_penalty: float
scalar_debt_switch_penalty: float
scalar_debt_transitions: tuple[tuple[float, float], ...]
all_direction_pure: bool
all_budget_respected: bool
```

- [ ] **Step 1: Write the frozen full-gate test before implementation**

```python
from experiments.t7b_residual_carrying_memory import _passes_t7b, t7b_demo


def test_t7b_frozen_gate_signed_route_memory_stops_sequence_drift():
    s = t7b_demo(seed=0)

    assert len(s.candidate_seeds) == 12
    assert len(set(s.candidate_seeds)) == 12
    assert s.all_direction_pure
    assert s.all_budget_respected
    assert s.residual_nonzero_writes >= 3
    assert s.residual_old_skill_accuracy >= s.reject_only_old_skill_accuracy - 0.02
    assert s.residual_switch_penalty <= s.reject_only_switch_penalty + 0.005
    assert s.residual_mean_accuracy >= s.reject_only_mean_accuracy + 0.01
    assert s.residual_mean_accuracy >= s.base_mean_accuracy + 0.005
    assert s.residual_nonzero_writes >= s.scalar_debt_nonzero_writes + 2
    assert s.residual_old_skill_accuracy >= s.scalar_debt_old_skill_accuracy - 0.01
    assert s.residual_mean_accuracy >= s.scalar_debt_mean_accuracy + 0.005
    assert s.repair_events >= 2
    assert _passes_t7b(s)
```

- [ ] **Step 2: Run only the full-gate test and verify RED on the scientific criterion**

Run:

```bash
pytest tests/test_gate_t7b.py::test_t7b_frozen_gate_signed_route_memory_stops_sequence_drift -v
```

If the first failure is import/plumbing, add only the minimum shell needed so the test reaches numerical assertions. Do not change any frozen threshold or margin.

- [ ] **Step 3: Implement the five-learner replay**

Reuse `t7a_demo(seed=0)` for base/T6/T7A/finite-control reference metrics and independently replay the same candidate stream for the two new cumulative controllers.

For signed residual state:

```python
anchors, residuals = _initial_route_state(residual_model, calibration)
```

After an accepted candidate targeting `s`:

```python
for k, step in decision.step_vectors_by_skill.items():
    before_norm = residual_rms(residuals[k])
    residuals[k] = residuals[k] + step
    after_norm = residual_rms(residuals[k])
    if after_norm < before_norm - 1e-8:
        record_repair(...)
residual_model = apply_edit(residual_model, decision.scaled_candidate)
anchors[s] = route_signature(residual_model, calibration, s)
residuals[s] = np.zeros_like(anchors[s])
```

For scalar debt, add step RMS for non-target skills and reset the target skill debt to `0.0` after an accepted target update.

- [ ] **Step 4: Implement `_passes_t7b` exactly from the frozen spec**

```python
def _passes_t7b(s):
    return bool(
        len(s.candidate_seeds) == 12
        and s.all_direction_pure
        and s.all_budget_respected
        and s.residual_nonzero_writes >= 3
        and s.residual_old_skill_accuracy >= s.reject_only_old_skill_accuracy - 0.02
        and s.residual_switch_penalty <= s.reject_only_switch_penalty + 0.005
        and s.residual_mean_accuracy >= s.reject_only_mean_accuracy + 0.01
        and s.residual_mean_accuracy >= s.base_mean_accuracy + 0.005
        and s.residual_nonzero_writes >= s.scalar_debt_nonzero_writes + 2
        and s.residual_old_skill_accuracy >= s.scalar_debt_old_skill_accuracy - 0.01
        and s.residual_mean_accuracy >= s.scalar_debt_mean_accuracy + 0.005
        and s.repair_events >= 2
    )
```

- [ ] **Step 5: Run the frozen gate once**

Run:

```bash
pytest tests/test_gate_t7b.py -v
python -m experiments.t7b_residual_carrying_memory
```

Interpretation rule:

- If PASS: freeze the positive result immediately.
- If FAIL: inspect diagnostics only to identify which frozen criterion failed; do not change budget, margins, candidate stream, scale bank, or direction set. Convert the scientific test into an explicit negative-result regression receipt that reproduces the observed failure.

- [ ] **Step 6: Run the complete local test suite available through CI**

Run through GitHub Actions after committing; the permanent workflow must execute pytest and all gates T0-T7B on both Python versions.

- [ ] **Step 7: Commit Task 3**

Commit message for a positive result:

```text
T7B: carry signed route residual across writes
```

Commit message for a negative result:

```text
T7B: record residual-memory gate failure
```

---

### Task 4: Result Receipt, README, CI, Review, and Merge

**Files:**
- Create: `T7B_RESULTS.md`
- Modify: `README.md`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: frozen `T7BSummary` and `_passes_t7b` outcome.
- Produces: permanent scientific receipt and CI reproduction on Python 3.11/3.12.

- [ ] **Step 1: Write `T7B_RESULTS.md` from the frozen run**

Include exact:

- residual budget;
- candidate seeds;
- selected scalar-debt and signed-residual scales;
- write counts;
- A/B/C, mean, old A/B, switch penalty for all five learners;
- final signed residual RMS and scalar debts;
- repair-event count and records;
- each frozen success criterion with PASS/FAIL;
- claim boundary explaining that finite trajectory measurement is used directly and T7B does not establish a deployable predictor of the residual.

- [ ] **Step 2: Update README gate ladder**

State T7A as the existing negative and T7B as either:

```text
T7B — signed residual memory: PASS
```

or

```text
T7B — signed residual memory: FAIL
```

with the exact measured reason. Do not use stronger wording than the spec allows.

- [ ] **Step 3: Extend permanent CI**

Append after T7A verification:

```yaml
      - name: Verify T7B frozen result
        run: |
          python - <<'PY'
          from experiments.t7b_residual_carrying_memory import _passes_t7b, t7b_demo
          summary = t7b_demo(seed=0)
          EXPECTED_PASS = True  # set once from the first frozen run; False if negative
          assert _passes_t7b(summary) is EXPECTED_PASS
          print("T7B frozen result reproduced", EXPECTED_PASS)
          PY
      - run: python -m experiments.t7b_residual_carrying_memory
```

Set `EXPECTED_PASS` exactly once from the first completed frozen run; this is a receipt, not a knob.

- [ ] **Step 4: Verify exact branch head on Python 3.11 and 3.12**

Require GitHub Actions to complete:

- pytest;
- T0-T5 executable gates;
- T6 negative receipt and practical demo;
- T7A negative receipt and executable comparison;
- T7B frozen result receipt and executable comparison.

- [ ] **Step 5: Review the branch diff against the spec**

Check specifically:

- no candidate direction rotation;
- no T7B-dependent budget tuning;
- no evaluation-set access during decisions;
- target anchors reset only after accepted target writes;
- scalar debt and signed residual use identical candidate stream and budget;
- finite per-step T7A control never informs T7B.

- [ ] **Step 6: Open PR, put PASS/FAIL in the title/body, squash-merge only after green CI**

The PR body must lead with the frozen scientific result and its failure/success boundary, not with implementation details.

- [ ] **Step 7: Verify merged `main` again**

Require the post-merge `main` CI matrix to complete successfully before claiming integration complete.

---

## Self-Review

**Spec coverage:** The plan covers the fixed world/stream, hidden-trajectory residual definition, moving anchors, reject-only-derived budget, scalar attacker, signed controller, repair events, all ten frozen criteria, diagnostics, docs, CI, review, and merge.

**Placeholder scan:** No TBD/TODO or unspecified implementation steps remain. The one CI literal `EXPECTED_PASS` is explicitly set exactly once from the first frozen scientific run and is part of the required receipt behavior, not an unfixed placeholder.

**Type consistency:** `ResidualBudgetReference`, `ResidualTrial`, `ResidualDecision`, `RepairEvent`, `T7BSummary`, `route_signature`, `residual_rms`, `build_frozen_candidate_stream`, `reject_only_residual_budget`, `consider_scalar_debt_candidate`, `consider_residual_candidate`, `t7b_demo`, and `_passes_t7b` are named consistently across tasks.
