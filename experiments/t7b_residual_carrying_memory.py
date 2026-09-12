"""T7B: residual-carrying compatibility memory.

T7A showed that individually small compatible writes can accumulate into
sequence-level forgetting. T7B keeps the same frozen 12 candidate matrices and
measures the collateral wake each accepted edit leaves in hidden-trajectory
response space.

This module starts with two primitives:

* a signed finite route residual that telescopes exactly in response space; and
* one untuned common residual budget derived from the already-frozen T6
  reject-only learner on the same proposal stream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.gate_t6_persistent_skill_compatibility import (
    CandidateEdit,
    apply_edit,
    batch_hidden_trajectories,
    make_dataset,
    make_seed_model,
)
from experiments.t6_continual_learning import (
    DEMO_CALIBRATION_SEED,
    DEMO_EXAMPLES_PER_SKILL,
    DEMO_FIRST_CANDIDATE_SEED,
    DEMO_PROPOSAL_SEED,
    DEMO_SCHEDULE,
    _next_useful_proposal,
    calibrated_risk_threshold,
    consider_candidate,
)


RESIDUAL_EXAMPLES = 48


@dataclass(frozen=True)
class ResidualBudgetReference:
    budget: float
    candidate_seeds: tuple[int, ...]
    accepted: int
    peak_by_skill: tuple[float, float, float]


def route_signature(
    model,
    calibration,
    skill: int,
    examples: int = RESIDUAL_EXAMPLES,
) -> np.ndarray:
    """Flatten exact recurrent trajectories for one protected skill.

    The initial all-zero state is omitted.  The result therefore contains only
    the material trajectory produced by the recurrent substrate itself.
    """
    subset = calibration.for_skill(int(skill))
    n = min(int(examples), subset.targets.size)
    if n <= 0:
        raise ValueError(f"calibration has no examples for skill {skill}")
    states = batch_hidden_trajectories(model, subset.inputs[:n])[:, 1:, :]
    return np.asarray(states, dtype=float).reshape(-1)


def residual_rms(vector: np.ndarray) -> float:
    """RMS magnitude of a signed residual vector."""
    x = np.asarray(vector, dtype=float).reshape(-1)
    if x.size == 0:
        raise ValueError("residual vector must be non-empty")
    return float(np.linalg.norm(x) / np.sqrt(x.size))


def build_frozen_candidate_stream(seed: int = 0) -> tuple[CandidateEdit, ...]:
    """Reproduce the exact accept-all-anchored 12-candidate T7A stream."""
    seed = int(seed)
    proposal = make_dataset(
        seed=DEMO_PROPOSAL_SEED + seed,
        examples_per_skill=DEMO_EXAMPLES_PER_SKILL,
    )
    anchor = make_seed_model(seed=23)
    next_seed = DEMO_FIRST_CANDIDATE_SEED + 1_000 * seed
    edits: list[CandidateEdit] = []
    for skill in DEMO_SCHEDULE:
        candidate, next_seed = _next_useful_proposal(
            anchor,
            proposal,
            int(skill),
            next_seed,
        )
        edits.append(candidate)
        anchor = apply_edit(anchor, candidate)
    return tuple(edits)


def _initial_route_state(model, calibration) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    anchors = {
        skill: route_signature(model, calibration, skill)
        for skill in range(3)
    }
    residuals = {
        skill: np.zeros_like(anchors[skill])
        for skill in range(3)
    }
    return anchors, residuals


def reject_only_residual_budget(seed: int = 0) -> ResidualBudgetReference:
    """Derive one common residual envelope from the frozen T6 reject-only path.

    No T7B outcome is consulted.  The replay uses the same 12 accept-all-anchored
    candidates as T7A and the existing T6 risk threshold/decision rule.
    """
    seed = int(seed)
    proposal = make_dataset(
        seed=DEMO_PROPOSAL_SEED + seed,
        examples_per_skill=DEMO_EXAMPLES_PER_SKILL,
    )
    calibration = make_dataset(
        seed=DEMO_CALIBRATION_SEED + seed,
        examples_per_skill=DEMO_EXAMPLES_PER_SKILL,
    )
    stream = build_frozen_candidate_stream(seed)
    model = make_seed_model(seed=23)
    _, residuals = _initial_route_state(model, calibration)
    retained: list[CandidateEdit] = []
    peaks = np.zeros(3, dtype=float)
    accepted = 0
    threshold = calibrated_risk_threshold()

    for candidate in stream:
        decision = consider_candidate(
            model,
            retained,
            candidate,
            proposal,
            calibration,
            risk_threshold=threshold,
        )
        if not decision.accepted:
            continue

        before = {
            skill: route_signature(model, calibration, skill)
            for skill in range(3)
        }
        new_model = apply_edit(model, candidate)
        after = {
            skill: route_signature(new_model, calibration, skill)
            for skill in range(3)
        }
        target = int(candidate.skill)
        for skill in range(3):
            if skill == target:
                residuals[skill] = np.zeros_like(after[skill])
            else:
                residuals[skill] = residuals[skill] + (after[skill] - before[skill])
            peaks[skill] = max(peaks[skill], residual_rms(residuals[skill]))

        model = new_model
        retained.append(candidate)
        accepted += 1

    budget = float(np.max(peaks))
    if budget <= 1e-12:
        raise RuntimeError("reject-only residual envelope is numerically empty")
    return ResidualBudgetReference(
        budget=budget,
        candidate_seeds=tuple(int(edit.seed) for edit in stream),
        accepted=int(accepted),
        peak_by_skill=(float(peaks[0]), float(peaks[1]), float(peaks[2])),
    )
