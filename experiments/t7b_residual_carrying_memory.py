"""T7B residual-carrying memory — RED shell.

This shell exists only to move the TDD failure from import plumbing to the
behavioral residual assertions. It is intentionally wrong and will be replaced
by the minimal implementation once RED is observed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.gate_t6_persistent_skill_compatibility import (
    CandidateEdit,
    apply_edit,
    make_dataset,
    make_seed_model,
)
from experiments.t6_continual_learning import (
    DEMO_EXAMPLES_PER_SKILL,
    DEMO_FIRST_CANDIDATE_SEED,
    DEMO_PROPOSAL_SEED,
    DEMO_SCHEDULE,
    _next_useful_proposal,
)


RESIDUAL_EXAMPLES = 48


@dataclass(frozen=True)
class ResidualBudgetReference:
    budget: float
    candidate_seeds: tuple[int, ...]
    accepted: int
    peak_by_skill: tuple[float, float, float]


def route_signature(model, calibration, skill: int, examples: int = RESIDUAL_EXAMPLES) -> np.ndarray:
    subset = calibration.for_skill(int(skill))
    n = min(int(examples), subset.targets.size)
    return np.zeros(n * 16 * model.W.shape[0], dtype=float)


def residual_rms(vector: np.ndarray) -> float:
    x = np.asarray(vector, dtype=float).reshape(-1)
    if x.size == 0:
        raise ValueError("residual vector must be non-empty")
    return float(np.linalg.norm(x) / np.sqrt(x.size))


def build_frozen_candidate_stream(seed: int = 0) -> tuple[CandidateEdit, ...]:
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


def reject_only_residual_budget(seed: int = 0) -> ResidualBudgetReference:
    stream = build_frozen_candidate_stream(seed)
    return ResidualBudgetReference(
        budget=0.0,
        candidate_seeds=tuple(edit.seed for edit in stream),
        accepted=1,
        peak_by_skill=(0.0, 0.0, 0.0),
    )
