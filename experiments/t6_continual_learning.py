"""Practical continual-learning policy built on the frozen T6 mechanism.

This module deliberately does not change the T6 scientific world or its failed
transport-order criterion.  It uses the surviving directional dynamic-risk
signal as a practical pre-commit guard and compares that guard against blindly
consolidating every locally useful edit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.gate_t6_persistent_skill_compatibility import (
    CandidateEdit,
    Dataset,
    Model,
    _causal_harm_score,
    _switch_penalty,
    apply_edit,
    evaluate_all,
    generate_candidates,
    make_dataset,
    make_seed_model,
    mean_squared_loss,
    run,
)


DEMO_SCHEDULE = (0, 1, 2, 1, 0, 2) * 2
DEMO_PROPOSAL_SEED = 101
DEMO_CALIBRATION_SEED = 103
DEMO_EVALUATION_SEED = 107
DEMO_EXAMPLES_PER_SKILL = 128
DEMO_FIRST_CANDIDATE_SEED = 32
MIN_TARGET_IMPROVEMENT = 1e-4


@dataclass(frozen=True)
class ConsolidationDecision:
    accepted: bool
    reason: str
    target_improvement: float
    risk_by_skill: dict[int, float]
    risk_threshold: float

    @property
    def max_risk(self) -> float:
        if not self.risk_by_skill:
            return 0.0
        return float(max(self.risk_by_skill.values()))


@dataclass(frozen=True)
class DemoEvent:
    index: int
    skill: int
    candidate_seed: int
    target_improvement_accept_all: float
    target_improvement_guarded: float
    max_risk: float
    risk_threshold: float
    accepted: bool
    reason: str
    guarded_accuracy: tuple[float, float, float]
    accept_all_accuracy: tuple[float, float, float]
    used_consider_candidate: bool = True


@dataclass(frozen=True)
class DemoSummary:
    candidates_considered: int
    accepted: int
    rejected: int
    accept_all_accepted: int
    candidate_seeds: tuple[int, ...]
    events: tuple[DemoEvent, ...]
    risk_threshold: float
    base_accuracy: tuple[float, float, float]
    final_accuracy_guarded: tuple[float, float, float]
    final_accuracy_accept_all: tuple[float, float, float]
    final_old_skill_accuracy_guarded: float
    final_old_skill_accuracy_accept_all: float
    final_mean_accuracy_guarded: float
    final_mean_accuracy_accept_all: float
    final_min_accuracy_guarded: float
    final_min_accuracy_accept_all: float
    final_switch_penalty_guarded: float
    final_switch_penalty_accept_all: float


def _as_triplet(values: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    if x.shape != (3,):
        raise ValueError("expected exactly three skill scores")
    return (float(x[0]), float(x[1]), float(x[2]))


def calibrated_risk_threshold() -> float:
    """Return the causal threshold fitted only on the frozen T6 train split."""
    return float(run().predictors["causal"].threshold)


def consider_candidate(
    model: Model,
    retained_edits: list[CandidateEdit] | tuple[CandidateEdit, ...],
    candidate: CandidateEdit,
    target_data: Dataset,
    calibration: Dataset,
    *,
    risk_threshold: float | None = None,
) -> ConsolidationDecision:
    """Decide whether one persistent edit may be consolidated.

    The target-utility check uses only the candidate's own skill.  Compatibility
    then asks whether the same edit is predicted to harm any *other* skill that
    already has a retained edit.  Actual cross-skill damage is never inspected.
    """
    threshold = (
        calibrated_risk_threshold()
        if risk_threshold is None
        else float(risk_threshold)
    )

    before = mean_squared_loss(model, target_data, candidate.skill)
    trial = apply_edit(model, candidate)
    after = mean_squared_loss(trial, target_data, candidate.skill)
    target_improvement = float(before - after)

    retained_skills = sorted(
        {
            int(edit.skill)
            for edit in retained_edits
            if int(edit.skill) != int(candidate.skill)
        }
    )
    risk_by_skill = {
        skill: _causal_harm_score(
            model,
            candidate.delta_w,
            calibration,
            skill,
            shuffled=False,
        )
        for skill in retained_skills
    }

    if target_improvement <= MIN_TARGET_IMPROVEMENT:
        return ConsolidationDecision(
            accepted=False,
            reason="not_useful_in_guarded_state",
            target_improvement=target_improvement,
            risk_by_skill=risk_by_skill,
            risk_threshold=threshold,
        )

    if any(risk > threshold for risk in risk_by_skill.values()):
        return ConsolidationDecision(
            accepted=False,
            reason="dynamic_risk",
            target_improvement=target_improvement,
            risk_by_skill=risk_by_skill,
            risk_threshold=threshold,
        )

    return ConsolidationDecision(
        accepted=True,
        reason="accept",
        target_improvement=target_improvement,
        risk_by_skill=risk_by_skill,
        risk_threshold=threshold,
    )


def _next_useful_proposal(
    model: Model,
    proposal_data: Dataset,
    skill: int,
    start_seed: int,
) -> tuple[CandidateEdit, int]:
    """Find the next target-only useful identity without consulting compatibility."""
    candidate_seed = int(start_seed)
    for _ in range(10_000):
        edits = generate_candidates(
            model,
            proposal_data,
            skill=skill,
            candidate_seeds=(candidate_seed,),
        )
        if edits:
            return edits[0], candidate_seed + 1
        candidate_seed += 1
    raise RuntimeError("could not find a target-only useful demo proposal")


def _mean_switch_penalty(model: Model, data: Dataset) -> float:
    values = []
    for from_skill in range(3):
        for to_skill in range(3):
            if from_skill == to_skill:
                continue
            values.append(_switch_penalty(model, data, from_skill, to_skill))
    return float(np.mean(values))


def continual_demo(seed: int = 0) -> DemoSummary:
    """Run the frozen 12-event guarded-vs-accept-all continual-learning demo.

    Candidate generation belongs to the accept-all trajectory, so both learners
    receive the exact same proposed matrices.  The guarded learner may reject a
    proposal; it never gets to request an easier replacement for itself.
    """
    proposal_data = make_dataset(
        seed=DEMO_PROPOSAL_SEED + int(seed),
        examples_per_skill=DEMO_EXAMPLES_PER_SKILL,
    )
    calibration_data = make_dataset(
        seed=DEMO_CALIBRATION_SEED + int(seed),
        examples_per_skill=DEMO_EXAMPLES_PER_SKILL,
    )
    evaluation_data = make_dataset(
        seed=DEMO_EVALUATION_SEED + int(seed),
        examples_per_skill=DEMO_EXAMPLES_PER_SKILL,
    )

    guarded = make_seed_model(seed=23)
    accept_all = make_seed_model(seed=23)
    base_accuracy = _as_triplet(evaluate_all(guarded, evaluation_data))
    threshold = calibrated_risk_threshold()

    retained_guarded: list[CandidateEdit] = []
    candidate_seeds: list[int] = []
    events: list[DemoEvent] = []
    next_seed = DEMO_FIRST_CANDIDATE_SEED + 1_000 * int(seed)
    accepted = 0

    for index, skill in enumerate(DEMO_SCHEDULE, start=1):
        candidate, next_seed = _next_useful_proposal(
            accept_all,
            proposal_data,
            skill,
            next_seed,
        )
        candidate_seeds.append(int(candidate.seed))

        # Blind learner commits every candidate that its own target-only proposal
        # mechanism found useful.
        accept_all = apply_edit(accept_all, candidate)

        # Guarded learner sees the same matrix.  Its decision is the production
        # consolidation API, not copied scoring logic in this demo loop.
        decision = consider_candidate(
            guarded,
            retained_guarded,
            candidate,
            proposal_data,
            calibration_data,
            risk_threshold=threshold,
        )
        if decision.accepted:
            guarded = apply_edit(guarded, candidate)
            retained_guarded.append(candidate)
            accepted += 1

        guarded_accuracy = _as_triplet(evaluate_all(guarded, evaluation_data))
        accept_all_accuracy = _as_triplet(evaluate_all(accept_all, evaluation_data))
        events.append(
            DemoEvent(
                index=index,
                skill=int(skill),
                candidate_seed=int(candidate.seed),
                target_improvement_accept_all=float(candidate.improvement),
                target_improvement_guarded=float(decision.target_improvement),
                max_risk=float(decision.max_risk),
                risk_threshold=float(decision.risk_threshold),
                accepted=bool(decision.accepted),
                reason=decision.reason,
                guarded_accuracy=guarded_accuracy,
                accept_all_accuracy=accept_all_accuracy,
            )
        )

    final_guarded = _as_triplet(evaluate_all(guarded, evaluation_data))
    final_accept_all = _as_triplet(evaluate_all(accept_all, evaluation_data))

    # The final event is skill C, so A and B are the already-existing skills at
    # that moment.  Keep this definition fixed and visible in the summary.
    old_guarded = float(np.mean(final_guarded[:2]))
    old_accept_all = float(np.mean(final_accept_all[:2]))

    return DemoSummary(
        candidates_considered=len(events),
        accepted=accepted,
        rejected=len(events) - accepted,
        accept_all_accepted=len(events),
        candidate_seeds=tuple(candidate_seeds),
        events=tuple(events),
        risk_threshold=float(threshold),
        base_accuracy=base_accuracy,
        final_accuracy_guarded=final_guarded,
        final_accuracy_accept_all=final_accept_all,
        final_old_skill_accuracy_guarded=old_guarded,
        final_old_skill_accuracy_accept_all=old_accept_all,
        final_mean_accuracy_guarded=float(np.mean(final_guarded)),
        final_mean_accuracy_accept_all=float(np.mean(final_accept_all)),
        final_min_accuracy_guarded=float(np.min(final_guarded)),
        final_min_accuracy_accept_all=float(np.min(final_accept_all)),
        final_switch_penalty_guarded=_mean_switch_penalty(guarded, evaluation_data),
        final_switch_penalty_accept_all=_mean_switch_penalty(accept_all, evaluation_data),
    )
