"""T7A: compatible partial writes.

T6 showed that a binary compatibility veto can protect retained computation but
mostly freezes learning. T7A keeps the proposed rank-1 direction fixed and
searches only a predeclared descending amplitude bank. The largest fraction
that remains target-useful and stays below every protected-skill dynamic-risk
threshold may become permanent. No compensation or direction rotation is
allowed in this stage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.gate_t6_persistent_skill_compatibility import (
    COLLISION_THRESHOLD,
    CandidateEdit,
    _causal_harm_score,
    apply_edit,
    evaluate_all,
    make_dataset,
    make_seed_model,
    mean_squared_loss,
)
from experiments.t6_continual_learning import (
    DEMO_CALIBRATION_SEED,
    DEMO_EVALUATION_SEED,
    DEMO_EXAMPLES_PER_SKILL,
    DEMO_FIRST_CANDIDATE_SEED,
    DEMO_PROPOSAL_SEED,
    DEMO_SCHEDULE,
    _mean_switch_penalty,
    _next_useful_proposal,
    calibrated_risk_threshold,
    consider_candidate,
)


SCALE_BANK = (1.0, 0.75, 0.5, 0.25, 0.125)
MIN_TARGET_IMPROVEMENT = 1e-4


@dataclass(frozen=True)
class ScaleTrial:
    scale: float
    target_improvement: float
    risk_by_skill: dict[int, float]

    @property
    def max_risk(self) -> float:
        if not self.risk_by_skill:
            return 0.0
        return float(max(self.risk_by_skill.values()))


@dataclass(frozen=True)
class ScaledDecision:
    accepted: bool
    reason: str
    scale: float
    target_improvement: float
    risk_by_skill: dict[int, float]
    risk_threshold: float
    scaled_candidate: CandidateEdit
    trials: tuple[ScaleTrial, ...]

    @property
    def max_risk(self) -> float:
        if not self.risk_by_skill:
            return 0.0
        return float(max(self.risk_by_skill.values()))


@dataclass(frozen=True)
class T7AEvent:
    index: int
    skill: int
    candidate_seed: int
    reject_only_accepted: bool
    safe_step_scale: float
    safe_step_target_improvement: float
    safe_step_max_risk: float
    oracle_scale: float


@dataclass(frozen=True)
class T7ADemoSummary:
    candidates_considered: int
    candidate_seeds: tuple[int, ...]
    reject_only_candidate_seeds: tuple[int, ...]
    accept_all_candidate_seeds: tuple[int, ...]
    oracle_candidate_seeds: tuple[int, ...]
    safe_step_scales: tuple[float, ...]
    oracle_scales: tuple[float, ...]
    safe_step_nonzero_writes: int
    reject_only_accepted: int
    oracle_nonzero_writes: int
    base_accuracy: tuple[float, float, float]
    safe_step_accuracy: tuple[float, float, float]
    reject_only_accuracy: tuple[float, float, float]
    accept_all_accuracy: tuple[float, float, float]
    oracle_accuracy: tuple[float, float, float]
    safe_step_old_skill_accuracy: float
    reject_only_old_skill_accuracy: float
    accept_all_old_skill_accuracy: float
    oracle_old_skill_accuracy: float
    safe_step_switch_penalty: float
    reject_only_switch_penalty: float
    accept_all_switch_penalty: float
    oracle_switch_penalty: float
    safe_step_mean_accuracy: float
    reject_only_mean_accuracy: float
    accept_all_mean_accuracy: float
    oracle_mean_accuracy: float
    base_mean_accuracy: float
    oracle_used_for_safe_step: bool
    events: tuple[T7AEvent, ...]


def _as_triplet(values: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    if x.shape != (3,):
        raise ValueError("expected three skill scores")
    return (float(x[0]), float(x[1]), float(x[2]))


def _scaled_candidate(
    model,
    candidate: CandidateEdit,
    target_data,
    scale: float,
) -> CandidateEdit:
    """Return the same candidate direction at exactly one scalar amplitude."""
    base_loss = mean_squared_loss(model, target_data, candidate.skill)
    scaled = CandidateEdit(
        skill=int(candidate.skill),
        seed=int(candidate.seed),
        delta_w=float(scale) * candidate.delta_w,
        base_loss=float(base_loss),
        edited_loss=float(base_loss),
    )
    edited_loss = mean_squared_loss(
        apply_edit(model, scaled),
        target_data,
        candidate.skill,
    )
    return CandidateEdit(
        skill=scaled.skill,
        seed=scaled.seed,
        delta_w=scaled.delta_w,
        base_loss=scaled.base_loss,
        edited_loss=float(edited_loss),
    )


def consider_scaled_candidate(
    model,
    *,
    candidate: CandidateEdit,
    target_data,
    calibration,
    risk_threshold: float,
) -> ScaledDecision:
    """Choose the largest useful compatible fraction of one fixed direction."""
    protected_skills = tuple(
        sorted(
            int(skill)
            for skill in np.unique(calibration.skills)
            if int(skill) != int(candidate.skill)
        )
    )
    trials: list[ScaleTrial] = []

    for scale in SCALE_BANK:
        scaled = _scaled_candidate(model, candidate, target_data, scale)
        risk_by_skill = {
            skill: _causal_harm_score(
                model,
                scaled.delta_w,
                calibration,
                skill,
                shuffled=False,
            )
            for skill in protected_skills
        }
        trial = ScaleTrial(
            scale=float(scale),
            target_improvement=float(scaled.improvement),
            risk_by_skill=risk_by_skill,
        )
        trials.append(trial)

        useful = trial.target_improvement > MIN_TARGET_IMPROVEMENT
        compatible = all(risk <= float(risk_threshold) for risk in risk_by_skill.values())
        if useful and compatible:
            return ScaledDecision(
                accepted=True,
                reason="safe_partial" if scale < 1.0 else "safe_full",
                scale=float(scale),
                target_improvement=float(scaled.improvement),
                risk_by_skill=risk_by_skill,
                risk_threshold=float(risk_threshold),
                scaled_candidate=scaled,
                trials=tuple(trials),
            )

    zero_loss = float(mean_squared_loss(model, target_data, candidate.skill))
    zero = CandidateEdit(
        skill=int(candidate.skill),
        seed=int(candidate.seed),
        delta_w=np.zeros_like(candidate.delta_w),
        base_loss=zero_loss,
        edited_loss=zero_loss,
    )
    last_risk = trials[-1].risk_by_skill if trials else {}
    return ScaledDecision(
        accepted=False,
        reason="no_compatible_scale",
        scale=0.0,
        target_improvement=0.0,
        risk_by_skill=last_risk,
        risk_threshold=float(risk_threshold),
        scaled_candidate=zero,
        trials=tuple(trials),
    )


def _finite_oracle_scale(model, candidate, proposal, calibration) -> tuple[float, CandidateEdit | None]:
    """Upper bound: choose largest scale using actual finite protected-skill loss."""
    protected = tuple(
        int(skill)
        for skill in np.unique(calibration.skills)
        if int(skill) != int(candidate.skill)
    )
    base_old = {
        skill: mean_squared_loss(model, calibration, skill)
        for skill in protected
    }

    for scale in SCALE_BANK:
        scaled = _scaled_candidate(model, candidate, proposal, scale)
        if scaled.improvement <= MIN_TARGET_IMPROVEMENT:
            continue
        trial_model = apply_edit(model, scaled)
        damage = {
            skill: mean_squared_loss(trial_model, calibration, skill) - base_old[skill]
            for skill in protected
        }
        if all(value <= COLLISION_THRESHOLD for value in damage.values()):
            return float(scale), scaled
    return 0.0, None


def t7a_demo(seed: int = 0) -> T7ADemoSummary:
    """Run the frozen four-learner T7A continual-learning comparison."""
    seed = int(seed)
    proposal = make_dataset(
        seed=DEMO_PROPOSAL_SEED + seed,
        examples_per_skill=DEMO_EXAMPLES_PER_SKILL,
    )
    calibration = make_dataset(
        seed=DEMO_CALIBRATION_SEED + seed,
        examples_per_skill=DEMO_EXAMPLES_PER_SKILL,
    )
    evaluation = make_dataset(
        seed=DEMO_EVALUATION_SEED + seed,
        examples_per_skill=DEMO_EXAMPLES_PER_SKILL,
    )

    accept_all = make_seed_model(seed=23)
    reject_only = make_seed_model(seed=23)
    safe_step = make_seed_model(seed=23)
    oracle = make_seed_model(seed=23)
    base_accuracy = _as_triplet(evaluate_all(accept_all, evaluation))
    risk_threshold = calibrated_risk_threshold()

    reject_retained: list[CandidateEdit] = []
    candidate_seeds: list[int] = []
    safe_scales: list[float] = []
    oracle_scales: list[float] = []
    events: list[T7AEvent] = []
    reject_accepted = 0
    next_seed = DEMO_FIRST_CANDIDATE_SEED + 1_000 * seed

    for index, skill in enumerate(DEMO_SCHEDULE, start=1):
        candidate, next_seed = _next_useful_proposal(
            accept_all,
            proposal,
            int(skill),
            next_seed,
        )
        candidate_seeds.append(int(candidate.seed))

        reject_decision = consider_candidate(
            reject_only,
            reject_retained,
            candidate,
            proposal,
            calibration,
            risk_threshold=risk_threshold,
        )
        if reject_decision.accepted:
            reject_only = apply_edit(reject_only, candidate)
            reject_retained.append(candidate)
            reject_accepted += 1

        safe_decision = consider_scaled_candidate(
            safe_step,
            candidate=candidate,
            target_data=proposal,
            calibration=calibration,
            risk_threshold=risk_threshold,
        )
        safe_scales.append(float(safe_decision.scale))
        if safe_decision.accepted:
            safe_step = apply_edit(safe_step, safe_decision.scaled_candidate)

        oracle_scale, oracle_edit = _finite_oracle_scale(
            oracle,
            candidate,
            proposal,
            calibration,
        )
        oracle_scales.append(float(oracle_scale))
        if oracle_edit is not None:
            oracle = apply_edit(oracle, oracle_edit)

        # Proposal generation is anchored to accept-all, exactly as in T6, so all
        # four learners receive the identical candidate matrix for this event.
        accept_all = apply_edit(accept_all, candidate)

        events.append(
            T7AEvent(
                index=index,
                skill=int(skill),
                candidate_seed=int(candidate.seed),
                reject_only_accepted=bool(reject_decision.accepted),
                safe_step_scale=float(safe_decision.scale),
                safe_step_target_improvement=float(safe_decision.target_improvement),
                safe_step_max_risk=float(safe_decision.max_risk),
                oracle_scale=float(oracle_scale),
            )
        )

    safe_accuracy = _as_triplet(evaluate_all(safe_step, evaluation))
    reject_accuracy = _as_triplet(evaluate_all(reject_only, evaluation))
    accept_accuracy = _as_triplet(evaluate_all(accept_all, evaluation))
    oracle_accuracy = _as_triplet(evaluate_all(oracle, evaluation))

    safe_old = float(np.mean(safe_accuracy[:2]))
    reject_old = float(np.mean(reject_accuracy[:2]))
    accept_old = float(np.mean(accept_accuracy[:2]))
    oracle_old = float(np.mean(oracle_accuracy[:2]))
    ids = tuple(candidate_seeds)

    return T7ADemoSummary(
        candidates_considered=len(candidate_seeds),
        candidate_seeds=ids,
        reject_only_candidate_seeds=ids,
        accept_all_candidate_seeds=ids,
        oracle_candidate_seeds=ids,
        safe_step_scales=tuple(safe_scales),
        oracle_scales=tuple(oracle_scales),
        safe_step_nonzero_writes=sum(scale > 0.0 for scale in safe_scales),
        reject_only_accepted=int(reject_accepted),
        oracle_nonzero_writes=sum(scale > 0.0 for scale in oracle_scales),
        base_accuracy=base_accuracy,
        safe_step_accuracy=safe_accuracy,
        reject_only_accuracy=reject_accuracy,
        accept_all_accuracy=accept_accuracy,
        oracle_accuracy=oracle_accuracy,
        safe_step_old_skill_accuracy=safe_old,
        reject_only_old_skill_accuracy=reject_old,
        accept_all_old_skill_accuracy=accept_old,
        oracle_old_skill_accuracy=oracle_old,
        safe_step_switch_penalty=_mean_switch_penalty(safe_step, evaluation),
        reject_only_switch_penalty=_mean_switch_penalty(reject_only, evaluation),
        accept_all_switch_penalty=_mean_switch_penalty(accept_all, evaluation),
        oracle_switch_penalty=_mean_switch_penalty(oracle, evaluation),
        safe_step_mean_accuracy=float(np.mean(safe_accuracy)),
        reject_only_mean_accuracy=float(np.mean(reject_accuracy)),
        accept_all_mean_accuracy=float(np.mean(accept_accuracy)),
        oracle_mean_accuracy=float(np.mean(oracle_accuracy)),
        base_mean_accuracy=float(np.mean(base_accuracy)),
        oracle_used_for_safe_step=False,
        events=tuple(events),
    )


if __name__ == "__main__":
    s = t7a_demo()
    print("T7A compatible partial writes")
    print("scale bank:", SCALE_BANK)
    print("candidate seeds:", s.candidate_seeds)
    print("safe-step scales:", s.safe_step_scales)
    print("oracle scales:", s.oracle_scales)
    print(
        "base       ",
        " ".join(f"{x:.3f}" for x in s.base_accuracy),
        f"mean={s.base_mean_accuracy:.3f}",
    )
    print(
        "accept-all ",
        " ".join(f"{x:.3f}" for x in s.accept_all_accuracy),
        f"mean={s.accept_all_mean_accuracy:.3f}",
        f"old={s.accept_all_old_skill_accuracy:.3f}",
        f"switch={s.accept_all_switch_penalty:.5f}",
    )
    print(
        "reject-only",
        " ".join(f"{x:.3f}" for x in s.reject_only_accuracy),
        f"mean={s.reject_only_mean_accuracy:.3f}",
        f"old={s.reject_only_old_skill_accuracy:.3f}",
        f"switch={s.reject_only_switch_penalty:.5f}",
        f"writes={s.reject_only_accepted}",
    )
    print(
        "safe-step  ",
        " ".join(f"{x:.3f}" for x in s.safe_step_accuracy),
        f"mean={s.safe_step_mean_accuracy:.3f}",
        f"old={s.safe_step_old_skill_accuracy:.3f}",
        f"switch={s.safe_step_switch_penalty:.5f}",
        f"writes={s.safe_step_nonzero_writes}",
    )
    print(
        "oracle     ",
        " ".join(f"{x:.3f}" for x in s.oracle_accuracy),
        f"mean={s.oracle_mean_accuracy:.3f}",
        f"old={s.oracle_old_skill_accuracy:.3f}",
        f"switch={s.oracle_switch_penalty:.5f}",
        f"writes={s.oracle_nonzero_writes}",
    )
