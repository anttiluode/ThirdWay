"""T7B: residual-carrying compatibility memory.

T7A showed that individually small compatible writes can accumulate into
sequence-level forgetting. T7B keeps the same frozen 12 candidate matrices and
measures the collateral wake each accepted edit leaves in hidden-trajectory
response space.

The central comparison is deliberately narrow:

* scalar cumulative debt remembers only how much collateral motion happened;
* signed residual memory remembers the direction of that route motion, so a
  later write can cancel part of an earlier wake.

Both use the same untuned budget derived from the already-frozen T6 reject-only
learner. No candidate direction is rotated or regenerated.
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
    evaluate_all,
)
from experiments.t7_compatible_partial_writes import (
    MIN_TARGET_IMPROVEMENT,
    SCALE_BANK,
    _scaled_candidate,
    t7a_demo,
)


RESIDUAL_EXAMPLES = 48


@dataclass(frozen=True)
class ResidualBudgetReference:
    budget: float
    candidate_seeds: tuple[int, ...]
    accepted: int
    peak_by_skill: tuple[float, float, float]


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


@dataclass(frozen=True)
class RepairEvent:
    index: int
    target_skill: int
    protected_skill: int
    candidate_seed: int
    scale: float
    before_rms: float
    step_rms: float
    after_rms: float


@dataclass(frozen=True)
class T7BSummary:
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
    residual_budget_peaks: tuple[float, float, float]
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


def route_signature(
    model,
    calibration,
    skill: int,
    examples: int = RESIDUAL_EXAMPLES,
) -> np.ndarray:
    """Flatten exact recurrent trajectories for one protected skill."""
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
    anchors = {skill: route_signature(model, calibration, skill) for skill in range(3)}
    residuals = {skill: np.zeros_like(anchors[skill]) for skill in range(3)}
    return anchors, residuals


def reject_only_residual_budget(seed: int = 0) -> ResidualBudgetReference:
    """Derive one common residual envelope from the frozen T6 reject-only path."""
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

        before = {skill: route_signature(model, calibration, skill) for skill in range(3)}
        new_model = apply_edit(model, candidate)
        after = {skill: route_signature(new_model, calibration, skill) for skill in range(3)}
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


def _zero_candidate(model, candidate: CandidateEdit, target_data) -> CandidateEdit:
    loss = float(mean_squared_loss(model, target_data, int(candidate.skill)))
    return CandidateEdit(
        skill=int(candidate.skill),
        seed=int(candidate.seed),
        delta_w=np.zeros_like(candidate.delta_w),
        base_loss=loss,
        edited_loss=loss,
    )


def _measure_scale(
    model,
    *,
    candidate: CandidateEdit,
    target_data,
    calibration,
    scale: float,
) -> tuple[CandidateEdit, dict[int, np.ndarray], dict[int, float]]:
    """Measure one fixed-amplitude candidate in exact trajectory space."""
    scaled = _scaled_candidate(model, candidate, target_data, float(scale))
    protected = tuple(skill for skill in range(3) if skill != int(candidate.skill))
    before = {skill: route_signature(model, calibration, skill) for skill in protected}
    trial_model = apply_edit(model, scaled)
    step_vectors = {
        skill: route_signature(trial_model, calibration, skill) - before[skill]
        for skill in protected
    }
    step_rms = {skill: residual_rms(step_vectors[skill]) for skill in protected}
    return scaled, step_vectors, step_rms


def consider_scalar_debt_candidate(
    model,
    *,
    candidate: CandidateEdit,
    target_data,
    calibration,
    debts: dict[int, float],
    budget: float,
) -> ResidualDecision:
    """Largest useful scale whose unsigned cumulative debt remains in budget."""
    trials: list[ResidualTrial] = []
    protected = tuple(skill for skill in range(3) if skill != int(candidate.skill))

    for scale in SCALE_BANK:
        scaled, step_vectors, step_rms = _measure_scale(
            model,
            candidate=candidate,
            target_data=target_data,
            calibration=calibration,
            scale=scale,
        )
        resultant = {
            skill: float(debts[skill]) + float(step_rms[skill])
            for skill in protected
        }
        trials.append(
            ResidualTrial(
                scale=float(scale),
                target_improvement=float(scaled.improvement),
                step_rms_by_skill=step_rms,
                resultant_rms_by_skill=resultant,
            )
        )
        useful = scaled.improvement > MIN_TARGET_IMPROVEMENT
        compatible = all(value <= float(budget) + 1e-12 for value in resultant.values())
        if useful and compatible:
            return ResidualDecision(
                accepted=True,
                reason="scalar_debt_safe",
                scale=float(scale),
                target_improvement=float(scaled.improvement),
                scaled_candidate=scaled,
                step_vectors_by_skill=step_vectors,
                step_rms_by_skill=step_rms,
                resultant_rms_by_skill=resultant,
                trials=tuple(trials),
            )

    return ResidualDecision(
        accepted=False,
        reason="scalar_debt_exhausted",
        scale=0.0,
        target_improvement=0.0,
        scaled_candidate=_zero_candidate(model, candidate, target_data),
        step_vectors_by_skill={},
        step_rms_by_skill={},
        resultant_rms_by_skill={},
        trials=tuple(trials),
    )


def consider_residual_candidate(
    model,
    *,
    candidate: CandidateEdit,
    target_data,
    calibration,
    residuals: dict[int, np.ndarray],
    budget: float,
) -> ResidualDecision:
    """Largest useful scale whose signed carried residual remains in budget."""
    trials: list[ResidualTrial] = []
    protected = tuple(skill for skill in range(3) if skill != int(candidate.skill))

    for scale in SCALE_BANK:
        scaled, step_vectors, step_rms = _measure_scale(
            model,
            candidate=candidate,
            target_data=target_data,
            calibration=calibration,
            scale=scale,
        )
        resultant_vectors = {
            skill: np.asarray(residuals[skill], dtype=float) + step_vectors[skill]
            for skill in protected
        }
        resultant = {skill: residual_rms(resultant_vectors[skill]) for skill in protected}
        trials.append(
            ResidualTrial(
                scale=float(scale),
                target_improvement=float(scaled.improvement),
                step_rms_by_skill=step_rms,
                resultant_rms_by_skill=resultant,
            )
        )
        useful = scaled.improvement > MIN_TARGET_IMPROVEMENT
        compatible = all(value <= float(budget) + 1e-12 for value in resultant.values())
        if useful and compatible:
            return ResidualDecision(
                accepted=True,
                reason="signed_residual_safe",
                scale=float(scale),
                target_improvement=float(scaled.improvement),
                scaled_candidate=scaled,
                step_vectors_by_skill=step_vectors,
                step_rms_by_skill=step_rms,
                resultant_rms_by_skill=resultant,
                trials=tuple(trials),
            )

    return ResidualDecision(
        accepted=False,
        reason="signed_residual_budget_exhausted",
        scale=0.0,
        target_improvement=0.0,
        scaled_candidate=_zero_candidate(model, candidate, target_data),
        step_vectors_by_skill={},
        step_rms_by_skill={},
        resultant_rms_by_skill={},
        trials=tuple(trials),
    )


def _triplet(values) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size != 3:
        raise ValueError(f"expected three skill values, got {array.size}")
    return (float(array[0]), float(array[1]), float(array[2]))


def t7b_demo(seed: int = 0) -> T7BSummary:
    """Run the frozen signed-residual versus scalar-debt comparison once."""
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

    reference = reject_only_residual_budget(seed)
    stream = build_frozen_candidate_stream(seed)
    t7a = t7a_demo(seed)
    candidate_seeds = tuple(int(candidate.seed) for candidate in stream)
    if candidate_seeds != t7a.candidate_seeds:
        raise RuntimeError("T7B candidate stream diverged from frozen T7A identities")

    scalar_model = make_seed_model(seed=23)
    residual_model = make_seed_model(seed=23)
    scalar_debts = {skill: 0.0 for skill in range(3)}
    _, residuals = _initial_route_state(residual_model, calibration)

    scalar_scales: list[float] = []
    residual_scales: list[float] = []
    scalar_transitions: list[tuple[float, float]] = []
    repairs: list[RepairEvent] = []
    all_direction_pure = True
    all_budget_respected = True

    for index, candidate in enumerate(stream, start=1):
        target = int(candidate.skill)

        scalar_decision = consider_scalar_debt_candidate(
            scalar_model,
            candidate=candidate,
            target_data=proposal,
            calibration=calibration,
            debts=scalar_debts,
            budget=reference.budget,
        )
        scalar_scales.append(float(scalar_decision.scale))
        if scalar_decision.accepted:
            for skill, after in scalar_decision.resultant_rms_by_skill.items():
                before = float(scalar_debts[skill])
                after = float(after)
                scalar_transitions.append((before, after))
                scalar_debts[skill] = after
            scalar_model = apply_edit(scalar_model, scalar_decision.scaled_candidate)
            scalar_debts[target] = 0.0

        residual_decision = consider_residual_candidate(
            residual_model,
            candidate=candidate,
            target_data=proposal,
            calibration=calibration,
            residuals=residuals,
            budget=reference.budget,
        )
        residual_scales.append(float(residual_decision.scale))
        if residual_decision.accepted:
            pure = np.allclose(
                residual_decision.scaled_candidate.delta_w,
                residual_decision.scale * candidate.delta_w,
                atol=1e-12,
                rtol=1e-12,
            )
            all_direction_pure = all_direction_pure and bool(pure)
            for skill, step in residual_decision.step_vectors_by_skill.items():
                before_rms = residual_rms(residuals[skill])
                new_residual = residuals[skill] + step
                after_rms = residual_rms(new_residual)
                step_rms = residual_rms(step)
                all_budget_respected = all_budget_respected and (
                    after_rms <= reference.budget + 1e-12
                )
                if after_rms < before_rms - 1e-8:
                    repairs.append(
                        RepairEvent(
                            index=int(index),
                            target_skill=target,
                            protected_skill=int(skill),
                            candidate_seed=int(candidate.seed),
                            scale=float(residual_decision.scale),
                            before_rms=float(before_rms),
                            step_rms=float(step_rms),
                            after_rms=float(after_rms),
                        )
                    )
                residuals[skill] = new_residual
            residual_model = apply_edit(residual_model, residual_decision.scaled_candidate)
            residuals[target] = np.zeros_like(residuals[target])

    scalar_accuracy = _triplet(evaluate_all(scalar_model, evaluation))
    residual_accuracy = _triplet(evaluate_all(residual_model, evaluation))
    scalar_old = float(np.mean(scalar_accuracy[:2]))
    residual_old = float(np.mean(residual_accuracy[:2]))
    scalar_mean = float(np.mean(scalar_accuracy))
    residual_mean = float(np.mean(residual_accuracy))
    final_residual = tuple(float(residual_rms(residuals[k])) for k in range(3))
    final_debt = tuple(float(scalar_debts[k]) for k in range(3))

    return T7BSummary(
        candidate_seeds=candidate_seeds,
        base_accuracy=t7a.base_accuracy,
        reject_only_accuracy=t7a.reject_only_accuracy,
        t7a_accuracy=t7a.safe_step_accuracy,
        scalar_debt_accuracy=scalar_accuracy,
        residual_accuracy=residual_accuracy,
        finite_control_accuracy=t7a.oracle_accuracy,
        reject_only_accepted=int(t7a.reject_only_accepted),
        scalar_debt_nonzero_writes=sum(scale > 0.0 for scale in scalar_scales),
        residual_nonzero_writes=sum(scale > 0.0 for scale in residual_scales),
        residual_scales=tuple(residual_scales),
        scalar_debt_scales=tuple(scalar_scales),
        residual_budget=float(reference.budget),
        residual_budget_peaks=reference.peak_by_skill,
        final_residual_rms=final_residual,
        final_scalar_debt=final_debt,
        repair_events=len(repairs),
        repair_event_records=tuple(repairs),
        residual_old_skill_accuracy=residual_old,
        scalar_debt_old_skill_accuracy=scalar_old,
        reject_only_old_skill_accuracy=float(t7a.reject_only_old_skill_accuracy),
        residual_mean_accuracy=residual_mean,
        scalar_debt_mean_accuracy=scalar_mean,
        reject_only_mean_accuracy=float(t7a.reject_only_mean_accuracy),
        base_mean_accuracy=float(t7a.base_mean_accuracy),
        residual_switch_penalty=float(_mean_switch_penalty(residual_model, evaluation)),
        reject_only_switch_penalty=float(t7a.reject_only_switch_penalty),
        scalar_debt_switch_penalty=float(_mean_switch_penalty(scalar_model, evaluation)),
        scalar_debt_transitions=tuple(scalar_transitions),
        all_direction_pure=bool(all_direction_pure),
        all_budget_respected=bool(all_budget_respected),
    )


def _passes_t7b(s: T7BSummary) -> bool:
    """Frozen T7B success criteria. Do not tune after observing seed=0."""
    return bool(
        len(s.candidate_seeds) == 12
        and len(set(s.candidate_seeds)) == 12
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


if __name__ == "__main__":
    summary = t7b_demo(seed=0)
    print("T7B residual-carrying memory")
    print("candidate seeds:", summary.candidate_seeds)
    print("budget:", f"{summary.residual_budget:.8f}")
    print("budget peaks:", summary.residual_budget_peaks)
    print("scalar scales:", summary.scalar_debt_scales)
    print("signed scales:", summary.residual_scales)
    print("repair events:", summary.repair_events)
    print(
        "scalar debt",
        " ".join(f"{x:.3f}" for x in summary.scalar_debt_accuracy),
        f"mean={summary.scalar_debt_mean_accuracy:.3f}",
        f"old={summary.scalar_debt_old_skill_accuracy:.3f}",
        f"switch={summary.scalar_debt_switch_penalty:.5f}",
        f"writes={summary.scalar_debt_nonzero_writes}",
    )
    print(
        "signed residual",
        " ".join(f"{x:.3f}" for x in summary.residual_accuracy),
        f"mean={summary.residual_mean_accuracy:.3f}",
        f"old={summary.residual_old_skill_accuracy:.3f}",
        f"switch={summary.residual_switch_penalty:.5f}",
        f"writes={summary.residual_nonzero_writes}",
    )
    print("final residual RMS:", summary.final_residual_rms)
    print("final scalar debt:", summary.final_scalar_debt)
    print("frozen gate pass:", _passes_t7b(summary))
    for event in summary.repair_event_records:
        print("repair", event)
