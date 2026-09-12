"""T7A: compatible partial writes.

T6 showed that a binary compatibility veto can protect retained computation but
mostly freezes learning.  T7A keeps the proposed rank-1 direction fixed and
searches only a predeclared descending amplitude bank.  The largest fraction
that remains target-useful and stays below every protected-skill dynamic-risk
threshold may become permanent.  No compensation or direction rotation is
allowed in this stage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.gate_t6_persistent_skill_compatibility import (
    CandidateEdit,
    _causal_harm_score,
    apply_edit,
    mean_squared_loss,
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

    zero = CandidateEdit(
        skill=int(candidate.skill),
        seed=int(candidate.seed),
        delta_w=np.zeros_like(candidate.delta_w),
        base_loss=float(mean_squared_loss(model, target_data, candidate.skill)),
        edited_loss=float(mean_squared_loss(model, target_data, candidate.skill)),
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
