"""T7A shell: compatible partial writes.

Production behavior is intentionally not implemented yet.  This shell exists so
RED tests fail on the required safe-step behavior rather than during import.
"""

from __future__ import annotations

from dataclasses import dataclass

from experiments.gate_t6_persistent_skill_compatibility import CandidateEdit


SCALE_BANK = (1.0, 0.75, 0.5, 0.25, 0.125)


@dataclass(frozen=True)
class ScaleTrial:
    scale: float
    target_improvement: float
    risk_by_skill: dict[int, float]


@dataclass(frozen=True)
class ScaledDecision:
    scale: float
    target_improvement: float
    risk_by_skill: dict[int, float]
    scaled_candidate: CandidateEdit
    trials: tuple[ScaleTrial, ...]


def consider_scaled_candidate(
    model,
    *,
    candidate: CandidateEdit,
    target_data,
    calibration,
    risk_threshold: float,
) -> ScaledDecision:
    """RED-phase placeholder; safe-step search is not implemented."""
    return ScaledDecision(
        scale=1.0,
        target_improvement=0.0,
        risk_by_skill={1: float("inf"), 2: float("inf")},
        scaled_candidate=candidate,
        trials=(),
    )
