"""Post-failure diagnostic for T7A cumulative compatibility drift.

This file does not alter T7A decisions.  It replays the frozen safe-step stream
and reports the actual finite non-target loss increment of every accepted write,
along with cumulative calibration-loss drift from the starting network.
"""

from __future__ import annotations

import numpy as np

from experiments.gate_t6_persistent_skill_compatibility import (
    apply_edit,
    make_dataset,
    make_seed_model,
    mean_squared_loss,
)
from experiments.t6_continual_learning import (
    DEMO_CALIBRATION_SEED,
    DEMO_EXAMPLES_PER_SKILL,
    DEMO_FIRST_CANDIDATE_SEED,
    DEMO_PROPOSAL_SEED,
    DEMO_SCHEDULE,
    _next_useful_proposal,
    calibrated_risk_threshold,
)
from experiments.t7_compatible_partial_writes import consider_scaled_candidate


def main() -> None:
    proposal = make_dataset(
        seed=DEMO_PROPOSAL_SEED,
        examples_per_skill=DEMO_EXAMPLES_PER_SKILL,
    )
    calibration = make_dataset(
        seed=DEMO_CALIBRATION_SEED,
        examples_per_skill=DEMO_EXAMPLES_PER_SKILL,
    )
    safe = make_seed_model(seed=23)
    proposal_anchor = make_seed_model(seed=23)
    threshold = calibrated_risk_threshold()
    baseline_loss = np.asarray(
        [mean_squared_loss(safe, calibration, skill) for skill in range(3)],
        dtype=float,
    )
    next_seed = DEMO_FIRST_CANDIDATE_SEED

    print("event skill seed scale pred_risk actual_max_non_target_dMSE cumulative_dMSE_A cumulative_dMSE_B cumulative_dMSE_C")
    increments: list[float] = []
    predicted: list[float] = []
    for index, skill in enumerate(DEMO_SCHEDULE, start=1):
        candidate, next_seed = _next_useful_proposal(
            proposal_anchor,
            proposal,
            int(skill),
            next_seed,
        )
        proposal_anchor = apply_edit(proposal_anchor, candidate)

        before = np.asarray(
            [mean_squared_loss(safe, calibration, k) for k in range(3)],
            dtype=float,
        )
        decision = consider_scaled_candidate(
            safe,
            candidate=candidate,
            target_data=proposal,
            calibration=calibration,
            risk_threshold=threshold,
        )
        if decision.accepted:
            safe = apply_edit(safe, decision.scaled_candidate)
        after = np.asarray(
            [mean_squared_loss(safe, calibration, k) for k in range(3)],
            dtype=float,
        )
        protected = [k for k in range(3) if k != int(skill)]
        actual_increment = float(max(after[k] - before[k] for k in protected))
        cumulative = after - baseline_loss
        increments.append(actual_increment)
        predicted.append(float(decision.max_risk))
        print(
            f"{index:02d} {int(skill)} {int(candidate.seed)} {decision.scale:.3f} "
            f"{decision.max_risk:.6f} {actual_increment:+.6f} "
            f"{cumulative[0]:+.6f} {cumulative[1]:+.6f} {cumulative[2]:+.6f}"
        )

    increments_arr = np.asarray(increments, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)
    print("summary")
    print(f"risk_threshold={threshold:.6f}")
    print(f"positive_actual_harm_steps={int(np.sum(increments_arr > 0.0))}/12")
    print(f"sum_positive_max_non_target_dMSE={float(np.sum(np.maximum(increments_arr, 0.0))):.6f}")
    if np.std(increments_arr) > 1e-15 and np.std(predicted_arr) > 1e-15:
        print(f"corr_predicted_actual_increment={float(np.corrcoef(predicted_arr, increments_arr)[0,1]):.6f}")
    final_loss = np.asarray(
        [mean_squared_loss(safe, calibration, skill) for skill in range(3)],
        dtype=float,
    )
    print("baseline_calibration_loss", " ".join(f"{x:.6f}" for x in baseline_loss))
    print("final_calibration_loss   ", " ".join(f"{x:.6f}" for x in final_loss))
    print("cumulative_loss_delta    ", " ".join(f"{x:+.6f}" for x in final_loss - baseline_loss))


if __name__ == "__main__":
    main()
