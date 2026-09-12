import numpy as np

from experiments.gate_t6_persistent_skill_compatibility import (
    generate_candidates,
    make_dataset,
    make_seed_model,
)
from experiments.t6_continual_learning import calibrated_risk_threshold
from experiments.t7_compatible_partial_writes import (
    SCALE_BANK,
    consider_scaled_candidate,
)


def test_t7a_chooses_largest_safe_useful_fraction_of_same_candidate():
    model = make_seed_model(seed=23)
    proposal = make_dataset(seed=101, examples_per_skill=128)
    calibration = make_dataset(seed=103, examples_per_skill=128)
    candidate = generate_candidates(
        model,
        proposal,
        skill=0,
        candidate_seeds=(32,),
    )[0]
    threshold = calibrated_risk_threshold()

    decision = consider_scaled_candidate(
        model,
        candidate=candidate,
        target_data=proposal,
        calibration=calibration,
        risk_threshold=threshold,
    )

    assert SCALE_BANK == (1.0, 0.75, 0.5, 0.25, 0.125)
    assert 0.0 < decision.scale < 1.0
    assert decision.scale in SCALE_BANK
    assert decision.target_improvement > 1e-4
    assert max(decision.risk_by_skill.values()) <= threshold

    # T7A must keep the original direction: only amplitude may change.
    assert np.allclose(
        decision.scaled_candidate.delta_w,
        decision.scale * candidate.delta_w,
    )

    # Because the bank is descending, every larger tested fraction must have
    # failed either target usefulness or compatibility.  The returned decision
    # records those pre-commit trials so this is auditable rather than inferred.
    larger = [trial for trial in decision.trials if trial.scale > decision.scale]
    assert larger
    assert all(
        trial.target_improvement <= 1e-4
        or max(trial.risk_by_skill.values()) > threshold
        for trial in larger
    )
