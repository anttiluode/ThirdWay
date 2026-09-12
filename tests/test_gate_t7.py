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
    t7a_demo,
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
    # failed either target usefulness or compatibility. The returned decision
    # records those pre-commit trials so this is auditable rather than inferred.
    larger = [trial for trial in decision.trials if trial.scale > decision.scale]
    assert larger
    assert all(
        trial.target_improvement <= 1e-4
        or max(trial.risk_by_skill.values()) > threshold
        for trial in larger
    )


def test_t7a_frozen_demo_must_escape_protective_stagnation_without_losing_safety():
    """Frozen T7A falsification: scalar safe steps must enable real learning."""
    demo = t7a_demo(seed=0)

    assert demo.candidates_considered == 12
    assert demo.candidate_seeds == demo.reject_only_candidate_seeds
    assert demo.candidate_seeds == demo.accept_all_candidate_seeds
    assert demo.candidate_seeds == demo.oracle_candidate_seeds
    assert len(set(demo.candidate_seeds)) == 12

    assert all(scale == 0.0 or scale in SCALE_BANK for scale in demo.safe_step_scales)
    assert demo.safe_step_nonzero_writes >= demo.reject_only_accepted + 2

    # Preserve essentially the T6 safety result.
    assert demo.safe_step_old_skill_accuracy >= demo.reject_only_old_skill_accuracy - 0.02
    assert demo.safe_step_switch_penalty <= demo.reject_only_switch_penalty + 0.005

    # But unlike reject-only T6, actually improve the repertoire.
    assert demo.safe_step_mean_accuracy >= demo.reject_only_mean_accuracy + 0.01
    assert demo.safe_step_mean_accuracy >= demo.base_mean_accuracy + 0.005

    # The finite-damage oracle is an upper-bound control, never a decision input
    # to the safe-step learner.
    assert demo.oracle_used_for_safe_step is False
