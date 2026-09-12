import numpy as np

from experiments.gate_t6_persistent_skill_compatibility import (
    apply_edit,
    make_dataset,
    make_seed_model,
)
from experiments.t7_compatible_partial_writes import SCALE_BANK, t7a_demo
from experiments.t7b_residual_carrying_memory import (
    RESIDUAL_EXAMPLES,
    _passes_t7b,
    build_frozen_candidate_stream,
    consider_residual_candidate,
    reject_only_residual_budget,
    residual_rms,
    route_signature,
    t7b_demo,
)


def test_t7b_route_residual_is_signed_and_telescopes_exactly():
    calibration = make_dataset(seed=103, examples_per_skill=128)
    base = make_seed_model(seed=23)
    stream = build_frozen_candidate_stream(seed=0)
    first = stream[0]
    second = stream[1]

    after_first = apply_edit(base, first)
    after_second = apply_edit(after_first, second)

    h0 = route_signature(base, calibration, skill=2)
    h1 = route_signature(after_first, calibration, skill=2)
    h2 = route_signature(after_second, calibration, skill=2)

    r1 = h1 - h0
    r2 = h2 - h1
    assert RESIDUAL_EXAMPLES == 48
    assert np.allclose(r1 + r2, h2 - h0, atol=1e-12, rtol=1e-12)
    assert residual_rms(r1) > 0.0


def test_t7b_stream_and_budget_are_frozen_from_existing_baseline():
    stream = build_frozen_candidate_stream(seed=0)
    t7a = t7a_demo(seed=0)
    reference = reject_only_residual_budget(seed=0)

    assert tuple(edit.seed for edit in stream) == t7a.candidate_seeds
    assert len(stream) == 12
    assert reference.candidate_seeds == t7a.candidate_seeds
    assert reference.accepted == t7a.reject_only_accepted == 1
    assert reference.budget > 1e-12
    assert reference.budget == max(reference.peak_by_skill)


def test_t7b_controller_never_rotates_candidate_and_respects_budget():
    proposal = make_dataset(seed=101, examples_per_skill=128)
    calibration = make_dataset(seed=103, examples_per_skill=128)
    model = make_seed_model(seed=23)
    stream = build_frozen_candidate_stream(seed=0)
    candidate = stream[0]
    reference = reject_only_residual_budget(seed=0)
    anchors = {k: route_signature(model, calibration, k) for k in range(3)}
    residuals = {k: np.zeros_like(anchors[k]) for k in range(3)}

    decision = consider_residual_candidate(
        model,
        candidate=candidate,
        target_data=proposal,
        calibration=calibration,
        residuals=residuals,
        budget=reference.budget,
    )

    assert decision.scale in (0.0,) + SCALE_BANK
    if decision.accepted:
        assert np.allclose(
            decision.scaled_candidate.delta_w,
            decision.scale * candidate.delta_w,
        )
        assert max(decision.resultant_rms_by_skill.values()) <= reference.budget + 1e-12


def test_t7b_frozen_gate_signed_route_memory_stops_sequence_drift():
    s = t7b_demo(seed=0)

    assert len(s.candidate_seeds) == 12
    assert len(set(s.candidate_seeds)) == 12
    assert s.all_direction_pure
    assert s.all_budget_respected
    assert s.residual_nonzero_writes >= 3
    assert s.residual_old_skill_accuracy >= s.reject_only_old_skill_accuracy - 0.02
    assert s.residual_switch_penalty <= s.reject_only_switch_penalty + 0.005
    assert s.residual_mean_accuracy >= s.reject_only_mean_accuracy + 0.01
    assert s.residual_mean_accuracy >= s.base_mean_accuracy + 0.005
    assert s.residual_nonzero_writes >= s.scalar_debt_nonzero_writes + 2
    assert s.residual_old_skill_accuracy >= s.scalar_debt_old_skill_accuracy - 0.01
    assert s.residual_mean_accuracy >= s.scalar_debt_mean_accuracy + 0.005
    assert s.repair_events >= 2
    assert all(after >= before - 1e-15 for before, after in s.scalar_debt_transitions)
    assert _passes_t7b(s)
