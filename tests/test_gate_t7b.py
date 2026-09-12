import numpy as np

from experiments.gate_t6_persistent_skill_compatibility import (
    apply_edit,
    make_dataset,
    make_seed_model,
)
from experiments.t7_compatible_partial_writes import t7a_demo
from experiments.t7b_residual_carrying_memory import (
    RESIDUAL_EXAMPLES,
    build_frozen_candidate_stream,
    reject_only_residual_budget,
    residual_rms,
    route_signature,
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
