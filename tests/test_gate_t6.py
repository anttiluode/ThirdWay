import numpy as np

from experiments.gate_t6_persistent_skill_compatibility import (
    COLLISION_THRESHOLD,
    Dataset,
    _passes_gate,
    build_default_pair_battery,
    evaluate_all,
    fisher_diagonal,
    generate_candidates,
    make_dataset,
    make_seed_model,
    run,
)
from experiments.t6_continual_learning import (
    consider_candidate,
    continual_demo,
)


def test_t6_world_is_deterministic_and_shared():
    d1 = make_dataset(seed=17, examples_per_skill=48)
    d2 = make_dataset(seed=17, examples_per_skill=48)

    assert np.array_equal(d1.inputs, d2.inputs)
    assert np.array_equal(d1.targets, d2.targets)
    assert np.array_equal(d1.skills, d2.skills)
    assert set(np.unique(d1.skills)) == {0, 1, 2}
    assert d1.inputs.shape == (144, 16, 6)

    model = make_seed_model(seed=23)
    assert model.W.shape == (24, 24)
    assert model.B.shape == (24, 6)
    assert model.b.shape == (24,)
    assert model.C.shape == (24,)


def test_t6_seed_model_is_nontrivial_but_imperfect_on_every_skill():
    data = make_dataset(seed=31, examples_per_skill=128)
    score = evaluate_all(make_seed_model(seed=23), data)

    assert score.shape == (3,)
    assert np.all(score > 0.52)
    assert np.all(score < 0.95)


def test_generated_candidates_are_rank1_and_individually_useful():
    data = make_dataset(seed=41, examples_per_skill=128)
    model = make_seed_model(seed=23)
    edits = generate_candidates(model, data, skill=1, candidate_seeds=range(12))

    assert len(edits) >= 4
    for edit in edits:
        assert edit.skill == 1
        assert edit.delta_w.shape == (24, 24)
        assert np.linalg.matrix_rank(edit.delta_w, tol=1e-9) == 1
        assert edit.edited_loss < edit.base_loss - 1e-4


def test_candidate_generation_cannot_see_non_target_labels():
    data = make_dataset(seed=41, examples_per_skill=96)
    changed_targets = data.targets.copy()
    changed_targets[data.skills != 1] *= -1.0
    changed = Dataset(
        inputs=data.inputs.copy(),
        targets=changed_targets,
        skills=data.skills.copy(),
    )
    model = make_seed_model(seed=23)

    a = generate_candidates(model, data, skill=1, candidate_seeds=range(8))
    b = generate_candidates(model, changed, skill=1, candidate_seeds=range(8))

    assert [e.seed for e in a] == [e.seed for e in b]
    assert len(a) > 0
    for left, right in zip(a, b):
        assert np.array_equal(left.delta_w, right.delta_w)
        assert left.base_loss == right.base_loss
        assert left.edited_loss == right.edited_loss


def test_pair_battery_uses_unseen_edit_identities_and_has_both_outcomes():
    battery = build_default_pair_battery()

    assert len(battery) == 192
    train_ids = {
        identity
        for case in battery
        if case.split == "train"
        for identity in (case.candidate_id, case.retained_id)
    }
    test_ids = {
        identity
        for case in battery
        if case.split == "test"
        for identity in (case.candidate_id, case.retained_id)
    }
    assert train_ids.isdisjoint(test_ids)

    held = [case for case in battery if case.split == "test"]
    assert len(held) == 96
    collided = [case for case in held if case.damage.old_skill_damage > COLLISION_THRESHOLD]
    safe = [case for case in held if case.damage.old_skill_damage <= COLLISION_THRESHOLD]
    assert len(collided) >= 0.20 * len(held)
    assert len(safe) >= 0.20 * len(held)

    assert all(case.retained.skill != case.candidate.skill for case in battery)
    assert all(case.retained.improvement > 1e-4 for case in battery)
    assert all(case.candidate.improvement > 1e-4 for case in battery)
    assert all(np.isfinite(case.damage.leakage_delta) for case in battery)
    assert all(np.isfinite(case.damage.switch_cost_delta) for case in battery)


def test_empirical_fisher_is_parameter_sensitivity_not_activation_cosine():
    model = make_seed_model(seed=23)
    data = make_dataset(seed=61, examples_per_skill=12)
    fisher = fisher_diagonal(model, data, skill=0, examples=8)

    assert fisher.shape == (24, 24)
    assert np.all(np.isfinite(fisher))
    assert np.all(fisher >= 0.0)
    assert float(np.sum(fisher)) > 0.0


def test_t6_frozen_scientific_gate_is_recorded_negative_result():
    """Lock the measured outcome instead of rewriting the failed hypothesis."""
    s = run()

    assert s.cases == 192
    assert s.test_cases == 96
    assert s.all_candidates_individually_useful
    assert s.heldout_collision_fraction >= 0.20
    assert s.heldout_safe_fraction >= 0.20

    expected = {
        "parameter",
        "gradient",
        "fisher",
        "static_jacobian",
        "causal",
        "causal_shuffled",
        "causal_reversed",
        "oracle",
    }
    assert set(s.predictors) == expected

    causal = s.predictors["causal"]
    assert causal.damage_correlation > 0.70
    for name in ("parameter", "gradient", "fisher", "static_jacobian"):
        assert causal.test_auroc >= s.predictors[name].test_auroc + 0.10

    assert s.skill_pair_families_with_positive_causal_advantage >= 4
    assert s.predictors["oracle"].test_auroc >= causal.test_auroc - 1e-12
    assert causal.test_auroc >= s.predictors["causal_reversed"].test_auroc + 0.10

    # The frozen gate fails specifically here: detailed transport order did not
    # carry the advantage. Keep that failure reproducible rather than relaxing it.
    assert causal.test_auroc < s.predictors["causal_shuffled"].test_auroc + 0.10
    assert _passes_gate(s) is False


def test_causal_guard_controls_persistent_consolidation_on_same_candidate_stream():
    assert callable(consider_candidate)
    demo = continual_demo(seed=0)

    assert demo.candidates_considered == 12
    assert demo.accepted + demo.rejected == demo.candidates_considered
    assert demo.accept_all_accepted == demo.candidates_considered
    assert demo.accepted < demo.accept_all_accepted
    assert demo.candidate_seeds[0] >= 32
    assert len(set(demo.candidate_seeds)) == demo.candidates_considered
    assert len(demo.events) == demo.candidates_considered
    assert all(event.used_consider_candidate for event in demo.events)

    # This is the practical demo target, not a rewrite of the failed T6 science:
    # the guarded policy should preserve the two skills that are old at the final
    # C event at least as well as blindly consolidating every locally useful edit.
    assert demo.final_old_skill_accuracy_guarded >= demo.final_old_skill_accuracy_accept_all
