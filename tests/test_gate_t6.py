import numpy as np

from experiments.gate_t6_persistent_skill_compatibility import (
    Dataset,
    evaluate_all,
    generate_candidates,
    make_dataset,
    make_seed_model,
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
