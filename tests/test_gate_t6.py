import numpy as np

from experiments.gate_t6_persistent_skill_compatibility import (
    evaluate_all,
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
