from experiments.gate_t5_nonlinear_collision_radar import run


def test_propagated_causal_radar_predicts_heldout_nonlinear_collisions():
    """Dynamic transport should beat raw/static overlap on held-out sites."""
    s = run()

    assert s.cases == 192
    assert s.train_cases == 96
    assert s.test_cases == 96
    assert s.min_isolated_effect > 1e-3

    assert s.causal_correlation > 0.98
    assert s.causal_test_accuracy > 0.95
    assert s.causal_test_precision > 0.95
    assert s.causal_test_recall > 0.95

    assert s.parameter_test_recall < 0.10
    assert s.fisher_test_recall < 0.10
    assert s.jacobian_test_recall < 0.10
    assert s.causal_test_accuracy > s.parameter_test_accuracy + 0.30
    assert s.causal_test_accuracy > s.fisher_test_accuracy + 0.30
    assert s.causal_test_accuracy > s.jacobian_test_accuracy + 0.30
