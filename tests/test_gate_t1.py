from experiments.gate_t1_moving_road import run_battery


def test_covariant_tracker_follows_moving_identity():
    r = run_battery()
    assert r.fixed_identity_accuracy < 0.60
    assert r.covariant_identity_accuracy > 0.99
    assert r.min_covariant_identity_accuracy > 0.99


def test_moving_identity_matters_for_contextual_routing():
    r = run_battery()
    assert r.covariant_route_reward > 0.99
    assert r.fixed_route_reward < 0.15


def test_exact_cocycle_transports_the_subspaces():
    r = run_battery()
    assert r.max_exact_covariance_residual < 1e-12
