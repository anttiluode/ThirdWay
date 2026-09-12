from experiments.gate_t0_fusion import run, run_battery


def test_single_seed_is_deterministic():
    a = run(3, mode="local")
    b = run(3, mode="local")
    assert a.late_route_accuracy == b.late_route_accuracy
    assert a.late_reward == b.late_reward
    assert (a.route_probabilities == b.route_probabilities).all()


def test_local_address_beats_source_destroying_controls():
    r = run_battery()
    assert r["local"].route_accuracy > 0.90
    assert r["local"].min_route_accuracy > 0.90
    assert r["local"].reward > r["pooled"].reward + 0.30
    assert r["local"].task_accuracy > r["pooled"].task_accuracy + 0.15
    assert r["shuffled"].route_accuracy < 0.10


def test_pooled_memory_cannot_create_context_specific_router():
    r = run_battery(modes=("pooled",))["pooled"]
    assert abs(r.route_accuracy - 0.5) < 0.05
    assert abs(r.mean_route_probabilities[0, 0] - 0.5) < 1e-12
    assert abs(r.mean_route_probabilities[1, 1] - 0.5) < 1e-12
