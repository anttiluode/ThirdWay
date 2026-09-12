from experiments.gate_t2_credit_on_moving_road import run_battery


def test_covariant_credit_stays_attached_to_moving_route():
    r = run_battery()
    assert r["covariant"].route_accuracy > 0.90
    assert r["covariant"].min_route_accuracy > 0.90
    assert r["covariant"].max_modal_credit_mismatch < 1e-12


def test_stale_coordinate_credit_hits_the_wrong_road():
    r = run_battery()
    assert r["stale"].route_accuracy < 0.30
    assert r["covariant"].reward > r["stale"].reward + 0.60
    assert r["covariant"].task_accuracy > r["stale"].task_accuracy + 0.30
