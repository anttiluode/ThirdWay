from experiments.gate_t4_causal_collision import run


def test_propagated_overlap_detects_collisions_static_overlap_misses():
    result = run()

    assert result.causal_accuracy == 1.0
    assert result.causal_precision == 1.0
    assert result.causal_recall == 1.0
    assert result.static_recall == 0.0
    assert result.static_precision == 0.0
    assert result.static_accuracy < 0.50
    assert result.max_exact_error < 1e-12
    assert result.harmless_static_rejected == 1.0
    assert result.transported_collision_missed_by_static == 1.0
