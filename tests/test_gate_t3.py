from experiments.gate_t3_identifiability_credit import _learn_with_tracker


def test_ambiguity_blocks_preserve_crossing_identity():
    overlap = _learn_with_tracker(0, "overlap")
    blocks = _learn_with_tracker(0, "blocks")

    assert blocks.semantic_fidelity > 0.98
    assert blocks.route_accuracy > 0.90
    assert blocks.route_accuracy > overlap.route_accuracy + 0.25


def test_exact_hidden_gauge_requires_more_evidence():
    blocks = _learn_with_tracker(0, "blocks", hidden_gauge=True)
    active = _learn_with_tracker(0, "active_family", hidden_gauge=True)

    assert blocks.semantic_fidelity < 0.20
    assert active.semantic_fidelity > 0.98
    assert active.route_accuracy > 0.90
    assert active.route_accuracy > blocks.route_accuracy + 0.50
    assert 0.10 < active.query_fraction < 0.30
