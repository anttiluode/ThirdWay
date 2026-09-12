import importlib.util


def test_gate_t5_module_exists():
    """T5 must exist as a separate executable nonlinear collision gate."""
    assert importlib.util.find_spec("experiments.gate_t5_nonlinear_collision_radar") is not None
