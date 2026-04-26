"""Regularizer registry helpers."""

from kge_jaxed.regularization.lp import LpRegularizer
from kge_jaxed.regularization.np import NpRegularizer

REGULARIZERS = {
    "lp": LpRegularizer,
    "np": NpRegularizer,
    "powersum": NpRegularizer,
    "power_sum": NpRegularizer,
    "n3": NpRegularizer,
}


def get_regularizer(name: str):
    """Get regularizer class by name."""
    if name not in REGULARIZERS:
        raise ValueError(f"Unknown regularizer '{name}'. Available: {list(REGULARIZERS.keys())}")
    return REGULARIZERS[name]


def list_regularizers():
    """Return list of available regularizer names."""
    return list(REGULARIZERS.keys())
