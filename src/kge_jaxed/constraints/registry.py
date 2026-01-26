"""Constrainer registry helpers."""

from kge_jaxed.constraints.constrainers import (
    clip,
    max_norm,
    non_negative,
    unit_modulus,
    unit_norm,
)

CONSTRAINERS = {
    "unit_norm": unit_norm,
    "max_norm": max_norm,
    "clip": clip,
    "non_negative": non_negative,
    "unit_modulus": unit_modulus,
}


def get_constrainer(name: str):
    """Get constrainer factory by name."""
    if name not in CONSTRAINERS:
        raise ValueError(f"Unknown constrainer '{name}'. Available: {list(CONSTRAINERS.keys())}")
    return CONSTRAINERS[name]


def list_constrainers():
    """Return list of available constrainer names."""
    return list(CONSTRAINERS.keys())
