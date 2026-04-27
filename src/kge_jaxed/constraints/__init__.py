"""Embedding constrainers."""

from kge_jaxed.constraints.constrainers import (
    clip,
    max_norm,
    non_negative,
    unit_modulus,
    unit_norm,
)

__all__ = [
    "clip",
    "max_norm",
    "non_negative",
    "unit_modulus",
    "unit_norm",
]
