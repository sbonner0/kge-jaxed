"""Constrainer registry helpers.

The registry maps user-facing string names from model configuration to
constrainer factory functions. Several PyKEEN-style aliases are included for
familiarity, e.g. ``"normalize"`` for ``unit_norm`` and ``"clamp"`` for
``clip``.
"""

from kge_jaxed.constraints.constrainers import (
    clip,
    max_norm,
    non_negative,
    unit_modulus,
    unit_norm,
)

CONSTRAINERS = {
    "unit_norm": unit_norm,
    "normalize": unit_norm,
    "max_norm": max_norm,
    "clamp_norm": max_norm,
    "clip": clip,
    "clamp": clip,
    "non_negative": non_negative,
    "unit_modulus": unit_modulus,
    "complex_normalize": unit_modulus,
}


def get_constrainer(name: str):
    """Get a constrainer factory by name.

    :param name: Registered constrainer name or alias.
    :type name: str
    :return: The corresponding constrainer factory.
    :rtype: Callable
    :raises ValueError: If ``name`` is not registered.
    """

    if name not in CONSTRAINERS:
        raise ValueError(f"Unknown constrainer '{name}'. Available: {list(CONSTRAINERS.keys())}")
    return CONSTRAINERS[name]


def list_constrainers():
    """Return the available constrainer names and aliases.

    :return: Registered constrainer names and aliases.
    :rtype: list[str]
    """

    return list(CONSTRAINERS.keys())
