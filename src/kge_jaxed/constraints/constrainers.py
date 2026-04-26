"""Embedding constrainer factories.

Constrainers are projection functions applied to embedding parameters outside the forward computation graph.
In this package they are applied after initialization and after each optimizer update through
``BaseKGE.apply_constraints()``.
"""

from collections.abc import Callable

import jax.numpy as jnp
from jax import Array


def unit_norm(*, eps: float = 1e-9) -> Callable[[Array], Array]:
    """Create a row-wise unit L2 norm constrainer.

    The returned constrainer divides each embedding row by its L2 norm along the last axis. Zero rows remain zero
    because the denominator is floored by ``eps``.

    :param eps: Minimum denominator used for numerical stability.
    :type eps: float
    :return: A constrainer mapping an array to its row-wise normalized version.
    :rtype: Callable[[Array], Array]
    """

    def apply(x: Array) -> Array:
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        return x / jnp.maximum(norm, eps)

    return apply


def max_norm(*, max_value: float = 1.0, eps: float = 1e-9) -> Callable[[Array], Array]:
    """Create a row-wise maximum L2 norm constrainer.

    The returned constrainer leaves rows with norm at most ``max_value`` unchanged and rescales larger rows
    down to norm ``max_value``. This is a projection onto an L2 ball along the final axis.

    :param max_value: Maximum allowed row norm.
    :type max_value: float
    :param eps: Minimum denominator used for numerical stability.
    :type eps: float
    :return: A constrainer mapping rows outside the L2 ball back onto the ball.
    :rtype: Callable[[Array], Array]
    :raises ValueError: If ``max_value`` is not positive.
    """

    if max_value <= 0:
        raise ValueError("max_value must be positive")

    def apply(x: Array) -> Array:
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        scale = jnp.minimum(1.0, max_value / jnp.maximum(norm, eps))
        return x * scale

    return apply


def clip(*, min_value: float = -1.0, max_value: float = 1.0) -> Callable[[Array], Array]:
    """Create an element-wise clipping constrainer.

    The returned constrainer clips every array value into the closed interval ``[min_value, max_value]``.

    :param min_value: Lower clipping bound.
    :type min_value: float
    :param max_value: Upper clipping bound.
    :type max_value: float
    :return: A constrainer that applies element-wise clipping.
    :rtype: Callable[[Array], Array]
    :raises ValueError: If ``min_value`` is greater than ``max_value``.
    """

    if min_value > max_value:
        raise ValueError("min_value must be less than or equal to max_value")

    def apply(x: Array) -> Array:
        return jnp.clip(x, min_value, max_value)

    return apply


def non_negative() -> Callable[[Array], Array]:
    """Create an element-wise non-negative constrainer.

    The returned constrainer replaces negative values with zero and leaves non-negative values unchanged.

    :return: A constrainer that projects values onto the non-negative orthant.
    :rtype: Callable[[Array], Array]
    """

    def apply(x: Array) -> Array:
        return jnp.maximum(x, 0)

    return apply


def unit_modulus(*, eps: float = 1e-9) -> Callable[[Array], Array]:
    """Create an element-wise unit-modulus constrainer.

    The returned constrainer projects each value onto magnitude one. For complex arrays this preserves
    each value's phase and is useful for RotatE-style relation embeddings. Values with magnitude at most ``eps``
    are mapped to ``1`` to avoid an undefined phase at zero.

    :param eps: Minimum magnitude treated as having a defined phase.
    :type eps: float
    :return: A constrainer mapping each value to unit modulus.
    :rtype: Callable[[Array], Array]
    """

    def apply(x: Array) -> Array:
        magnitude = jnp.abs(x)
        projected = x / jnp.where(magnitude > eps, magnitude, jnp.ones_like(magnitude))
        return jnp.where(magnitude > eps, projected, jnp.ones_like(x))

    return apply
