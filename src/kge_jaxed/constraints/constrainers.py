from collections.abc import Callable

import jax.numpy as jnp
from jax import Array


def unit_norm(*, eps: float = 1e-9) -> Callable[[Array], Array]:
    def apply(x: Array) -> Array:
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        return x / jnp.maximum(norm, eps)

    return apply


def max_norm(*, max_value: float = 1.0, eps: float = 1e-9) -> Callable[[Array], Array]:
    def apply(x: Array) -> Array:
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        scale = jnp.minimum(1.0, max_value / jnp.maximum(norm, eps))
        return x * scale

    return apply


def clip(*, min_value: float = -1.0, max_value: float = 1.0) -> Callable[[Array], Array]:
    def apply(x: Array) -> Array:
        return jnp.clip(x, min_value, max_value)

    return apply


def non_negative() -> Callable[[Array], Array]:
    def apply(x: Array) -> Array:
        return jnp.maximum(x, 0)

    return apply


def unit_modulus(*, eps: float = 1e-9) -> Callable[[Array], Array]:
    def apply(x: Array) -> Array:
        magnitude = jnp.abs(x)
        return x / jnp.maximum(magnitude, eps)

    return apply
