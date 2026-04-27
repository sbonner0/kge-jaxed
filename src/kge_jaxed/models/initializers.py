"""Embedding initializer resolution helpers.

Initializers are callables used by ``flax.nnx.Embed`` when creating embedding parameters.
They set the starting point for optimization but do not impose any ongoing constraint after training begins.
"""

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax.nnx import initializers as nnx_initializers


def _complex_from_real_init(real_init: Callable) -> Callable:
    """Wrap a real-valued initializer for complex-valued embeddings.

    The returned initializer draws independent real and imaginary parts with the provided real initializer, then
    combines them into a complex array.

    :param real_init: Initializer used independently for real and imaginary parts.
    :type real_init: Callable
    :return: A complex-valued embedding initializer.
    :rtype: Callable
    """

    def init(key, shape, dtype=jnp.complex64):
        dtype = jnp.dtype(dtype)
        real_dtype = jnp.float64 if dtype == jnp.complex128 else jnp.float32
        key_r, key_i = jax.random.split(key)
        real = real_init(key_r, shape, dtype=real_dtype)
        imag = real_init(key_i, shape, dtype=real_dtype)
        return real.astype(dtype) + 1j * imag.astype(dtype)

    return init


def _complex_phase_init() -> Callable:
    """Create a complex unit-phase initializer.

    The returned initializer samples phases uniformly from ``[0, 2*pi)`` and returns ``cos(phase) + i sin(phase)``.
    This is useful for RotatE relation embeddings, where each relation component represents a rotation.

    :return: A complex initializer with unit-modulus values.
    :rtype: Callable
    :raises TypeError: If the requested parameter dtype is not complex.
    """

    def init(key, shape, dtype=jnp.complex64):
        dtype = jnp.dtype(dtype)
        if not jnp.issubdtype(dtype, jnp.complexfloating):
            raise TypeError("init_phases requires a complex dtype.")

        real_dtype = jnp.float64 if dtype == jnp.complex128 else jnp.float32
        phases = jax.random.uniform(
            key,
            shape,
            minval=0.0,
            maxval=2.0 * jnp.pi,
            dtype=real_dtype,
        )
        return jnp.cos(phases).astype(dtype) + 1j * jnp.sin(phases).astype(dtype)

    return init


def _normalized_init(base_init: Callable, *, eps: float = 1e-9) -> Callable:
    """Wrap an initializer with row-wise L2 normalization.

    The returned initializer first calls ``base_init`` and then normalizes each row along the last axis.

    :param base_init: Initializer whose output should be normalized.
    :type base_init: Callable
    :param eps: Minimum denominator used for numerical stability.
    :type eps: float
    :return: A normalized initializer.
    :rtype: Callable
    """

    def init(key, shape, dtype=jnp.float32):
        weights = base_init(key, shape, dtype=dtype)
        norm = jnp.linalg.norm(weights, axis=-1, keepdims=True)
        return weights / jnp.maximum(norm, eps)

    return init


def resolve_embedding_init(
    embedding_init: str | Callable | None, embedding_init_kwargs: dict | None
) -> Callable | None:
    """Resolve an embedding initializer from a string name or callable.

    String names cover common real, normalized, and complex initializers. A callable is returned unchanged unless
    ``embedding_init_kwargs`` are provided, in which case it is wrapped with ``functools.partial``.

    :param embedding_init: Initializer name (e.g., "uniform", "xavier") or a callable initializer.
    :type embedding_init: str | Callable | None
    :param embedding_init_kwargs: Optional kwargs for the initializer.
        For xavier/glorot, you may pass scale/mode/distribution to use variance scaling.
        For normalized initializers, you may also pass ``eps`` for numerical stability.
    :type embedding_init_kwargs: dict | None
    :return: A callable initializer, or None to use the underlying layer default.
    :rtype: Callable | None
    :raises TypeError: If ``embedding_init`` is neither a string, callable, nor ``None``.
    :raises ValueError: If ``embedding_init`` is an unknown string name.
    """
    if embedding_init is None:
        return None
    if callable(embedding_init):
        if embedding_init_kwargs:
            return functools.partial(embedding_init, **embedding_init_kwargs)
        return embedding_init
    if not isinstance(embedding_init, str):
        raise TypeError("embedding_init must be a string, a callable, or None.")

    from kge_jaxed.registry import initializers

    name = embedding_init.strip().lower()
    available = initializers.names()
    if name not in available:
        raise ValueError(f"Unknown embedding_init '{embedding_init}'. Available: {available}")
    return initializers.build(name, **dict(embedding_init_kwargs or {}))


def _maybe_variance_scaling(kwargs: dict, *, distribution: str) -> Callable:
    """Resolve Glorot-style initialization with optional variance scaling.

    If ``kwargs`` contains ``scale``, ``mode``, or ``distribution``, this returns ``variance_scaling`` with
    those values. Otherwise, it returns the standard Glorot uniform or normal initializer matching ``distribution``.

    :param kwargs: Initializer keyword arguments.
    :type kwargs: dict
    :param distribution: Default distribution, either ``"uniform"`` or
        ``"truncated_normal"``.
    :type distribution: str
    :return: A variance-scaling or Glorot initializer.
    :rtype: Callable
    """

    local_kwargs = dict(kwargs)
    if {"scale", "mode", "distribution"} & local_kwargs.keys():
        return nnx_initializers.variance_scaling(
            scale=local_kwargs.pop("scale", 1.0),
            mode=local_kwargs.pop("mode", "fan_avg"),
            distribution=local_kwargs.pop("distribution", distribution),
            **local_kwargs,
        )
    return (
        nnx_initializers.glorot_uniform(**local_kwargs)
        if distribution == "uniform"
        else nnx_initializers.glorot_normal(**local_kwargs)
    )


def _default_initializer(**kwargs) -> Callable | None:
    return None


def _uniform_initializer(**kwargs) -> Callable:
    return nnx_initializers.uniform(**kwargs)


def _uniform_norm_initializer(**kwargs) -> Callable:
    local_kwargs = dict(kwargs)
    eps = float(local_kwargs.pop("eps", 1e-9))
    return _normalized_init(nnx_initializers.uniform(**local_kwargs), eps=eps)


def _normal_initializer(**kwargs) -> Callable:
    return nnx_initializers.normal(**kwargs)


def _normal_norm_initializer(**kwargs) -> Callable:
    local_kwargs = dict(kwargs)
    eps = float(local_kwargs.pop("eps", 1e-9))
    return _normalized_init(nnx_initializers.normal(**local_kwargs), eps=eps)


def _complex_normal_initializer(**kwargs) -> Callable:
    return _complex_from_real_init(nnx_initializers.normal(**kwargs))


def _xavier_uniform_initializer(**kwargs) -> Callable:
    return _maybe_variance_scaling(kwargs, distribution="uniform")


def _xavier_uniform_norm_initializer(**kwargs) -> Callable:
    local_kwargs = dict(kwargs)
    eps = float(local_kwargs.pop("eps", 1e-9))
    return _normalized_init(_maybe_variance_scaling(local_kwargs, distribution="uniform"), eps=eps)


def _xavier_normal_initializer(**kwargs) -> Callable:
    return _maybe_variance_scaling(kwargs, distribution="truncated_normal")


def _xavier_normal_norm_initializer(**kwargs) -> Callable:
    local_kwargs = dict(kwargs)
    eps = float(local_kwargs.pop("eps", 1e-9))
    return _normalized_init(_maybe_variance_scaling(local_kwargs, distribution="truncated_normal"), eps=eps)


def _zeros_initializer(**kwargs) -> Callable:
    return nnx_initializers.zeros


def _ones_initializer(**kwargs) -> Callable:
    return nnx_initializers.ones


def _orthogonal_initializer(**kwargs) -> Callable:
    return nnx_initializers.orthogonal(**kwargs)


def _complex_uniform_initializer(**kwargs) -> Callable:
    return _complex_from_real_init(nnx_initializers.uniform(**kwargs))


def _complex_phase_initializer(**kwargs) -> Callable:
    return _complex_phase_init()
