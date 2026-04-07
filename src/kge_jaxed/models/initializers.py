from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax.nnx import initializers as nnx_initializers


def _complex_from_real_init(real_init: Callable) -> Callable:
    def init(key, shape, dtype=jnp.complex64):
        dtype = jnp.dtype(dtype)
        real_dtype = jnp.float64 if dtype == jnp.complex128 else jnp.float32
        key_r, key_i = jax.random.split(key)
        real = real_init(key_r, shape, dtype=real_dtype)
        imag = real_init(key_i, shape, dtype=real_dtype)
        return real.astype(dtype) + 1j * imag.astype(dtype)

    return init


def _complex_phase_init() -> Callable:
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
    def init(key, shape, dtype=jnp.float32):
        weights = base_init(key, shape, dtype=dtype)
        norm = jnp.linalg.norm(weights, axis=-1, keepdims=True)
        return weights / jnp.maximum(norm, eps)

    return init


def resolve_embedding_init(
    embedding_init: str | Callable | None, embedding_init_kwargs: dict | None
) -> Callable | None:
    """
    Resolve an embedding initializer from a string name or a callable initializer.

    :param embedding_init: Initializer name (e.g., "uniform", "xavier") or a callable initializer.
    :type embedding_init: str | Callable | None
    :param embedding_init_kwargs: Optional kwargs used only for string-based initializers.
        For xavier/glorot, you may pass scale/mode/distribution to use variance scaling.
        For normalized initializers, you may also pass ``eps`` for numerical stability.
    :type embedding_init_kwargs: dict | None
    :return: A callable initializer, or None to use the underlying layer default.
    :rtype: Callable | None
    """
    if embedding_init is None:
        return None
    if callable(embedding_init):
        return embedding_init
    if not isinstance(embedding_init, str):
        raise TypeError("embedding_init must be a string, a callable, or None.")

    name = embedding_init.strip().lower()
    kwargs = dict(embedding_init_kwargs or {})

    if name in {"default"}:
        return None
    if name in {"uniform"}:
        return nnx_initializers.uniform(**kwargs)
    if name in {"normal"}:
        return nnx_initializers.normal(**kwargs)
    if name in {"complex_normal"}:
        return _complex_from_real_init(nnx_initializers.normal(**kwargs))
    if name in {"xavier", "glorot", "xavier_uniform", "glorot_uniform"}:
        return _maybe_variance_scaling(kwargs, distribution="uniform")
    if name in {"xavier_uniform_norm", "glorot_uniform_norm", "xavier_norm", "glorot_norm"}:
        eps = float(kwargs.pop("eps", 1e-9))
        return _normalized_init(_maybe_variance_scaling(kwargs, distribution="uniform"), eps=eps)
    if name in {"xavier_normal", "glorot_normal"}:
        return _maybe_variance_scaling(kwargs, distribution="truncated_normal")
    if name in {"zeros"}:
        return nnx_initializers.zeros
    if name in {"ones"}:
        return nnx_initializers.ones
    if name in {"orthogonal"}:
        return nnx_initializers.orthogonal(**kwargs)
    if name in {"complex_uniform"}:
        return _complex_from_real_init(nnx_initializers.uniform(**kwargs))
    if name in {"complex_phases", "init_phases", "phases"}:
        return _complex_phase_init()

    available = [
        "default",
        "uniform",
        "normal",
        "complex_normal",
        "xavier",
        "xavier_uniform",
        "xavier_uniform_norm",
        "xavier_norm",
        "xavier_normal",
        "glorot_uniform",
        "glorot_uniform_norm",
        "glorot_norm",
        "glorot_normal",
        "zeros",
        "ones",
        "orthogonal",
        "complex_uniform",
        "complex_phases",
        "init_phases",
        "phases",
    ]
    raise ValueError(f"Unknown embedding_init '{embedding_init}'. Available: {available}")


def _maybe_variance_scaling(kwargs: dict, *, distribution: str) -> Callable:
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
