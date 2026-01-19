from collections.abc import Callable

from jax.nn import initializers as jax_initializers


def resolve_embedding_init(
    embedding_init: str | Callable | None, embedding_init_kwargs: dict | None
) -> Callable | None:
    """
    Resolve an embedding initializer from a string name or a callable initializer.

    :param embedding_init: Initializer name (e.g., "uniform", "xavier") or a callable initializer.
    :type embedding_init: str | Callable | None
    :param embedding_init_kwargs: Optional kwargs used only for string-based initializers.
        For xavier/glorot, you may pass scale/mode/distribution to use variance scaling.
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
        return jax_initializers.uniform(**kwargs)
    if name in {"normal"}:
        return jax_initializers.normal(**kwargs)
    if name in {"xavier", "glorot", "xavier_uniform", "glorot_uniform"}:
        return _maybe_variance_scaling(kwargs, distribution="uniform")
    if name in {"xavier_normal", "glorot_normal"}:
        return _maybe_variance_scaling(kwargs, distribution="truncated_normal")
    if name in {"zeros"}:
        return jax_initializers.zeros
    if name in {"ones"}:
        return jax_initializers.ones
    if name in {"orthogonal"}:
        return jax_initializers.orthogonal(**kwargs)

    available = [
        "default",
        "uniform",
        "normal",
        "xavier",
        "xavier_uniform",
        "xavier_normal",
        "glorot_uniform",
        "glorot_normal",
        "zeros",
        "ones",
        "orthogonal",
    ]
    raise ValueError(f"Unknown embedding_init '{embedding_init}'. Available: {available}")


def _maybe_variance_scaling(kwargs: dict, *, distribution: str) -> Callable:
    local_kwargs = dict(kwargs)
    if {"scale", "mode", "distribution"} & local_kwargs.keys():
        return jax_initializers.variance_scaling(
            scale=local_kwargs.pop("scale", 1.0),
            mode=local_kwargs.pop("mode", "fan_avg"),
            distribution=local_kwargs.pop("distribution", distribution),
            **local_kwargs,
        )
    return (
        jax_initializers.glorot_uniform(**local_kwargs)
        if distribution == "uniform"
        else jax_initializers.glorot_normal(**local_kwargs)
    )
