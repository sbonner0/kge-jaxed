from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


class NpRegularizer:
    """Np regularizer (sum |x|^p) over the last axis of each parameter leaf.

    This is a power penalty, not a norm: it omits the 1/p root, so large
    components are penalized more aggressively than Lp. For example, p=3
    corresponds to the N3 regularizer used in some KGE literature.
    """

    def __init__(self, p: float = 3.0, reduction: str = "mean") -> None:
        if p <= 0:
            raise ValueError("p must be positive")
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.p = float(p)
        self.reduction = reduction

    def __call__(self, params: Any) -> jnp.ndarray:
        leaves = jax.tree_util.tree_leaves(params)
        if not leaves:
            return jnp.array(0.0)

        def _leaf_value(x: jnp.ndarray) -> jnp.ndarray:
            values = jnp.sum(jnp.abs(x) ** self.p, axis=-1)
            if self.reduction == "mean":
                return jnp.mean(values)
            return jnp.sum(values)

        values = [_leaf_value(jnp.asarray(x)) for x in leaves]
        if len(values) == 1:
            return values[0]
        if self.reduction == "mean":
            return jnp.mean(jnp.stack(values))
        return jnp.sum(jnp.stack(values))
