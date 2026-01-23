from typing import Any

import jax
import jax.numpy as jnp


class LpRegularizer:
    """Lp norm regularizer over the last axis of each parameter leaf.

    This computes an Lp norm per row (last axis), then reduces across rows
    and leaves. Larger p values penalize large components more strongly, while
    smaller p values encourage sparsity.
    """

    def __init__(self, p: float = 2.0, reduction: str = "mean") -> None:
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
            norms = jnp.linalg.norm(x, ord=self.p, axis=-1)
            if self.reduction == "mean":
                return jnp.mean(norms)
            return jnp.sum(norms)

        values = [_leaf_value(jnp.asarray(x)) for x in leaves]
        if len(values) == 1:
            return values[0]
        if self.reduction == "mean":
            return jnp.mean(jnp.stack(values))
        return jnp.sum(jnp.stack(values))
