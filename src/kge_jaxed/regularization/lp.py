import functools
import math
from typing import Any

import jax
import jax.numpy as jnp


@functools.lru_cache(maxsize=64)
def _expected_norm(p: float | str, d: int) -> float:
    if isinstance(p, str):
        p = float(p)
    if math.isinf(p) and p > 0:
        raise NotImplementedError("Normalization for inf norm is not implemented")
    if math.isfinite(p):
        exp_abs_norm_p = math.pow(2, p / 2) * math.gamma((p + 1) / 2) / math.sqrt(math.pi)
        return math.pow(exp_abs_norm_p * d, 1 / p)
    raise TypeError(f"norm not implemented for {type(p)}: {p}")


class LpRegularizer:
    """Lp norm regularizer over the last axis of each parameter leaf.

    This computes an Lp norm per row (last axis), then reduces across rows
    and leaves. Larger p values penalize large components more strongly, while
    smaller p values encourage sparsity.
    """

    def __init__(self, p: float = 2.0, reduction: str = "mean", normalize: bool = False) -> None:
        if p <= 0:
            raise ValueError("p must be positive")
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.p = float(p)
        self.reduction = reduction
        self.normalize = bool(normalize)

    def __call__(self, params: Any) -> jnp.ndarray:
        leaves = jax.tree_util.tree_leaves(params)
        if not leaves:
            return jnp.array(0.0)

        def _leaf_value(x: jnp.ndarray) -> jnp.ndarray:
            norms = jnp.linalg.norm(x, ord=self.p, axis=-1)
            if self.normalize:
                effective_dim = x.shape[-1] * 2 if jnp.iscomplexobj(x) else x.shape[-1]
                expected = jnp.asarray(_expected_norm(self.p, int(effective_dim)), dtype=norms.dtype)
                norms = norms / expected
            if self.reduction == "mean":
                return jnp.mean(norms)
            return jnp.sum(norms)

        values = [_leaf_value(jnp.asarray(x)) for x in leaves]
        if len(values) == 1:
            return values[0]
        if self.reduction == "mean":
            return jnp.mean(jnp.stack(values))
        return jnp.sum(jnp.stack(values))
