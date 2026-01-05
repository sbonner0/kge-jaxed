"""Utilities for consistent RNG handling across the project."""

from dataclasses import dataclass, field

import jax
from flax import nnx


def make_base_key(seed: int) -> jax.Array:
    """Create a process-safe base key from a seed."""
    base = jax.random.PRNGKey(int(seed))
    return jax.random.fold_in(base, jax.process_index())


def make_model_rngs(seed: int) -> nnx.Rngs:
    """Create model RNG streams from a seed."""
    key = make_base_key(seed)
    k_params, k_dropout = jax.random.split(key, 2)
    return nnx.Rngs(params=k_params, dropout=k_dropout)


@dataclass
class RngManager:
    """Helper for deriving init and per-step keys from a single seed."""

    seed: int
    base_key: jax.Array = field(init=False)
    init_key: jax.Array = field(init=False)
    train_key: jax.Array = field(init=False)

    def __post_init__(self) -> None:
        self.base_key = make_base_key(self.seed)
        self.init_key, self.train_key = jax.random.split(self.base_key)

    def init_rngs(self) -> nnx.Rngs:
        """One-off RNGs for parameter initialization."""
        k_params, k_dropout, k_neg = jax.random.split(self.init_key, 3)
        return nnx.Rngs(params=k_params, dropout=k_dropout, neg=k_neg)

    def step_key(self, step: int, phase: int = 0) -> jax.Array:
        """Generate a key for a specific step and phase."""
        k = jax.random.fold_in(self.train_key, int(phase))
        return jax.random.fold_in(k, int(step))
