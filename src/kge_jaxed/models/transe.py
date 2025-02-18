import jax.numpy as jnp
from jax import Array

from kge_jaxed.models.base_kge import BaseKGE


class TransE(BaseKGE):
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int, norm: int = 2) -> None:
        super().__init__(num_entities, num_relations, embedding_dim)
        self.norm = norm

    def interaction_function(self, h: Array, r: Array, t: Array) -> Array:

        score = h + r - t
        return -jnp.linalg.norm(score, ord=self.norm, axis=1)
