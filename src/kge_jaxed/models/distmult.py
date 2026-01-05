import jax.numpy as jnp
from flax import nnx
from jax import Array

from kge_jaxed.models.base_kge import BaseKGE


class DistMult(BaseKGE):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        entity_embedding_kwargs: dict = {},
        relation_embedding_kwargs: dict = {},
        rngs: nnx.Rngs | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            num_entities,
            num_relations,
            embedding_dim,
            entity_embedding_kwargs=entity_embedding_kwargs,
            relation_embedding_kwargs=relation_embedding_kwargs,
            rngs=rngs,
            seed=seed,
        )

    def interaction_function(self, h: Array, r: Array, t: Array) -> Array:
        return jnp.sum(h * r * t, axis=1)
