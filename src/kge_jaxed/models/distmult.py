import jax.numpy as jnp
from flax import nnx
from jax import Array

from kge_jaxed.models.base_kge import BaseKGE
from kge_jaxed.registries import MODELS


@MODELS.register("distmult")
class DistMult(BaseKGE):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ) -> None:
        super().__init__(num_entities, num_relations, embedding_dim, rngs=rngs)

    def interaction_function(self, h: Array, r: Array, t: Array) -> Array:
        return jnp.sum(h * r * t, axis=1)
