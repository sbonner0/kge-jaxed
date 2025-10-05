import jax.numpy as jnp
from flax import nnx
from jax import Array

from kge_jaxed.models.base_kge import BaseKGE
from kge_jaxed.registries import MODELS


@MODELS.register("transe")
class TransE(BaseKGE):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        norm: int = 2,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ) -> None:
        super().__init__(num_entities, num_relations, embedding_dim, rngs=rngs)
        self.norm = norm

    def interaction_function(self, h: Array, r: Array, t: Array) -> Array:

        score = h + r - t
        return -jnp.linalg.norm(score, ord=self.norm, axis=1)


if __name__ == "__main__":
    import jax

    model = TransE(num_entities=100, num_relations=10, embedding_dim=32, rngs=nnx.Rngs(0))
    model.score_hrt(jax.numpy.array([[0, 3, 2]]))
