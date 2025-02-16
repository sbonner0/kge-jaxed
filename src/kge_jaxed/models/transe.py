import jax.numpy as jnp

from kge_jaxed.models.base_kge import BaseKGE


class TransE(BaseKGE):
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int, norm: int = 2) -> None:
        super().__init__(num_entities, num_relations, embedding_dim)
        self.norm = norm

    def score_hrt(self, triples):
        h = self.entity_embedding(triples[:, 0])
        r = self.relation_embedding(triples[:, 1])
        t = self.entity_embedding(triples[:, 2])

        score = h + r - t
        return -jnp.linalg.norm(score, ord=self.norm, axis=1)
