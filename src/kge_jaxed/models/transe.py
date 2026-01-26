import jax.numpy as jnp
from flax import nnx
from jax import Array

from kge_jaxed.models.base_kge import BaseKGE


class TransE(BaseKGE):
    DEFAULT_NORM = 1
    DEFAULT_ENTITY_CONSTRAINER_NAME = "unit_norm"
    DEFAULT_RELATION_CONSTRAINER_NAME = "unit_norm"
    DEFAULT_ENTITY_CONSTRAINER_KWARGS: dict = {}
    DEFAULT_RELATION_CONSTRAINER_KWARGS: dict = {}

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        norm: int = DEFAULT_NORM,
        entity_embedding_kwargs: dict | None = None,
        relation_embedding_kwargs: dict | None = None,
        entity_regularizer_name: str | None = None,
        relation_regularizer_name: str | None = None,
        entity_regularizer_kwargs: dict | None = None,
        relation_regularizer_kwargs: dict | None = None,
        entity_constrainer_name: str | None = None,
        relation_constrainer_name: str | None = None,
        entity_constrainer_kwargs: dict | None = None,
        relation_constrainer_kwargs: dict | None = None,
        rngs: nnx.Rngs | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Initialize a TransE model.

        TransE models relations as translations in embedding space: a valid triple
        (h, r, t) should satisfy h + r ≈ t. The score used during training is the
        negative p-norm distance between h + r and t:
            score(h, r, t) = -||h + r - t||_p

        Defaults:
            norm: 1
            entity_constrainer: unit_norm
            relation_constrainer: unit_norm

        :param num_entities: Number of entities in the knowledge graph.
        :type num_entities: int
        :param num_relations: Number of relations in the knowledge graph.
        :type num_relations: int
        :param embedding_dim: Dimensionality of the embeddings.
        :type embedding_dim: int
        :param norm: Norm order p for the distance (commonly 1 or 2).
        :type norm: int
        :param entity_embedding_kwargs: Extra args for the entity embedding module.
        :type entity_embedding_kwargs: dict, optional
        :param relation_embedding_kwargs: Extra args for the relation embedding module.
        :type relation_embedding_kwargs: dict, optional
        :param entity_regularizer_name: Regularizer name for entity embeddings.
        :type entity_regularizer_name: str | None, optional
        :param relation_regularizer_name: Regularizer name for relation embeddings.
        :type relation_regularizer_name: str | None, optional
        :param entity_regularizer_kwargs: Regularizer kwargs for entities (may include weight).
        :type entity_regularizer_kwargs: dict | None, optional
        :param relation_regularizer_kwargs: Regularizer kwargs for relations (may include weight).
        :type relation_regularizer_kwargs: dict | None, optional
        :param entity_constrainer_name: Constrainer name for entity embeddings.
        :type entity_constrainer_name: str | None, optional
        :param relation_constrainer_name: Constrainer name for relation embeddings.
        :type relation_constrainer_name: str | None, optional
        :param entity_constrainer_kwargs: Constrainer kwargs for entity embeddings.
        :type entity_constrainer_kwargs: dict | None, optional
        :param relation_constrainer_kwargs: Constrainer kwargs for relation embeddings.
        :type relation_constrainer_kwargs: dict | None, optional
        :param rngs: Flax NNX RNGs for initialization and dropout.
        :type rngs: nnx.Rngs, optional
        :param seed: Seed used to create RNGs if none are provided.
        :type seed: int, optional

        Reference:
            Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J., and Yakhnenko, O.
            "Translating Embeddings for Modeling Multi-relational Data."
            NeurIPS 2013.
        """
        entity_constrainer_name, entity_constrainer_kwargs = self._resolve_defaults(
            entity_constrainer_name,
            entity_constrainer_kwargs,
            self.DEFAULT_ENTITY_CONSTRAINER_NAME,
            self.DEFAULT_ENTITY_CONSTRAINER_KWARGS,
        )
        relation_constrainer_name, relation_constrainer_kwargs = self._resolve_defaults(
            relation_constrainer_name,
            relation_constrainer_kwargs,
            self.DEFAULT_RELATION_CONSTRAINER_NAME,
            self.DEFAULT_RELATION_CONSTRAINER_KWARGS,
        )
        super().__init__(
            num_entities,
            num_relations,
            embedding_dim,
            entity_embedding_kwargs=entity_embedding_kwargs,
            relation_embedding_kwargs=relation_embedding_kwargs,
            entity_regularizer_name=entity_regularizer_name,
            relation_regularizer_name=relation_regularizer_name,
            entity_regularizer_kwargs=entity_regularizer_kwargs,
            relation_regularizer_kwargs=relation_regularizer_kwargs,
            entity_constrainer_name=entity_constrainer_name,
            relation_constrainer_name=relation_constrainer_name,
            entity_constrainer_kwargs=entity_constrainer_kwargs,
            relation_constrainer_kwargs=relation_constrainer_kwargs,
            rngs=rngs,
            seed=seed,
        )
        self.norm = norm

    def interaction_function(self, h: Array, r: Array, t: Array) -> Array:
        """
        Compute TransE scores for a batch of triples.

        The TransE score is the negative distance between h + r and t:
            score(h, r, t) = -||h + r - t||_p
        where p is the norm specified by ``norm``.

        :param h: Head entity embeddings of shape [B, D].
        :type h: Array
        :param r: Relation embeddings of shape [B, D].
        :type r: Array
        :param t: Tail entity embeddings of shape [B, D].
        :type t: Array
        :return: Scores of shape [B], higher is better.
        :rtype: Array
        """
        score = h + r - t
        return -jnp.linalg.norm(score, ord=self.norm, axis=1)


if __name__ == "__main__":
    import jax

    model = TransE(num_entities=100, num_relations=10, embedding_dim=32, seed=0)
    model.score_hrt(jax.numpy.array([[0, 3, 2]]))
