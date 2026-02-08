import jax.numpy as jnp
from flax import nnx
from jax import Array

from kge_jaxed.models.base_kge import BaseKGE


class TransE(BaseKGE):
    """TransE model for knowledge graph embedding."""

    DEFAULT_EMBEDDING_DIM = 50
    DEFAULT_NORM = 1
    DEFAULT_ENTITY_CONSTRAINER_KWARGS = {"name": "unit_norm"}
    DEFAULT_RELATION_CONSTRAINER_KWARGS = {"name": "unit_norm"}

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        entity_embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        relation_embedding_dim: int | None = None,
        norm: int = DEFAULT_NORM,
        entity_embedding_kwargs: dict | None = None,
        relation_embedding_kwargs: dict | None = None,
        entity_regularizer_kwargs: dict | None = None,
        relation_regularizer_kwargs: dict | None = None,
        entity_constrainer_kwargs: dict | None = None,
        relation_constrainer_kwargs: dict | None = None,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        """
        Initialize a TransE model.

        TransE models relations as translations in embedding space: a valid triple
        (h, r, t) should satisfy h + r ≈ t. The score used during training is the
        negative p-norm distance between h + r and t:
            score(h, r, t) = -||h + r - t||_p

        Defaults:
            entity_embedding_dim: 50
            norm: 1
            entity_constrainer: unit_norm
            relation_constrainer: unit_norm

        :param num_entities: Number of entities in the knowledge graph.
        :type num_entities: int
        :param num_relations: Number of relations in the knowledge graph.
        :type num_relations: int
        :param entity_embedding_dim: Dimensionality of entity embeddings.
        :type entity_embedding_dim: int
        :param relation_embedding_dim: Dimensionality of relation embeddings. If None,
            uses ``entity_embedding_dim``.
        :type relation_embedding_dim: int | None, optional
        :param norm: Norm order p for the distance (commonly 1 or 2).
        :type norm: int
        :param entity_embedding_kwargs: Extra args for the entity embedding module.
        :type entity_embedding_kwargs: dict, optional
        :param relation_embedding_kwargs: Extra args for the relation embedding module.
        :type relation_embedding_kwargs: dict, optional
        :param entity_regularizer_kwargs: Regularizer config for entities. Supports
            ``name`` and regularizer kwargs, plus optional ``weight``.
        :type entity_regularizer_kwargs: dict | None, optional
        :param relation_regularizer_kwargs: Regularizer config for relations. Supports
            ``name`` and regularizer kwargs, plus optional ``weight``.
        :type relation_regularizer_kwargs: dict | None, optional
        :param entity_constrainer_kwargs: Constrainer config for entity embeddings.
            Supports ``name`` and constrainer kwargs.
        :type entity_constrainer_kwargs: dict | None, optional
        :param relation_constrainer_kwargs: Constrainer config for relation embeddings.
            Supports ``name`` and constrainer kwargs.
        :type relation_constrainer_kwargs: dict | None, optional
        :param rngs: Flax NNX RNGs for initialization and dropout.
        :type rngs: nnx.Rngs, optional

        Reference:
            Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J., and Yakhnenko, O.
            "Translating Embeddings for Modeling Multi-relational Data."
            NeurIPS 2013.
        """
        if entity_constrainer_kwargs is None:
            entity_constrainer_kwargs = dict(self.DEFAULT_ENTITY_CONSTRAINER_KWARGS)
        if relation_constrainer_kwargs is None:
            relation_constrainer_kwargs = dict(self.DEFAULT_RELATION_CONSTRAINER_KWARGS)

        super().__init__(
            num_entities=num_entities,
            num_relations=num_relations,
            entity_embedding_dim=entity_embedding_dim,
            relation_embedding_dim=relation_embedding_dim,
            entity_embedding_kwargs=entity_embedding_kwargs,
            relation_embedding_kwargs=relation_embedding_kwargs,
            entity_regularizer_kwargs=entity_regularizer_kwargs,
            relation_regularizer_kwargs=relation_regularizer_kwargs,
            entity_constrainer_kwargs=entity_constrainer_kwargs,
            relation_constrainer_kwargs=relation_constrainer_kwargs,
            rngs=rngs,
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

    model = TransE(num_entities=100, num_relations=10, entity_embedding_dim=32, rngs=nnx.Rngs(0))
    model.score_hrt(jax.numpy.array([[0, 3, 2]]))
