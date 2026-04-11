import jax.numpy as jnp
from flax import nnx
from jax import Array

from kge_jaxed.models.base_kge import BaseKGE


class DistMult(BaseKGE):
    """DistMult model for knowledge graph embedding."""

    DEFAULT_EMBEDDING_DIM = 50
    DEFAULT_ENTITY_CONSTRAINER_KWARGS = {"name": "unit_norm"}
    DEFAULT_RELATION_REGULARIZER_KWARGS = {
        "name": "lp",
        "normalize": True,
        "p": 2.0,
        "weight": 0.1,
    }

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        entity_embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        relation_embedding_dim: int | None = None,
        entity_embedding_kwargs: dict | None = None,
        relation_embedding_kwargs: dict | None = None,
        entity_regularizer_kwargs: dict | None = None,
        relation_regularizer_kwargs: dict | None = None,
        entity_constrainer_kwargs: dict | None = None,
        relation_constrainer_kwargs: dict | None = None,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        """
        Initialize a DistMult model.

        DistMult scores triples via a bilinear diagonal interaction:
            score(h, r, t) = <h, r, t> = sum_i h_i * r_i * t_i

        Defaults:
            entity_embedding_dim: 50
            entity_constrainer: unit_norm
            relation_regularizer: lp with {"normalize": True, "p": 2.0, "weight": 0.1}

        :param num_entities: Number of entities in the knowledge graph.
        :type num_entities: int
        :param num_relations: Number of relations in the knowledge graph.
        :type num_relations: int
        :param entity_embedding_dim: Dimensionality of entity embeddings.
        :type entity_embedding_dim: int
        :param relation_embedding_dim: Dimensionality of relation embeddings. If None,
            uses ``entity_embedding_dim``.
        :type relation_embedding_dim: int | None, optional
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
            Yang, B., Yih, W., He, X., Gao, J., and Deng, L.
            "Embedding Entities and Relations for Learning and Inference in Knowledge Bases."
            ICLR 2015.
        """
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

    def interaction_function(self, h: Array, r: Array, t: Array) -> Array:
        """
        Compute DistMult scores for a batch of triples.

        The DistMult score is the tri-linear dot product:
            score(h, r, t) = sum_i h_i * r_i * t_i

        :param h: Head entity embeddings of shape [B, D].
        :type h: Array
        :param r: Relation embeddings of shape [B, D].
        :type r: Array
        :param t: Tail entity embeddings of shape [B, D].
        :type t: Array
        :return: Scores of shape [B], higher is better.
        :rtype: Array
        """
        return jnp.sum(h * r * t, axis=1)
