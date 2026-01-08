"""The base class for knowledge graph embedding models."""

from abc import ABC, abstractmethod

from flax import nnx
from jax import Array

from kge_jaxed.models.base_embedding import BaseEmbedding
from kge_jaxed.rngs import make_model_rngs


class BaseKGE(ABC, nnx.Module):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        entity_embedding_kwargs: dict | None = None,
        relation_embedding_kwargs: dict | None = None,
        rngs: nnx.Rngs | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Initialize the base class for knowledge graph embedding models.

        :param num_entities: Number of entities.
        :type num_entities: int
        :param num_relations: Number of relations.
        :type num_relations: int
        :param embedding_dim: Dimensionality of the embeddings.
        :type embedding_dim: int
        :param entity_embedding_kwargs: Args for the entity embedding, defaults to {}
        :type entity_embedding_kwargs: dict, optional
        :param relation_embedding_kwargs: Args for the relation embedding, defaults to {}
        :type relation_embedding_kwargs: dict, optional
        :param rngs: RNGs for the module, required unless a seed is provided
        :type rngs: nnx.Rngs, optional
        :param seed: Seed to initialize RNG streams if rngs is not provided
        :type seed: int, optional
        """

        if rngs is None:
            if seed is None:
                raise ValueError("BaseKGE requires rngs or seed to be provided.")
            rngs = make_model_rngs(seed)

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

        if entity_embedding_kwargs is None:
            entity_embedding_kwargs = {}
        if relation_embedding_kwargs is None:
            relation_embedding_kwargs = {}

        self.entity_embedding = BaseEmbedding(
            num_embeddings=self.num_entities, embedding_dim=self.embedding_dim, **entity_embedding_kwargs, rngs=rngs
        )
        self.relation_embedding = BaseEmbedding(
            num_embeddings=self.num_relations, embedding_dim=self.embedding_dim, **relation_embedding_kwargs, rngs=rngs
        )

    def score_hrt(self, triples: Array, *, dropout_rngs: nnx.Rngs | None = None) -> Array:
        """
        Score a batch of triples.

        :param triples: Input triples of shape [B, 3] where B is the batch size.
        :type triples: Array
        :param dropout_rngs: RNGs for dropout, defaults to None
        :type dropout_rngs: nnx.Rngs | None, optional
        :return: Scores for each triple of shape [B]
        :rtype: Array
        """

        # Always pass rngs through; BaseEmbedding treats None as deterministic.
        h = self.entity_embedding(triples[:, 0], rngs=dropout_rngs)
        r = self.relation_embedding(triples[:, 1], rngs=dropout_rngs)
        t = self.entity_embedding(triples[:, 2], rngs=dropout_rngs)

        return self.interaction_function(h, r, t)

    def uses_dropout(self) -> bool:
        return bool(self.entity_embedding.dropout_rate) or bool(self.relation_embedding.dropout_rate)

    @abstractmethod
    def interaction_function(self, h: Array, r: Array, t: Array) -> Array:
        raise NotImplementedError
