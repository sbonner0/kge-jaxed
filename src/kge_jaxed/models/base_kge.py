from abc import ABC, abstractmethod

from jax import Array
from jax.typing import ArrayLike

from kge_jaxed.models.base_embedding import BaseEmbedding


class BaseKGE(ABC):

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        entity_embedding_kwargs: dict = {},
        relation_embedding_kwargs: dict = {},
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
        """

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

        self.entity_embedding = BaseEmbedding(
            num_embeddings=self.num_entities, embedding_dim=self.embedding_dim, **entity_embedding_kwargs
        )
        self.relation_embedding = BaseEmbedding(
            num_embeddings=self.num_relations, embedding_dim=self.embedding_dim, **relation_embedding_kwargs
        )

    def score_hrt(self, triples: Array) -> Array:

        h = self.entity_embedding(triples[:, 0])
        r = self.relation_embedding(triples[:, 1])
        t = self.entity_embedding(triples[:, 2])

        return self.interaction_function(h, r, t)

    @abstractmethod
    def interaction_function(self, h: Array, r: Array, t: Array) -> Array:
        raise NotImplementedError
