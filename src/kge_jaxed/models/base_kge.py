"""The base class for knowledge graph embedding models."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
from flax import nnx
from jax import Array

from kge_jaxed.constraints.registry import get_constrainer
from kge_jaxed.models.base_embedding import BaseEmbedding
from kge_jaxed.regularization.registry import get_regularizer
from kge_jaxed.rngs import make_model_rngs


class BaseKGE(ABC, nnx.Module):
    """Base class for knowledge graph embedding models."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        entity_embedding_dim: int,
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
        Initialize the base class for knowledge graph embedding models.

        :param num_entities: Number of entities.
        :type num_entities: int
        :param num_relations: Number of relations.
        :type num_relations: int
        :param entity_embedding_dim: Dimensionality of entity embeddings.
        :type entity_embedding_dim: int
        :param relation_embedding_dim: Dimensionality of relation embeddings. If None,
            uses ``entity_embedding_dim``.
        :type relation_embedding_dim: int | None, optional
        :param entity_embedding_kwargs: Args for the entity embedding, defaults to {}
        :type entity_embedding_kwargs: dict, optional
        :param relation_embedding_kwargs: Args for the relation embedding, defaults to {}
        :type relation_embedding_kwargs: dict, optional
        :param entity_regularizer_kwargs: Regularizer config for entities. Supports
            ``name`` and regularizer kwargs, plus optional ``weight`` handled by BaseKGE.
        :type entity_regularizer_kwargs: dict | None, optional
        :param relation_regularizer_kwargs: Regularizer config for relations. Supports
            ``name`` and regularizer kwargs, plus optional ``weight`` handled by BaseKGE.
        :type relation_regularizer_kwargs: dict | None, optional
        :param entity_constrainer_kwargs: Constrainer config for entity embeddings.
            Supports ``name`` and constrainer kwargs.
        :type entity_constrainer_kwargs: dict | None, optional
        :param relation_constrainer_kwargs: Constrainer config for relation embeddings.
            Supports ``name`` and constrainer kwargs.
        :type relation_constrainer_kwargs: dict | None, optional
        :param rngs: RNGs for the module. If None, a default RNG stream is created
            with seed ``42``.
        :type rngs: nnx.Rngs, optional
        """

        if rngs is None:
            rngs = make_model_rngs(42)

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_embedding_dim = entity_embedding_dim
        self.relation_embedding_dim = (
            relation_embedding_dim if relation_embedding_dim is not None else entity_embedding_dim
        )

        if entity_embedding_kwargs is None:
            entity_embedding_kwargs = {}
        if relation_embedding_kwargs is None:
            relation_embedding_kwargs = {}
        if entity_regularizer_kwargs is None:
            entity_regularizer_kwargs = {}
        if relation_regularizer_kwargs is None:
            relation_regularizer_kwargs = {}
        if entity_constrainer_kwargs is None:
            entity_constrainer_kwargs = {}
        if relation_constrainer_kwargs is None:
            relation_constrainer_kwargs = {}

        # Build embeddings
        self.entity_embedding = BaseEmbedding(
            num_embeddings=self.num_entities,
            embedding_dim=self.entity_embedding_dim,
            **entity_embedding_kwargs,
            rngs=rngs,
        )
        self.relation_embedding = BaseEmbedding(
            num_embeddings=self.num_relations,
            embedding_dim=self.relation_embedding_dim,
            **relation_embedding_kwargs,
            rngs=rngs,
        )

        # Build regularizers for entity and relation embeddings.
        entity_regularizer_cfg = dict(entity_regularizer_kwargs)
        self.entity_regularizer_weight = float(entity_regularizer_cfg.pop("weight", 0.0))
        self.entity_regularizer = self._build_regularizer(entity_regularizer_cfg)

        relation_regularizer_cfg = dict(relation_regularizer_kwargs)
        self.relation_regularizer_weight = float(relation_regularizer_cfg.pop("weight", 0.0))
        self.relation_regularizer = self._build_regularizer(relation_regularizer_cfg)

        # Build constrainers for entity and relation embeddings
        self.entity_constrainer = self._build_constrainer(entity_constrainer_kwargs)
        self.relation_constrainer = self._build_constrainer(relation_constrainer_kwargs)

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

    @property
    def uses_dropout(self) -> bool:
        return bool(self.entity_embedding.dropout_rate) or bool(self.relation_embedding.dropout_rate)

    def entity_weights(self) -> Array:
        return self.entity_embedding.weights()

    def relation_weights(self) -> Array:
        return self.relation_embedding.weights()

    def regularization_loss(self) -> Array:
        """
        Compute the regularization loss for entity and relation embeddings.

        :return: Regularization loss
        :rtype: Array
        """
        loss = jnp.array(0.0)

        if self.entity_regularizer is not None and self.entity_regularizer_weight > 0:
            loss = loss + jnp.asarray(self.entity_regularizer_weight) * self.entity_regularizer(self.entity_weights())

        if self.relation_regularizer is not None and self.relation_regularizer_weight > 0:
            loss = loss + jnp.asarray(self.relation_regularizer_weight) * self.relation_regularizer(
                self.relation_weights()
            )
        return loss

    def apply_constraints(self) -> None:
        """
        Apply constraints to entity and relation embeddings if constrainers are defined.
        """
        if self.entity_constrainer is None and self.relation_constrainer is None:
            return
        self.entity_embedding.apply_constrainer(self.entity_constrainer)
        self.relation_embedding.apply_constrainer(self.relation_constrainer)

    @staticmethod
    def _build_regularizer(kwargs: dict[str, Any]) -> Any | None:
        name = kwargs.get("name")
        if name is None:
            return None

        regularizer_cls = get_regularizer(name)
        regularizer_kwargs = {k: v for k, v in kwargs.items() if k != "name"}
        return regularizer_cls(**regularizer_kwargs)

    @staticmethod
    def _build_constrainer(kwargs: dict[str, Any]) -> Callable[[Array], Array] | None:
        name = kwargs.get("name")
        if name is None:
            return None
        constrainer_factory = get_constrainer(name)
        constrainer_kwargs = {k: v for k, v in kwargs.items() if k != "name"}
        return constrainer_factory(**constrainer_kwargs)

    @abstractmethod
    def interaction_function(self, h: Array, r: Array, t: Array) -> Array:
        raise NotImplementedError
