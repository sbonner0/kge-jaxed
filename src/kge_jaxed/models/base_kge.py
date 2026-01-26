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
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
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
            num_embeddings=self.num_entities, embedding_dim=self.embedding_dim, **entity_embedding_kwargs, rngs=rngs
        )
        self.relation_embedding = BaseEmbedding(
            num_embeddings=self.num_relations, embedding_dim=self.embedding_dim, **relation_embedding_kwargs, rngs=rngs
        )

        self.entity_regularizer_name = entity_regularizer_name
        self.relation_regularizer_name = relation_regularizer_name
        self.entity_regularizer_kwargs = dict(entity_regularizer_kwargs)
        self.relation_regularizer_kwargs = dict(relation_regularizer_kwargs)
        self.entity_regularizer_weight = float(self.entity_regularizer_kwargs.pop("weight", 0.0))
        self.relation_regularizer_weight = float(self.relation_regularizer_kwargs.pop("weight", 0.0))
        if self.entity_regularizer_name is None and self.entity_regularizer_weight > 0:
            raise ValueError("entity_regularizer_name must be set when entity_regularizer_weight > 0")
        if self.relation_regularizer_name is None and self.relation_regularizer_weight > 0:
            raise ValueError("relation_regularizer_name must be set when relation_regularizer_weight > 0")
        self.entity_regularizer = self._build_regularizer(
            self.entity_regularizer_name,
            self.entity_regularizer_kwargs,
        )
        self.relation_regularizer = self._build_regularizer(
            self.relation_regularizer_name,
            self.relation_regularizer_kwargs,
        )
        self.entity_constrainer = self._build_constrainer(entity_constrainer_name, entity_constrainer_kwargs)
        self.relation_constrainer = self._build_constrainer(relation_constrainer_name, relation_constrainer_kwargs)

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

    def entity_weights(self) -> Array:
        return self.entity_embedding.weights()

    def relation_weights(self) -> Array:
        return self.relation_embedding.weights()

    def regularization_loss(self) -> Array:
        loss = jnp.array(0.0)
        if self.entity_regularizer is not None and self.entity_regularizer_weight > 0:
            loss = loss + jnp.asarray(self.entity_regularizer_weight) * self.entity_regularizer(self.entity_weights())
        if self.relation_regularizer is not None and self.relation_regularizer_weight > 0:
            loss = loss + jnp.asarray(self.relation_regularizer_weight) * self.relation_regularizer(
                self.relation_weights()
            )
        return loss

    def apply_constraints(self) -> None:
        if self.entity_constrainer is None and self.relation_constrainer is None:
            return
        self.entity_embedding.apply_constrainer(self.entity_constrainer)
        self.relation_embedding.apply_constrainer(self.relation_constrainer)

    @staticmethod
    def _merge_kwargs(defaults: dict[str, Any] | None, overrides: dict[str, Any] | None) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        if defaults:
            merged.update(defaults)
        if overrides:
            merged.update(overrides)
        return merged

    @classmethod
    def _resolve_defaults(
        cls,
        name: str | None,
        kwargs: dict[str, Any] | None,
        default_name: str | None,
        default_kwargs: dict[str, Any] | None,
    ) -> tuple[str | None, dict[str, Any] | None]:
        if name is None:
            name = default_name
            kwargs = cls._merge_kwargs(default_kwargs, kwargs)
        elif name == default_name:
            kwargs = cls._merge_kwargs(default_kwargs, kwargs)
        return name, kwargs

    @staticmethod
    def _build_regularizer(name: str | None, kwargs: dict[str, Any]) -> Any | None:
        if name is None:
            return None
        regularizer_cls = get_regularizer(name)
        return regularizer_cls(**kwargs)

    @staticmethod
    def _build_constrainer(name: str | None, kwargs: dict[str, Any]) -> Callable[[Array], Array] | None:
        if name is None:
            return None
        constrainer_factory = get_constrainer(name)
        return constrainer_factory(**kwargs)

    @abstractmethod
    def interaction_function(self, h: Array, r: Array, t: Array) -> Array:
        raise NotImplementedError
