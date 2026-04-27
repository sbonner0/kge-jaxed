"""The base class for knowledge graph embedding models."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar

import jax.numpy as jnp
from flax import nnx
from jax import Array

from kge_jaxed.models.base_embedding import BaseEmbedding
from kge_jaxed.registry import constrainers, regularizers
from kge_jaxed.rngs import make_model_rngs


class BaseKGE(ABC, nnx.Module):
    """Base class for knowledge graph embedding models.

    Score convention:
        Implementations must return scores where ``higher is better`` (more plausible).
        All built-in losses are written against this convention.

    Subclasses can define class-level defaults for embedding, regularizer, and
    constrainer configuration by overriding the ``DEFAULT_*_KWARGS`` dictionaries.
    Constructor config resolution follows these rules:

    * ``None`` uses the subclass default.
    * Embedding kwargs are merged over subclass defaults, so callers can set
      options like ``dropout_rate`` without losing model-specific initializers
      or dtypes. To change an embedding default, override the specific key.
    * Regularizer and constrainer kwargs replace subclass defaults, because
      merging those configs can accidentally pass stale kwargs to a different
      registered component.
    * For regularizers and constrainers, an empty dict disables the subclass
      default.
    """

    DEFAULT_ENTITY_EMBEDDING_KWARGS: ClassVar[dict[str, Any]] = {}
    DEFAULT_RELATION_EMBEDDING_KWARGS: ClassVar[dict[str, Any]] = {}
    DEFAULT_ENTITY_REGULARIZER_KWARGS: ClassVar[dict[str, Any]] = {}
    DEFAULT_RELATION_REGULARIZER_KWARGS: ClassVar[dict[str, Any]] = {}
    DEFAULT_ENTITY_CONSTRAINER_KWARGS: ClassVar[dict[str, Any]] = {}
    DEFAULT_RELATION_CONSTRAINER_KWARGS: ClassVar[dict[str, Any]] = {}

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
        :param entity_embedding_kwargs: Args for the entity embedding. If ``None``,
            uses the subclass default. Provided kwargs are merged over the
            subclass default.
        :type entity_embedding_kwargs: dict, optional
        :param relation_embedding_kwargs: Args for the relation embedding. If
            ``None``, uses the subclass default. Provided kwargs are merged over
            the subclass default.
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
        Configured constrainers are applied once after initialization and after
        optimizer updates during training.
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

        entity_embedding_kwargs = self._config_or_default(
            entity_embedding_kwargs,
            "DEFAULT_ENTITY_EMBEDDING_KWARGS",
            merge=True,
        )
        relation_embedding_kwargs = self._config_or_default(
            relation_embedding_kwargs,
            "DEFAULT_RELATION_EMBEDDING_KWARGS",
            merge=True,
        )
        entity_regularizer_kwargs = self._config_or_default(
            entity_regularizer_kwargs,
            "DEFAULT_ENTITY_REGULARIZER_KWARGS",
        )
        relation_regularizer_kwargs = self._config_or_default(
            relation_regularizer_kwargs,
            "DEFAULT_RELATION_REGULARIZER_KWARGS",
        )
        entity_constrainer_kwargs = self._config_or_default(
            entity_constrainer_kwargs,
            "DEFAULT_ENTITY_CONSTRAINER_KWARGS",
        )
        relation_constrainer_kwargs = self._config_or_default(
            relation_constrainer_kwargs,
            "DEFAULT_RELATION_CONSTRAINER_KWARGS",
        )

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
        self.apply_constraints()

    def score_hrt(self, triples: Array, *, dropout_rngs: nnx.Rngs | None = None) -> Array:
        """
        Score a batch of triples.

        :param triples: Input triples of shape [B, 3] where B is the batch size.
        :type triples: Array
        :param dropout_rngs: RNGs for dropout, defaults to None
        :type dropout_rngs: nnx.Rngs | None, optional
        :return: Scores for each triple of shape [B], where higher means more plausible
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

    def regularization_loss_for_ids(
        self,
        entity_ids: Array | None = None,
        relation_ids: Array | None = None,
    ) -> Array:
        """
        Compute regularization loss only for the embeddings touched by the current batch.

        :param entity_ids: Entity ids whose embeddings should be regularized
        :type entity_ids: Array | None
        :param relation_ids: Relation ids whose embeddings should be regularized
        :type relation_ids: Array | None
        :return: Regularization loss
        :rtype: Array
        """
        loss = jnp.array(0.0)

        if self.entity_regularizer is not None and self.entity_regularizer_weight > 0 and entity_ids is not None:
            entity_rows = self.entity_embedding(entity_ids)
            loss = loss + jnp.asarray(self.entity_regularizer_weight) * self.entity_regularizer(entity_rows)

        if self.relation_regularizer is not None and self.relation_regularizer_weight > 0 and relation_ids is not None:
            relation_rows = self.relation_embedding(relation_ids)
            loss = loss + jnp.asarray(self.relation_regularizer_weight) * self.relation_regularizer(relation_rows)

        return loss

    def apply_constraints(self) -> None:
        """
        Apply constraints to entity and relation embeddings if constrainers are defined.
        """
        if self.entity_constrainer is None and self.relation_constrainer is None:
            return
        self.entity_embedding.apply_constrainer(self.entity_constrainer)
        self.relation_embedding.apply_constrainer(self.relation_constrainer)

    @classmethod
    def _config_or_default(
        cls,
        config: dict[str, Any] | None,
        default_name: str,
        *,
        merge: bool = False,
    ) -> dict[str, Any]:
        default_config = dict(getattr(cls, default_name))
        if config is None:
            return default_config
        if merge:
            return default_config | dict(config)
        return dict(config)

    @staticmethod
    def _build_regularizer(kwargs: dict[str, Any]) -> Any | None:
        name = kwargs.get("name")
        if name is None:
            return None

        regularizer_cls = regularizers.get(name)
        regularizer_kwargs = {k: v for k, v in kwargs.items() if k != "name"}
        return regularizer_cls(**regularizer_kwargs)

    @staticmethod
    def _build_constrainer(kwargs: dict[str, Any]) -> Callable[[Array], Array] | None:
        name = kwargs.get("name")
        if name is None:
            return None
        constrainer_factory = constrainers.get(name)
        constrainer_kwargs = {k: v for k, v in kwargs.items() if k != "name"}
        return constrainer_factory(**constrainer_kwargs)

    @abstractmethod
    def interaction_function(self, h: Array, r: Array, t: Array) -> Array:
        """Compute triple plausibility scores with the ``higher is better`` convention."""
        raise NotImplementedError
