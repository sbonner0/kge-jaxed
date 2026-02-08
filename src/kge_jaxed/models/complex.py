import jax.numpy as jnp
from flax import nnx
from jax import Array

from kge_jaxed.models.base_kge import BaseKGE


class ComplEx(BaseKGE):
    """ComplEx model for knowledge graph embedding."""

    DEFAULT_ENTITY_EMBEDDING_SIZE = 128
    DEFAULT_ENTITY_EMBEDDING_KWARGS = {
        "param_dtype": jnp.complex64,
        "embedding_init": "complex_normal",
    }
    DEFAULT_RELATION_EMBEDDING_KWARGS = {
        "param_dtype": jnp.complex64,
        "embedding_init": "complex_normal",
    }
    DEFAULT_ENTITY_REGULARIZER_KWARGS = {
        "name": "lp",
        "normalize": True,
        "p": 2.0,
        "weight": 0.01,
    }
    DEFAULT_RELATION_REGULARIZER_KWARGS = {
        "name": "lp",
        "normalize": True,
        "p": 2.0,
        "weight": 0.01,
    }

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        entity_embedding_dim: int = DEFAULT_ENTITY_EMBEDDING_SIZE,
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
        Initialize a ComplEx model.

        ComplEx represents entities and relations as complex-valued embeddings.
        The score for a triple (h, r, t) is computed using the Hermitian dot product:
            score(h, r, t) = Re(<h, r, conjugate(t)>)

        Defaults:
            entity_embedding_dim: 128
            entity_embedding_kwargs: {"param_dtype": jnp.complex64, "embedding_init": "complex_normal"}
            relation_embedding_kwargs: {"param_dtype": jnp.complex64, "embedding_init": "complex_normal"}
            entity_regularizer: lp with {"normalize": True, "p": 2.0, "weight": 0.01}
            relation_regularizer: lp with {"normalize": True, "p": 2.0, "weight": 0.01}

        :param num_entities: Number of entities in the knowledge graph.
        :type num_entities: int
        :param num_relations: Number of relations in the knowledge graph.
        :type num_relations: int
        :param entity_embedding_dim: Dimensionality of entity embeddings (real and imaginary parts).
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
            Trouillon, T., Welbl, J., Riedel, S., Gaussier, E., and Bordes, A.
            "Complex Embeddings for Simple Link Prediction."
            ICML 2016.
        """
        if entity_embedding_kwargs is None:
            entity_embedding_kwargs = dict(self.DEFAULT_ENTITY_EMBEDDING_KWARGS)
        if relation_embedding_kwargs is None:
            relation_embedding_kwargs = dict(self.DEFAULT_RELATION_EMBEDDING_KWARGS)
        if entity_regularizer_kwargs is None:
            entity_regularizer_kwargs = dict(self.DEFAULT_ENTITY_REGULARIZER_KWARGS)
        if relation_regularizer_kwargs is None:
            relation_regularizer_kwargs = dict(self.DEFAULT_RELATION_REGULARIZER_KWARGS)

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
        Compute ComplEx scores for a batch of triples.

        The ComplEx score is computed using the Hermitian dot product:
            score(h, r, t) = Re(<h, r, conjugate(t)>)
        where Re denotes the real part of the complex number.

        :param h: Head entity embeddings of shape [B, D].
        :type h: Array
        :param r: Relation embeddings of shape [B, D].
        :type r: Array
        :param t: Tail entity embeddings of shape [B, D].
        :type t: Array
        :return: Scores of shape [B], higher is better.
        :rtype: Array
        """

        return jnp.real(jnp.sum(h * r * jnp.conj(t), axis=-1))


if __name__ == "__main__":
    import jax

    model = ComplEx(num_entities=100, num_relations=10, entity_embedding_dim=32, rngs=nnx.Rngs(0))
    model.score_hrt(jax.numpy.array([[0, 3, 2]]))
