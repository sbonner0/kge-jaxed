import jax.numpy as jnp
from flax import nnx
from jax import Array

from kge_jaxed.models.base_kge import BaseKGE


class ComplEx(BaseKGE):
    DEFAULT_ENTITY_EMBEDDING_KWARGS = {
        "param_dtype": jnp.complex64,
        "embedding_init": "complex_normal",
    }
    DEFAULT_RELATION_EMBEDDING_KWARGS = {
        "param_dtype": jnp.complex64,
        "embedding_init": "complex_normal",
    }
    DEFAULT_ENTITY_REGULARIZER_NAME = "lp"
    DEFAULT_RELATION_REGULARIZER_NAME = "lp"
    DEFAULT_ENTITY_REGULARIZER_KWARGS = {
        "normalize": True,
        "p": 2.0,
        "weight": 0.01,
    }
    DEFAULT_RELATION_REGULARIZER_KWARGS = {
        "normalize": True,
        "p": 2.0,
        "weight": 0.01,
    }

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
        rngs: nnx.Rngs | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Initialize a ComplEx model.

        ComplEx represents entities and relations as complex-valued embeddings.
        The score for a triple (h, r, t) is computed using the Hermitian dot product:
            score(h, r, t) = Re(<h, r, conjugate(t)>)

        Defaults:
            entity_embedding_kwargs: {"param_dtype": jnp.complex64, "embedding_init": "complex_normal"}
            relation_embedding_kwargs: {"param_dtype": jnp.complex64, "embedding_init": "complex_normal"}
            entity_regularizer: lp with {"normalize": True, "p": 2.0, "weight": 0.01}
            relation_regularizer: lp with {"normalize": True, "p": 2.0, "weight": 0.01}

        :param num_entities: Number of entities in the knowledge graph.
        :type num_entities: int
        :param num_relations: Number of relations in the knowledge graph.
        :type num_relations: int
        :param embedding_dim: Dimensionality of the embeddings (real and imaginary parts).
        :type embedding_dim: int
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
        :param rngs: Flax NNX RNGs for initialization and dropout.
        :type rngs: nnx.Rngs, optional
        :param seed: Seed used to create RNGs if none are provided.
        :type seed: int, optional

        Reference:
            Trouillon, T., Welbl, J., Riedel, S., Gaussier, E., and Bordes, A.
            "Complex Embeddings for Simple Link Prediction."
            ICML 2016.
        """
        if entity_regularizer_name is None:
            entity_regularizer_name = self.DEFAULT_ENTITY_REGULARIZER_NAME
            entity_regularizer_kwargs = {
                **self.DEFAULT_ENTITY_REGULARIZER_KWARGS,
                **(entity_regularizer_kwargs or {}),
            }
        elif entity_regularizer_name == self.DEFAULT_ENTITY_REGULARIZER_NAME:
            entity_regularizer_kwargs = {
                **self.DEFAULT_ENTITY_REGULARIZER_KWARGS,
                **(entity_regularizer_kwargs or {}),
            }

        if relation_regularizer_name is None:
            relation_regularizer_name = self.DEFAULT_RELATION_REGULARIZER_NAME
            relation_regularizer_kwargs = {
                **self.DEFAULT_RELATION_REGULARIZER_KWARGS,
                **(relation_regularizer_kwargs or {}),
            }
        elif relation_regularizer_name == self.DEFAULT_RELATION_REGULARIZER_NAME:
            relation_regularizer_kwargs = {
                **self.DEFAULT_RELATION_REGULARIZER_KWARGS,
                **(relation_regularizer_kwargs or {}),
            }

        super().__init__(
            num_entities=num_entities,
            num_relations=num_relations,
            embedding_dim=embedding_dim,
            entity_embedding_kwargs={
                **self.DEFAULT_ENTITY_EMBEDDING_KWARGS,
                **(entity_embedding_kwargs or {}),
            },
            relation_embedding_kwargs={
                **self.DEFAULT_RELATION_EMBEDDING_KWARGS,
                **(relation_embedding_kwargs or {}),
            },
            entity_regularizer_name=entity_regularizer_name,
            relation_regularizer_name=relation_regularizer_name,
            entity_regularizer_kwargs=entity_regularizer_kwargs,
            relation_regularizer_kwargs=relation_regularizer_kwargs,
            rngs=rngs,
            seed=seed,
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

    model = ComplEx(num_entities=100, num_relations=10, embedding_dim=32, seed=0)
    model.score_hrt(jax.numpy.array([[0, 3, 2]]))
