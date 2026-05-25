import jax.numpy as jnp
from flax import nnx
from jax import Array

from kge_jaxed.models.base_kge import BaseKGE


class TranS(BaseKGE):
    """TranS model for knowledge graph embedding."""

    DEFAULT_EMBEDDING_DIM = 50
    DEFAULT_NORM = 1
    DEFAULT_ENTITY_EMBEDDING_KWARGS = {"embedding_init": "xavier"}
    DEFAULT_RELATION_EMBEDDING_KWARGS = {"embedding_init": "xavier_uniform_norm"}
    DEFAULT_ENTITY_CONSTRAINER_KWARGS = {}
    DEFAULT_RELATION_CONSTRAINER_KWARGS = {}

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
        Initialize a TranS model.

        TranS replaces the single relation vector used by traditional translation
        models with a synthetic relation representation. Each entity has a main
        and adjoint vector, and each relation has a main vector plus head- and
        tail-conditioned adjoint vectors:
            score(h, r, t) = -||h * (t_tilde + 1) - t * (h_tilde + 1)
                              + r_bar * h + r + r_hat * t||_p

        Defaults:
            entity_embedding_dim: 50 per representation
            norm: 1
            entity_embedding_kwargs: {"embedding_init": "xavier"}
            relation_embedding_kwargs: {"embedding_init": "xavier_uniform_norm"}
            entity_constrainer: None
            relation_constrainer: None

        :param num_entities: Number of entities in the knowledge graph.
        :type num_entities: int
        :param num_relations: Number of relations in the knowledge graph.
        :type num_relations: int
        :param entity_embedding_dim: Dimensionality of each entity/relation representation.
        :type entity_embedding_dim: int
        :param relation_embedding_dim: Optional dimensionality of each relation representation.
            If provided, it must equal ``entity_embedding_dim`` because TranS combines
            entity and relation component vectors elementwise.
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
            Zhang, X., Yang, Q., & Xu, D.
            "TranS: Transition-based Knowledge Graph Embedding with Synthetic Relation Representation."
            EMNLP 2022.
        """
        self.num_entity_representations = 2
        self.num_relation_representations = 3
        self.representation_dim = int(entity_embedding_dim)
        if relation_embedding_dim is not None and int(relation_embedding_dim) != self.representation_dim:
            raise ValueError("TranS requires relation_embedding_dim to match entity_embedding_dim")

        super().__init__(
            num_entities=num_entities,
            num_relations=num_relations,
            entity_embedding_dim=self.representation_dim * self.num_entity_representations,
            relation_embedding_dim=self.representation_dim * self.num_relation_representations,
            entity_embedding_kwargs=entity_embedding_kwargs,
            relation_embedding_kwargs=relation_embedding_kwargs,
            entity_regularizer_kwargs=entity_regularizer_kwargs,
            relation_regularizer_kwargs=relation_regularizer_kwargs,
            entity_constrainer_kwargs=entity_constrainer_kwargs,
            relation_constrainer_kwargs=relation_constrainer_kwargs,
            rngs=rngs,
        )
        self.entity_embedding_dim = self.representation_dim
        self.relation_embedding_dim = self.representation_dim
        self.norm = norm

    def _split_embedding(self, x: Array, num_representations: int) -> list[Array]:
        return jnp.split(x, num_representations, axis=-1)

    def interaction_function(self, h: Array, r: Array, t: Array) -> Array:
        """
        Compute TranS scores for a batch of triples.

        :param h: Head entity embeddings of shape [B, 2D].
        :type h: Array
        :param r: Relation embeddings of shape [B, 3D].
        :type r: Array
        :param t: Tail entity embeddings of shape [B, 2D].
        :type t: Array
        :return: Scores of shape [B], higher is better.
        :rtype: Array
        """
        # using notation and variable names from the original paper
        h1, h_tilde = self._split_embedding(h, self.num_entity_representations)
        t1, t_tilde = self._split_embedding(t, self.num_entity_representations)
        r1, r_hat, r_bar = self._split_embedding(r, self.num_relation_representations)

        # Unit vector of ones for the elementwise addition in the paper's formulation
        unit_vector = jnp.ones_like(h1)

        R_h = h1 * (t_tilde + unit_vector)
        R_t = t1 * (h_tilde + unit_vector)
        R_r = r_bar * h1 + r1 + r_hat * t1

        score = R_h - R_t + R_r
        return -jnp.linalg.norm(score, ord=self.norm, axis=-1)


if __name__ == "__main__":
    import jax

    model = TranS(num_entities=100, num_relations=10, entity_embedding_dim=32, rngs=nnx.Rngs(0))
    print(model.score_hrt(jax.numpy.array([[0, 3, 2]])))
