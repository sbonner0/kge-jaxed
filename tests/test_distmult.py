import jax.numpy as jnp
from flax import nnx

from kge_jaxed.models.distmult import DistMult


def _set_embedding_weights(model: DistMult, entity_w: jnp.ndarray, relation_w: jnp.ndarray) -> None:
    model.entity_embedding.emb.embedding.set_value(jnp.asarray(entity_w))
    model.relation_embedding.emb.embedding.set_value(jnp.asarray(relation_w))


def test_distmult_constructor_defaults() -> None:
    model = DistMult(num_entities=8, num_relations=3, rngs=nnx.Rngs(0))

    assert model.entity_embedding_dim == DistMult.DEFAULT_EMBEDDING_DIM
    assert model.relation_embedding_dim == DistMult.DEFAULT_EMBEDDING_DIM
    assert model.entity_constrainer is not None
    assert model.relation_constrainer is None
    assert model.relation_regularizer is not None
    assert model.relation_regularizer_weight == 0.1


def test_distmult_score_hrt_matches_trilinear_dot() -> None:
    model = DistMult(num_entities=3, num_relations=2, entity_embedding_dim=2, rngs=nnx.Rngs(0))
    _set_embedding_weights(
        model,
        entity_w=jnp.array([[1.0, 2.0], [3.0, 4.0], [1.0, 1.0]], dtype=jnp.float32),
        relation_w=jnp.array([[2.0, 1.0], [1.0, -1.0]], dtype=jnp.float32),
    )

    triples = jnp.array([[0, 0, 1], [2, 1, 1]], dtype=jnp.int32)
    scores = model.score_hrt(triples)

    expected = jnp.array([14.0, -1.0], dtype=jnp.float32)
    assert jnp.allclose(scores, expected)


def test_distmult_apply_constraints_affects_entities_only() -> None:
    model = DistMult(num_entities=2, num_relations=2, entity_embedding_dim=3, rngs=nnx.Rngs(0))
    entity_w = jnp.array([[3.0, 4.0, 0.0], [1.0, 2.0, 2.0]], dtype=jnp.float32)
    relation_w = jnp.array([[2.0, 0.0, 0.0], [0.0, 5.0, 12.0]], dtype=jnp.float32)
    _set_embedding_weights(model, entity_w=entity_w, relation_w=relation_w)

    model.apply_constraints()

    entity_norms = jnp.linalg.norm(model.entity_weights(), axis=1)
    assert jnp.allclose(entity_norms, jnp.ones_like(entity_norms), atol=1e-6)
    assert jnp.allclose(model.relation_weights(), relation_w)
