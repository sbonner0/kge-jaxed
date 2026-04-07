import jax.numpy as jnp
from flax import nnx

from kge_jaxed.models.transe import TransE


def _set_embedding_weights(model: TransE, entity_w: jnp.ndarray, relation_w: jnp.ndarray) -> None:
    model.entity_embedding.emb.embedding.set_value(jnp.asarray(entity_w))
    model.relation_embedding.emb.embedding.set_value(jnp.asarray(relation_w))


def test_transe_constructor_defaults() -> None:
    model = TransE(num_entities=8, num_relations=3, rngs=nnx.Rngs(0))

    assert model.entity_embedding_dim == TransE.DEFAULT_EMBEDDING_DIM
    assert model.relation_embedding_dim == TransE.DEFAULT_EMBEDDING_DIM
    assert model.entity_constrainer is not None
    assert model.relation_constrainer is None


def test_transe_initial_constraints_unit_norm() -> None:
    model = TransE(num_entities=8, num_relations=3, entity_embedding_dim=5, rngs=nnx.Rngs(0))

    entity_norms = jnp.linalg.norm(model.entity_weights(), axis=1)
    relation_norms = jnp.linalg.norm(model.relation_weights(), axis=1)
    assert jnp.allclose(entity_norms, jnp.ones_like(entity_norms), atol=1e-6)
    assert jnp.allclose(relation_norms, jnp.ones_like(relation_norms), atol=1e-6)


def test_transe_score_hrt_matches_l1_distance() -> None:
    model = TransE(num_entities=3, num_relations=2, entity_embedding_dim=2, norm=1, rngs=nnx.Rngs(0))
    _set_embedding_weights(
        model,
        entity_w=jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=jnp.float32),
        relation_w=jnp.array([[1.0, 1.0], [0.0, 1.0]], dtype=jnp.float32),
    )

    triples = jnp.array([[0, 0, 2], [1, 1, 0]], dtype=jnp.int32)
    scores = model.score_hrt(triples)

    expected = jnp.array([-1.0, -3.0], dtype=jnp.float32)
    assert jnp.allclose(scores, expected)


def test_transe_apply_constraints_default_entity_unit_norm() -> None:
    model = TransE(num_entities=2, num_relations=2, entity_embedding_dim=3, rngs=nnx.Rngs(0))
    _set_embedding_weights(
        model,
        entity_w=jnp.array([[3.0, 4.0, 0.0], [1.0, 2.0, 2.0]], dtype=jnp.float32),
        relation_w=jnp.array([[2.0, 0.0, 0.0], [0.0, 5.0, 12.0]], dtype=jnp.float32),
    )

    model.apply_constraints()

    entity_norms = jnp.linalg.norm(model.entity_weights(), axis=1)
    relation_norms = jnp.linalg.norm(model.relation_weights(), axis=1)
    assert jnp.allclose(entity_norms, jnp.ones_like(entity_norms), atol=1e-6)
    assert jnp.allclose(relation_norms, jnp.array([2.0, 13.0], dtype=jnp.float32), atol=1e-6)


def test_transe_apply_constraints_explicit_relation_unit_norm() -> None:
    model = TransE(
        num_entities=2,
        num_relations=2,
        entity_embedding_dim=3,
        relation_constrainer_kwargs={"name": "unit_norm"},
        rngs=nnx.Rngs(0),
    )
    _set_embedding_weights(
        model,
        entity_w=jnp.array([[3.0, 4.0, 0.0], [1.0, 2.0, 2.0]], dtype=jnp.float32),
        relation_w=jnp.array([[2.0, 0.0, 0.0], [0.0, 5.0, 12.0]], dtype=jnp.float32),
    )

    model.apply_constraints()

    relation_norms = jnp.linalg.norm(model.relation_weights(), axis=1)
    assert jnp.allclose(relation_norms, jnp.ones_like(relation_norms), atol=1e-6)
