import jax.numpy as jnp
from flax import nnx

from kge_jaxed.models.rotate import RotatE


def _set_embedding_weights(model: RotatE, entity_w: jnp.ndarray, relation_w: jnp.ndarray) -> None:
    model.entity_embedding.emb.embedding.set_value(jnp.asarray(entity_w))
    model.relation_embedding.emb.embedding.set_value(jnp.asarray(relation_w))


def test_rotate_constructor_defaults() -> None:
    model = RotatE(num_entities=6, num_relations=3, rngs=nnx.Rngs(0))

    assert model.entity_embedding_dim == RotatE.DEFAULT_ENTITY_EMBEDDING_SIZE
    assert model.relation_embedding_dim == RotatE.DEFAULT_ENTITY_EMBEDDING_SIZE
    assert model.entity_weights().dtype == jnp.complex64
    assert model.relation_weights().dtype == jnp.complex64
    assert model.entity_constrainer is None
    assert model.relation_constrainer is not None
    relation_modulus = jnp.abs(model.relation_weights())
    assert jnp.allclose(relation_modulus, jnp.ones_like(relation_modulus), atol=1e-6)


def test_rotate_score_hrt_matches_l2_distance() -> None:
    model = RotatE(num_entities=3, num_relations=2, entity_embedding_dim=2, norm=2, rngs=nnx.Rngs(0))
    _set_embedding_weights(
        model,
        entity_w=jnp.array(
            [
                [1.0 + 0.0j, 0.0 + 1.0j],  # id 0
                [1.0 + 1.0j, 2.0 + 0.0j],  # id 1
                [0.0 + 1.0j, 0.0 + 1.0j],  # id 2
            ],
            dtype=jnp.complex64,
        ),
        relation_w=jnp.array(
            [
                [0.0 + 1.0j, 1.0 + 0.0j],  # id 0
                [1.0 + 0.0j, 0.0 - 1.0j],  # id 1
            ],
            dtype=jnp.complex64,
        ),
    )

    triples = jnp.array([[0, 0, 2], [1, 1, 0]], dtype=jnp.int32)
    scores = model.score_hrt(triples)

    expected = jnp.array([0.0, -jnp.sqrt(10.0)], dtype=jnp.float32)
    assert jnp.allclose(scores, expected)


def test_rotate_apply_constraints_unit_modulus() -> None:
    model = RotatE(num_entities=2, num_relations=2, entity_embedding_dim=2, rngs=nnx.Rngs(0))
    _set_embedding_weights(
        model,
        entity_w=jnp.array(
            [
                [1.0 + 1.0j, 2.0 + 2.0j],
                [3.0 + 3.0j, 4.0 + 4.0j],
            ],
            dtype=jnp.complex64,
        ),
        relation_w=jnp.array(
            [
                [3.0 + 4.0j, 1.0 + 1.0j],
                [2.0 - 2.0j, 5.0 + 12.0j],
            ],
            dtype=jnp.complex64,
        ),
    )

    model.apply_constraints()

    relation_modulus = jnp.abs(model.relation_weights())
    assert jnp.allclose(relation_modulus, jnp.ones_like(relation_modulus), atol=1e-6)


def test_rotate_apply_constraints_leaves_entities_unchanged() -> None:
    model = RotatE(num_entities=2, num_relations=2, entity_embedding_dim=2, rngs=nnx.Rngs(0))
    _set_embedding_weights(
        model,
        entity_w=jnp.array(
            [
                [1.0 + 2.0j, 3.0 + 4.0j],
                [5.0 - 6.0j, 7.0 + 8.0j],
            ],
            dtype=jnp.complex64,
        ),
        relation_w=jnp.array(
            [
                [3.0 + 4.0j, 1.0 + 1.0j],
                [2.0 - 2.0j, 5.0 + 12.0j],
            ],
            dtype=jnp.complex64,
        ),
    )

    before_entities = model.entity_weights().copy()
    model.apply_constraints()
    after_entities = model.entity_weights()

    assert jnp.allclose(after_entities, before_entities, atol=1e-7)
