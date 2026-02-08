import jax.numpy as jnp
from flax import nnx

from kge_jaxed.models.complex import ComplEx


def _set_embedding_weights(model: ComplEx, entity_w: jnp.ndarray, relation_w: jnp.ndarray) -> None:
    model.entity_embedding.emb.embedding.set_value(jnp.asarray(entity_w))
    model.relation_embedding.emb.embedding.set_value(jnp.asarray(relation_w))


def test_complex_constructor_defaults() -> None:
    model = ComplEx(num_entities=6, num_relations=3, entity_embedding_dim=4, rngs=nnx.Rngs(0))

    assert model.entity_embedding_dim == 4
    assert model.relation_embedding_dim == 4
    assert model.entity_weights().dtype == jnp.complex64
    assert model.relation_weights().dtype == jnp.complex64
    assert model.entity_regularizer is not None
    assert model.relation_regularizer is not None
    assert model.entity_regularizer_weight == 0.01
    assert model.relation_regularizer_weight == 0.01


def test_complex_score_hrt_matches_hermitian_dot() -> None:
    model = ComplEx(num_entities=3, num_relations=2, entity_embedding_dim=2, rngs=nnx.Rngs(0))
    _set_embedding_weights(
        model,
        entity_w=jnp.array(
            [
                [1.0 + 2.0j, 2.0 + 0.0j],  # id 0
                [2.0 + 1.0j, 1.0 - 1.0j],  # id 1
                [1.0 + 0.0j, 1.0 + 1.0j],  # id 2
            ],
            dtype=jnp.complex64,
        ),
        relation_w=jnp.array(
            [
                [1.0 + 0.0j, 0.0 + 1.0j],  # id 0
                [1.0 + 1.0j, 1.0 + 0.0j],  # id 1
            ],
            dtype=jnp.complex64,
        ),
    )

    triples = jnp.array([[0, 0, 1], [2, 1, 0]], dtype=jnp.int32)
    scores = model.score_hrt(triples)

    expected = jnp.array([2.0, 5.0], dtype=jnp.float32)
    assert jnp.allclose(scores, expected)


def test_complex_score_output_is_real() -> None:
    model = ComplEx(num_entities=4, num_relations=2, entity_embedding_dim=3, rngs=nnx.Rngs(0))
    triples = jnp.array([[0, 0, 1], [2, 1, 3]], dtype=jnp.int32)
    scores = model.score_hrt(triples)

    assert not jnp.iscomplexobj(scores)
