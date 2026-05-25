import jax.numpy as jnp
import pytest
from flax import nnx

from kge_jaxed.models.trans import TranS


def _set_embedding_weights(model: TranS, entity_w: jnp.ndarray, relation_w: jnp.ndarray) -> None:
    model.entity_embedding.emb.embedding.set_value(jnp.asarray(entity_w))
    model.relation_embedding.emb.embedding.set_value(jnp.asarray(relation_w))


def test_trans_constructor_uses_component_dimension_publicly() -> None:
    model = TranS(num_entities=8, num_relations=3, entity_embedding_dim=5, rngs=nnx.Rngs(0))

    assert model.entity_embedding_dim == 5
    assert model.relation_embedding_dim == 5
    assert model.representation_dim == 5
    assert model.entity_embedding.emb.embedding.shape == (8, 10)
    assert model.relation_embedding.emb.embedding.shape == (3, 15)
    assert model.entity_constrainer is None
    assert model.relation_constrainer is None


def test_trans_rejects_mismatched_relation_component_dimension() -> None:
    with pytest.raises(ValueError, match="relation_embedding_dim"):
        TranS(
            num_entities=8,
            num_relations=3,
            entity_embedding_dim=5,
            relation_embedding_dim=4,
            rngs=nnx.Rngs(0),
        )


def test_trans_allows_matching_relation_component_dimension() -> None:
    model = TranS(
        num_entities=8,
        num_relations=3,
        entity_embedding_dim=5,
        relation_embedding_dim=5,
        rngs=nnx.Rngs(0),
    )

    assert model.entity_embedding.emb.embedding.shape == (8, 10)
    assert model.relation_embedding.emb.embedding.shape == (3, 15)


def test_trans_score_hrt_matches_paper_formula_with_unit_vector() -> None:
    model = TranS(num_entities=3, num_relations=2, entity_embedding_dim=2, norm=1, rngs=nnx.Rngs(0))
    _set_embedding_weights(
        model,
        entity_w=jnp.array(
            [
                [1.0, 2.0, 0.5, 1.0],
                [3.0, 1.0, 2.0, 0.5],
                [0.0, 1.0, 1.0, 2.0],
            ],
            dtype=jnp.float32,
        ),
        relation_w=jnp.array(
            [
                [0.5, 1.0, 2.0, 0.5, 1.5, 2.0],
                [1.0, -1.0, 0.0, 2.0, 0.5, -0.5],
            ],
            dtype=jnp.float32,
        ),
    )

    triples = jnp.array([[0, 0, 1], [2, 1, 0]], dtype=jnp.int32)
    scores = model.score_hrt(triples)

    expected = jnp.array([-13.0, -2.5], dtype=jnp.float32)
    assert jnp.allclose(scores, expected)
