import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from kge_jaxed.negative_sampling.uniform_negative_sampling import (
    uniform_balanced_sampler,
    uniform_corruptions,
)
from kge_jaxed.registry import models

NUM_ENTITIES = 7
NUM_RELATIONS = 3
TRIPLES = jnp.array([[0, 0, 1], [2, 1, 3], [4, 2, 5], [6, 0, 2]], dtype=jnp.int32)
K = 5


@pytest.mark.parametrize("model_name", ["transe", "trans", "distmult", "complex", "rotate"])
def test_score_with_corruptions_matches_score_hrt(model_name: str) -> None:
    model = models.get(model_name)(
        num_entities=NUM_ENTITIES,
        num_relations=NUM_RELATIONS,
        entity_embedding_dim=12,
        rngs=nnx.Rngs(0),
    )
    key = jax.random.PRNGKey(0)
    side, neg_ent = uniform_corruptions(TRIPLES, NUM_ENTITIES, k=K, key=key)

    pos_scores, neg_scores = model.score_with_corruptions(TRIPLES, neg_ent, side)

    expected_pos = model.score_hrt(TRIPLES)
    neg_triples = uniform_balanced_sampler(TRIPLES, NUM_ENTITIES, k=K, key=key)
    expected_neg = model.score_hrt(neg_triples.reshape(-1, 3)).reshape(TRIPLES.shape[0], K)

    assert pos_scores.shape == (TRIPLES.shape[0],)
    assert neg_scores.shape == (TRIPLES.shape[0], K)
    assert jnp.allclose(pos_scores, expected_pos, atol=1e-6)
    assert jnp.allclose(neg_scores, expected_neg, atol=1e-6)
