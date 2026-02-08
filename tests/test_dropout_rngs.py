import jax
import jax.numpy as jnp
from flax import nnx

from kge_jaxed.models.transe import TransE
from kge_jaxed.rngs import make_model_rngs


def test_dropout_rng_determinism() -> None:
    model = TransE(
        num_entities=10,
        num_relations=5,
        entity_embedding_dim=16,
        rngs=make_model_rngs(0),
        entity_embedding_kwargs={"dropout_rate": 0.5},
        relation_embedding_kwargs={"dropout_rate": 0.5},
    )

    triples = jnp.array([[1, 2, 3], [4, 0, 5], [2, 1, 0], [3, 4, 6]], dtype=jnp.int32)

    key = jax.random.PRNGKey(123)
    out_a = model.score_hrt(triples, dropout_rngs=nnx.Rngs(dropout=key))
    out_b = model.score_hrt(triples, dropout_rngs=nnx.Rngs(dropout=key))

    key2 = jax.random.PRNGKey(456)
    out_c = model.score_hrt(triples, dropout_rngs=nnx.Rngs(dropout=key2))

    assert jnp.array_equal(out_a, out_b)
    assert not jnp.array_equal(out_a, out_c)
