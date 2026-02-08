import jax
import jax.numpy as jnp
from flax import nnx

from kge_jaxed.models.transe import TransE
from kge_jaxed.rngs import make_model_rngs


def _param_leaves(model: nnx.Module) -> list[jax.Array]:
    state = nnx.state(model, nnx.Param)
    return jax.tree_util.tree_leaves(state)


def test_model_init_same_seed() -> None:
    model_a = TransE(num_entities=10, num_relations=5, entity_embedding_dim=8, rngs=make_model_rngs(0))
    model_b = TransE(num_entities=10, num_relations=5, entity_embedding_dim=8, rngs=make_model_rngs(0))

    leaves_a = _param_leaves(model_a)
    leaves_b = _param_leaves(model_b)

    assert len(leaves_a) == len(leaves_b)
    assert all(jnp.array_equal(a, b) for a, b in zip(leaves_a, leaves_b))


def test_model_init_different_seed() -> None:
    model_a = TransE(num_entities=10, num_relations=5, entity_embedding_dim=8, rngs=make_model_rngs(0))
    model_b = TransE(num_entities=10, num_relations=5, entity_embedding_dim=8, rngs=make_model_rngs(1))

    leaves_a = _param_leaves(model_a)
    leaves_b = _param_leaves(model_b)

    assert len(leaves_a) == len(leaves_b)
    assert any(not jnp.array_equal(a, b) for a, b in zip(leaves_a, leaves_b))
