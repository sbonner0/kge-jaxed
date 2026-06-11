"""JIT-compiled training step helpers."""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx

from kge_jaxed.models.base_kge import BaseKGE
from kge_jaxed.negative_sampling.uniform_negative_sampling import (
    uniform_corruptions,
)


def _touched_ids(
    pos_batch: jnp.ndarray,
    neg_ent: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    entity_ids = jnp.concatenate([pos_batch[:, 0], pos_batch[:, 2], neg_ent.reshape(-1)])
    relation_ids = pos_batch[:, 1]
    return entity_ids, relation_ids


@partial(
    nnx.jit,
    static_argnames=(
        "num_negative_samples",
        "num_entities",
        "loss_fn",
    ),
)
def train_step_fn(
    model: BaseKGE,
    optimizer: nnx.Optimizer,
    step_key: jax.Array,
    batch: jnp.ndarray,
    num_negative_samples: int,
    num_entities: int,
    loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    """Run one JIT-compiled optimization step."""

    def loss_on_model(m: BaseKGE) -> jnp.ndarray:
        use_dropout = bool(getattr(m, "uses_dropout", False))
        if use_dropout:
            neg_key, dropout_key = jax.random.split(step_key, 2)
            dropout_rngs = nnx.Rngs(dropout=dropout_key)
        else:
            neg_key = step_key
            dropout_rngs = None

        side, neg_ent = uniform_corruptions(
            triples=batch,
            num_entities=num_entities,
            k=num_negative_samples,
            key=neg_key,
        )
        pos_scores, neg_scores = m.score_with_corruptions(batch, neg_ent, side, dropout_rngs=dropout_rngs)
        entity_ids, relation_ids = _touched_ids(batch, neg_ent)
        loss = loss_fn(pos_scores, neg_scores)
        return loss + m.regularization_loss_for_ids(entity_ids, relation_ids)

    loss, grads = nnx.value_and_grad(loss_on_model)(model)
    optimizer.update(model, grads)
    model.apply_constraints()
    return loss
