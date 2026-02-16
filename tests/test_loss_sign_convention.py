import jax.numpy as jnp
import pytest

from kge_jaxed.loss_functions.losses import (
    bce_loss,
    margin_ranking_loss,
    softplus_loss,
)


@pytest.mark.parametrize(
    "loss_fn",
    [
        margin_ranking_loss,
        bce_loss,
        softplus_loss,
    ],
)
def test_loss_decreases_when_positive_score_increases(loss_fn) -> None:
    pos_low = jnp.array([0.0], dtype=jnp.float32)
    pos_high = jnp.array([0.25], dtype=jnp.float32)
    neg = jnp.array([[0.0]], dtype=jnp.float32)

    low_loss = float(loss_fn(pos_low, neg))
    high_loss = float(loss_fn(pos_high, neg))

    assert high_loss < low_loss


@pytest.mark.parametrize(
    "loss_fn",
    [
        margin_ranking_loss,
        bce_loss,
        softplus_loss,
    ],
)
def test_loss_increases_when_negative_score_increases(loss_fn) -> None:
    pos = jnp.array([0.0], dtype=jnp.float32)
    neg_low = jnp.array([[0.0]], dtype=jnp.float32)
    neg_high = jnp.array([[0.25]], dtype=jnp.float32)

    low_loss = float(loss_fn(pos, neg_low))
    high_loss = float(loss_fn(pos, neg_high))

    assert high_loss > low_loss
