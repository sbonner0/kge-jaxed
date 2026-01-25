"""Loss functions for Knowledge Graph Embedding (KGE) models."""

import jax
import jax.numpy as jnp
import optax


def margin_ranking_loss(
    pos_scores: jnp.ndarray,
    neg_scores: jnp.ndarray,
    margin: float = 1.0,
) -> jnp.ndarray:
    """
    Margin ranking loss from precomputed scores.

    :param pos_scores: Positive scores [B]
    :param neg_scores: Negative scores [B, K]
    :param margin: Margin value, defaults to 1.0
    :return: Scalar loss value
    """
    loss = jnp.maximum(0.0, margin - pos_scores[:, None] + neg_scores)
    return jnp.mean(loss)


def make_margin_ranking_loss(margin: float = 1.0):
    """Factory returning a margin ranking loss with fixed margin."""

    def loss_fn(pos_scores: jnp.ndarray, neg_scores: jnp.ndarray) -> jnp.ndarray:
        return margin_ranking_loss(pos_scores, neg_scores, margin=margin)

    return loss_fn


def bce_loss(pos_scores: jnp.ndarray, neg_scores: jnp.ndarray) -> jnp.ndarray:
    """
    Binary cross-entropy loss from precomputed scores.

    :param pos_scores: Positive scores [B]
    :param neg_scores: Negative scores [B, K] or [B * K]
    :return: Scalar loss value
    """
    neg_scores = neg_scores.reshape(-1)
    logits = jnp.concatenate([pos_scores, neg_scores])
    labels = jnp.concatenate([jnp.ones_like(pos_scores), jnp.zeros_like(neg_scores)])
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels))


def softplus_loss(pos_scores: jnp.ndarray, neg_scores: jnp.ndarray) -> jnp.ndarray:
    """
    Softplus loss from precomputed scores.

    :param pos_scores: Positive scores [B]
    :param neg_scores: Negative scores [B, K] or [B * K]
    :return: Scalar loss value
    """
    neg_scores = neg_scores.reshape(-1)
    scores = jnp.concatenate([pos_scores, neg_scores])
    labels = jnp.concatenate([jnp.ones_like(pos_scores), -jnp.ones_like(neg_scores)])
    return jnp.mean(jax.nn.softplus(-labels * scores))
