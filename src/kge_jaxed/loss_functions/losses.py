"""Loss functions for Knowledge Graph Embedding (KGE) models."""

import jax.numpy as jnp
import optax
from flax import nnx


def margin_ranking_loss(model, batch, neg_batch, margin=1.0, *, dropout_rngs: nnx.Rngs | None = None):
    """
    Margin Ranking Loss for KGE models with multiple negatives per positive.

    :param model: KGE model with score_hrt method
    :param batch: Positive triples [B, 3]
    :param neg_batch: Negative triples [B*K, 3] (reshaped from [B, K, 3])
    :param margin: Margin value, defaults to 1.0
    :param dropout_rngs: Optional rngs for dropout during scoring
    :return: Scalar loss value
    """
    B = batch.shape[0]
    K = neg_batch.shape[0] // B

    # Score positives: [B]
    if dropout_rngs is None:
        pos_scores = model.score_hrt(batch)
        neg_scores = model.score_hrt(neg_batch).reshape(B, K)
    else:
        pos_scores = model.score_hrt(batch, dropout_rngs=dropout_rngs)
        neg_scores = model.score_hrt(neg_batch, dropout_rngs=dropout_rngs).reshape(B, K)

    # Broadcast pos_scores [B] -> [B, 1] to compare with neg_scores [B, K]
    loss = jnp.maximum(0, margin - pos_scores[:, None] + neg_scores)

    return jnp.mean(loss)


def bce_loss(model, batch, neg_batch, *, dropout_rngs: nnx.Rngs | None = None):
    """
    Binary Cross-Entropy loss for KGE models.

    :param model: _description_
    :type model: _type_
    :param batch: _description_
    :type batch: _type_
    :param neg_batch: _description_
    :type neg_batch: _type_
    :param dropout_rngs: Optional rngs for dropout during scoring
    :return: _description_
    :rtype: _type_
    """
    if dropout_rngs is None:
        pos_scores = model.score_hrt(batch)
        neg_scores = model.score_hrt(neg_batch)
    else:
        pos_scores = model.score_hrt(batch, dropout_rngs=dropout_rngs)
        neg_scores = model.score_hrt(neg_batch, dropout_rngs=dropout_rngs)

    logits = jnp.concatenate([pos_scores, neg_scores])
    labels = jnp.concatenate([jnp.ones_like(pos_scores), jnp.zeros_like(neg_scores)])
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels))
