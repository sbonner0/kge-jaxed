"""Loss functions for Knowledge Graph Embedding (KGE) models.

Score convention:
    All losses in this module assume ``higher scores are better`` (more plausible).
    In other words, training should increase positive triple scores and decrease
    negative triple scores. Model interaction functions must follow this contract.
"""

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

    Assumes ``higher scores are better`` and enforces
    ``pos_scores > neg_scores + margin``.

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

    Assumes logits where larger values indicate higher plausibility. Positive
    triples are labelled 1 and negatives are labelled 0.

    :param pos_scores: Positive scores [B]
    :param neg_scores: Negative scores [B, K] or [B * K]
    :return: Scalar loss value
    """
    neg_scores = neg_scores.reshape(-1)
    logits = jnp.concatenate([pos_scores, neg_scores])
    labels = jnp.concatenate([jnp.ones_like(pos_scores), jnp.zeros_like(neg_scores)])
    loss = optax.sigmoid_binary_cross_entropy(logits, labels)
    return jnp.mean(loss)


def softplus_loss(pos_scores: jnp.ndarray, neg_scores: jnp.ndarray) -> jnp.ndarray:
    """
    Softplus loss from precomputed scores.

    Assumes ``higher scores are better`` and applies ``softplus(-s)`` for
    positives and ``softplus(s)`` for negatives.

    :param pos_scores: Positive scores [B]
    :param neg_scores: Negative scores [B, K] or [B * K]
    :return: Scalar loss value
    """
    neg_scores = neg_scores.reshape(-1)
    scores = jnp.concatenate([pos_scores, neg_scores])
    labels = jnp.concatenate([jnp.ones_like(pos_scores), -jnp.ones_like(neg_scores)])
    return jnp.mean(jax.nn.softplus(-labels * scores))


def self_adversarial_negative_sampling_loss(
    pos_scores: jnp.ndarray,
    neg_scores: jnp.ndarray,
    adversarial_temperature: float = 1.0,
    margin: float = 0.0,
) -> jnp.ndarray:
    """
    RotatE-style self-adversarial negative sampling loss (NSSA).

    This loss uses adversarial weights over negative samples:
    ``softmax(adversarial_temperature * neg_scores)`` and applies a weighted
    negative log-sigmoid term. Higher-scoring negatives therefore contribute more.

    Assumes ``higher scores are better`` and follows the RotatE/PyKEEN sign
    convention with ``margin + score``:
      - positive term: ``-log_sigmoid(margin + pos_score)``
      - negative term: ``-log_sigmoid(-(margin + neg_score))``

    Adversarial weights are always computed with ``stop_gradient`` (paper
    style), so gradients do not flow through the weight softmax itself.

    :param pos_scores: Positive scores [B]
    :param neg_scores: Negative scores [B, K]
    :param adversarial_temperature: Inverse softmax temperature for negative weights
    :param margin: Additive margin (gamma) applied to both positive and negative terms
    :return: Scalar loss value
    """
    neg_for_weights = jax.lax.stop_gradient(neg_scores)
    neg_weights = jax.nn.softmax(adversarial_temperature * neg_for_weights, axis=-1)

    pos_term = -jax.nn.log_sigmoid(margin + pos_scores)
    neg_term_unreduced = -jax.nn.log_sigmoid(-(margin + neg_scores))
    neg_term = jnp.sum(neg_weights * neg_term_unreduced, axis=-1)

    # Match RotatE-style training objective scaling: average positive/negative terms.
    return 0.5 * (jnp.mean(pos_term) + jnp.mean(neg_term))
