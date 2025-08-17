"""Loss functions for Knowledge Graph Embedding (KGE) models."""

import jax.numpy as jnp
import optax

from kge_jaxed.loss_functions.loss_utils import loss_fn_wrapper
from kge_jaxed.registries import LOSSES


@LOSSES.register("mrl")
def margin_ranking_loss(model, batch, neg_batch, margin=1.0):
    """Margin Ranking Loss for KGE models.

    :param model: _description_
    :type model: _type_
    :param batch: _description_
    :type batch: _type_
    :param neg_batch: _description_
    :type neg_batch: _type_
    :param margin: _description_, defaults to 1.0
    :type margin: float, optional
    :return: _description_
    :rtype: _type_
    """
    pos_scores, neg_scores = loss_fn_wrapper(model, batch, neg_batch)
    return jnp.mean(jnp.maximum(0, margin - pos_scores + neg_scores))


@LOSSES.register("bce")
def bce_loss(
    model,
    batch,
    neg_batch,
):
    """
    Binary Cross-Entropy loss for KGE models.

    :param model: _description_
    :type model: _type_
    :param batch: _description_
    :type batch: _type_
    :param neg_batch: _description_
    :type neg_batch: _type_
    :return: _description_
    :rtype: _type_
    """
    pos_scores, neg_scores = loss_fn_wrapper(model, batch, neg_batch)
    logits = jnp.concatenate([pos_scores, neg_scores])
    labels = jnp.concatenate([jnp.ones_like(pos_scores), jnp.zeros_like(neg_scores)])
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels))
