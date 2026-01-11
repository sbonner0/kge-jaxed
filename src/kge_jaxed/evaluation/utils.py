"""Ranking-based evaluation metrics for KGE models."""

from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np


@partial(jax.jit, static_argnames=("num_entities", "corruption_side"))
def score_all_entities_batch(
    model,
    triples: jnp.ndarray,
    num_entities: int,
    corruption_side: Literal["head", "tail"] = "tail",
) -> jnp.ndarray:
    """
    Score a batch of triples against all possible entity replacements.

    :param model: KGE model with score_hrt method
    :type model: Any
    :param triples: Batch of triples [B, 3]
    :type triples: jnp.ndarray
    :param num_entities: Total number of entities
    :type num_entities: int
    :param corruption_side: Which side to corrupt ("head" or "tail")
    :type corruption_side: Literal["head", "tail"]
    :return: Scores for all entity replacements [B, num_entities]
    :rtype: jnp.ndarray
    """
    h = triples[:, 0]
    r = triples[:, 1]
    t = triples[:, 2]
    batch_size = triples.shape[0]
    all_entities = jnp.arange(num_entities, dtype=triples.dtype)

    if corruption_side == "tail":
        batch_h = jnp.broadcast_to(h[:, None], (batch_size, num_entities))
        batch_r = jnp.broadcast_to(r[:, None], (batch_size, num_entities))
        batch_t = jnp.broadcast_to(all_entities[None, :], (batch_size, num_entities))
    else:  # head
        batch_h = jnp.broadcast_to(all_entities[None, :], (batch_size, num_entities))
        batch_r = jnp.broadcast_to(r[:, None], (batch_size, num_entities))
        batch_t = jnp.broadcast_to(t[:, None], (batch_size, num_entities))

    corrupted_batch = jnp.stack([batch_h, batch_r, batch_t], axis=2)
    flat_triples = corrupted_batch.reshape(-1, 3)
    scores = model.score_hrt(flat_triples)
    return scores.reshape(batch_size, num_entities)


def compute_group_ranks(
    scores: jnp.ndarray,
    true_indices: np.ndarray,
    filtered_indices: np.ndarray | None = None,
) -> jnp.ndarray:
    """
    Compute ranks for multiple true entities sharing the same score vector.

    :param scores: Scores for all entities [num_entities]
    :type scores: jnp.ndarray
    :param true_indices: True entity indices for this group [K]
    :type true_indices: np.ndarray
    :param filtered_indices: Optional filtered indices [F]
    :type filtered_indices: np.ndarray | None
    :return: Ranks for each true entity [K]
    :rtype: jnp.ndarray
    """
    if true_indices.size == 0:
        return jnp.array([], dtype=jnp.int32)

    true_indices_jax = jnp.asarray(true_indices, dtype=jnp.int32)
    true_scores = scores[true_indices_jax]
    better_counts = jnp.sum(scores[None, :] > true_scores[:, None], axis=1)

    if filtered_indices is not None and filtered_indices.size > 0:
        filtered_scores = scores[jnp.asarray(filtered_indices, dtype=jnp.int32)]
        better_filtered = jnp.sum(filtered_scores[None, :] > true_scores[:, None], axis=1)
    else:
        better_filtered = 0

    ranks = better_counts + 1 - better_filtered
    return jnp.maximum(ranks, 1)
