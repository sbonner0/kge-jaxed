"""Utilities for ranking-based KGE evaluation."""

from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray


@partial(jax.jit, static_argnames=("num_entities", "corruption_side"))
def score_all_entities_batch(
    model,
    triples: jnp.ndarray,
    num_entities: int,
    corruption_side: Literal["head", "tail"] = "tail",
) -> jnp.ndarray:
    """
    Score a batch of triples against all possible entity replacements.
    This will return an array of shape [B, num_entities], i.e. what is the score of each entity used to
    complete the batch of partial triples.

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
    all_scores: jnp.ndarray | NDArray[np.floating],
    target_ids: NDArray[np.integer],
    filtered_ids: NDArray[np.integer] | None = None,
) -> NDArray[np.int32]:
    """
    Compute ranks for multiple true entities sharing the same score vector.

    :param all_scores: Scores for all entities [num_entities]
    :type all_scores: jnp.ndarray
    :param target_ids: True entity indices for this group [K]
    :type target_ids: np.ndarray
    :param filtered_ids: Optional filtered indices [F]
    :type filtered_ids: np.ndarray | None
    :return: Ranks for each true entity [K]
    :rtype: np.ndarray
    """
    all_scores = np.asarray(all_scores, dtype=np.float32)
    target_ids = np.asarray(target_ids, dtype=np.int32)
    target_scores = all_scores[target_ids]

    # Standard optimistic rank: count candidates with strictly higher score.
    higher_counts = np.sum(all_scores[None, :] > target_scores[:, None], axis=1)

    # Filtered evaluation removes other known positives for the same query. We keep
    # the current target implicitly because it is never strictly higher than itself.
    if filtered_ids is None or filtered_ids.size == 0:
        return higher_counts.astype(np.int32) + 1

    filtered_ids = np.unique(np.asarray(filtered_ids, dtype=np.int32))
    filtered_scores = all_scores[filtered_ids]
    filtered_higher_counts = np.sum(filtered_scores[None, :] > target_scores[:, None], axis=1)

    return np.maximum(higher_counts - filtered_higher_counts + 1, 1).astype(np.int32)
