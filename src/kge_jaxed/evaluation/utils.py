"""Ranking-based evaluation metrics for KGE models."""

from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np


@partial(jax.jit, static_argnames=("num_entities", "corruption_side"))
def score_all_entities(
    model,
    triple: jnp.ndarray,
    num_entities: int,
    corruption_side: Literal["head", "tail"] = "tail",
) -> jnp.ndarray:
    """
    Score a single triple against all possible entity replacements.

    :param model: KGE model with score_hrt method
    :param triple: Single triple [3] (head, relation, tail)
    :param num_entities: Total number of entities
    :param corruption_side: Which side to corrupt ("head" or "tail")
    :return: Scores for all entity replacements [num_entities]
    """
    h, r, t = triple[0], triple[1], triple[2]

    # Create all possible corruptions
    all_entities = jnp.arange(num_entities, dtype=triple.dtype)

    if corruption_side == "tail":
        batch_h = jnp.full(num_entities, h, dtype=triple.dtype)
        batch_r = jnp.full(num_entities, r, dtype=triple.dtype)
        batch_t = all_entities
    else:  # head
        batch_h = all_entities
        batch_r = jnp.full(num_entities, r, dtype=triple.dtype)
        batch_t = jnp.full(num_entities, t, dtype=triple.dtype)

    # Stack into batch of triples [N, 3]
    corrupted_batch = jnp.stack([batch_h, batch_r, batch_t], axis=1)

    # Score all at once
    scores = model.score_hrt(corrupted_batch)

    return scores


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
    :param triples: Batch of triples [B, 3]
    :param num_entities: Total number of entities
    :param corruption_side: Which side to corrupt ("head" or "tail")
    :return: Scores for all entity replacements [B, num_entities]
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


def compute_rank(
    scores: jnp.ndarray,
    true_idx: jnp.ndarray,  # Changed from int to jnp.ndarray
    filtered_indices: jnp.ndarray,
) -> jnp.ndarray:  # Changed return type to jnp.ndarray
    """
    Compute rank of true entity given scores.

    :param scores: Scores for all entities [num_entities]
    :param true_idx: Index of the true entity (scalar array)
    :param filtered_indices: Indices of other true entities to filter out
    :return: Rank of true entity (1-indexed, scalar array)
    """
    true_score = scores[true_idx]

    # Count how many entities scored better (strict inequality)
    better_than_true = scores > true_score
    rank = jnp.sum(better_than_true) + 1

    # Filtered setting: remove other known true entities from ranking
    # Only apply filtering if we have filtered indices
    num_filtered = filtered_indices.shape[0]

    def apply_filtering(rank, filtered_indices):
        filtered_scores = scores[filtered_indices]
        better_filtered = jnp.sum(filtered_scores > true_score)
        return rank - better_filtered

    def no_filtering(rank, filtered_indices):
        return rank

    # Use conditional to handle empty filtered_indices
    rank = jax.lax.cond(
        num_filtered > 0,
        apply_filtering,
        no_filtering,
        rank,
        filtered_indices,
    )

    return jnp.maximum(rank, 1)  # Ensure rank >= 1


@jax.jit
def compute_ranks_unfiltered(scores: jnp.ndarray, true_indices: jnp.ndarray) -> jnp.ndarray:
    """
    Compute unfiltered ranks for a batch of score vectors.

    :param scores: Scores [B, num_entities]
    :param true_indices: True entity indices [B]
    :return: Ranks [B]
    """
    batch_indices = jnp.arange(scores.shape[0])
    true_scores = scores[batch_indices, true_indices]
    return jnp.sum(scores > true_scores[:, None], axis=1) + 1


def find_filtered_indices(
    filter_triples: np.ndarray,
    h: int,
    r: int,
    t: int,
    corruption_side: Literal["head", "tail"],
) -> np.ndarray:
    """
    Find indices of other true entities to filter (non-JIT, runs on host).

    :param filter_triples: All known triples [M, 3]
    :param h: Head entity
    :param r: Relation
    :param t: Tail entity
    :param corruption_side: Which side to corrupt
    :return: Array of entity indices to filter
    """
    if corruption_side == "tail":
        # Find all triples with same (h, r) but different t
        mask = (filter_triples[:, 0] == h) & (filter_triples[:, 1] == r) & (filter_triples[:, 2] != t)
        return filter_triples[mask, 2]
    else:  # head
        # Find all triples with same (r, t) but different h
        mask = (filter_triples[:, 1] == r) & (filter_triples[:, 2] == t) & (filter_triples[:, 0] != h)
        return filter_triples[mask, 0]


def rank_triple(
    model,
    triple: np.ndarray,
    num_entities: int,
    corruption_side: Literal["head", "tail"],
    filter_triples: np.ndarray | None = None,
) -> int:
    """
    Compute rank for a single triple (host-side wrapper).

    :param model: KGE model
    :param triple: Single triple [3]
    :param num_entities: Total number of entities
    :param corruption_side: Which side to corrupt
    :param filter_triples: All known triples for filtering [M, 3]
    :return: Rank (1-indexed, Python int)
    """
    # Find filtered indices on host (avoids boolean indexing in JIT)
    if filter_triples is not None:
        h, r, t = int(triple[0]), int(triple[1]), int(triple[2])
        filtered_indices_np = find_filtered_indices(filter_triples, h, r, t, corruption_side)
        filtered_indices = jnp.asarray(filtered_indices_np, dtype=jnp.int32)
    else:
        filtered_indices = jnp.array([], dtype=jnp.int32)

    triple_jax = jnp.asarray(triple, dtype=jnp.int32)
    scores = score_all_entities(model, triple_jax, num_entities, corruption_side)
    true_idx = triple_jax[2] if corruption_side == "tail" else triple_jax[0]
    rank_array = compute_rank(scores, true_idx, filtered_indices)

    return int(rank_array)
