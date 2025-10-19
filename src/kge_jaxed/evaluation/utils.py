"""Ranking-based evaluation metrics for KGE models."""

from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from flax import nnx


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


@jax.jit
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


def find_filtered_indices(
    filter_triples: jnp.ndarray,
    h: int,
    r: int,
    t: int,
    corruption_side: Literal["head", "tail"],
) -> jnp.ndarray:
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


@partial(nnx.jit, static_argnames=("num_entities", "corruption_side"))
def rank_triple_jit(
    model,
    triple: jnp.ndarray,
    num_entities: int,
    corruption_side: Literal["head", "tail"],
    filtered_indices: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute rank for a single triple (JIT-compiled core).

    :param model: KGE model
    :param triple: Single triple [3]
    :param num_entities: Total number of entities
    :param corruption_side: Which side to corrupt
    :param filtered_indices: Pre-computed filtered entity indices
    :return: Rank (1-indexed, scalar array)
    """
    # Score all entities
    scores = score_all_entities(model, triple, num_entities, corruption_side)

    # Get true entity index (keep as JAX array, don't convert to int)
    true_idx = triple[2] if corruption_side == "tail" else triple[0]

    # Compute rank with filtering
    return compute_rank(scores, true_idx, filtered_indices)


def rank_triple(
    model,
    triple: jnp.ndarray,
    num_entities: int,
    corruption_side: Literal["head", "tail"],
    filter_triples: jnp.ndarray | None = None,
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
    filtered_indices = None
    if filter_triples is not None:
        h, r, t = int(triple[0]), int(triple[1]), int(triple[2])
        filtered_indices = find_filtered_indices(filter_triples, h, r, t, corruption_side)

        # Convert to JAX array (can be empty)
        if len(filtered_indices) > 0:
            filtered_indices = jnp.array(filtered_indices, dtype=jnp.int32)
        else:
            filtered_indices = jnp.array([], dtype=jnp.int32)
    else:
        filtered_indices = jnp.array([], dtype=jnp.int32)

    # Call JIT-compiled function
    rank_array = rank_triple_jit(model, triple, num_entities, corruption_side, filtered_indices)

    # Convert to Python int only at the very end
    return int(rank_array)
