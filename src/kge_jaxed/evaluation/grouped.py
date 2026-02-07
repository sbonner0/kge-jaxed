"""Grouped evaluation utilities for KGE ranking."""

from typing import Literal

import jax.numpy as jnp
import numpy as np
import pandas as pd

from kge_jaxed.evaluation.utils import compute_group_ranks, score_all_entities_batch
from kge_jaxed.models.base_kge import BaseKGE


def build_group_maps(
    df: pd.DataFrame,
    key_cols: list[str],
    value_col: str,
) -> tuple[dict[tuple[int, int], tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """
    Build grouping maps for evaluation.

    :param df: Input DataFrame containing triples.
    :type df: pd.DataFrame
    :param key_cols: Columns defining a group key (e.g., ["head", "relation"]).
    :type key_cols: list[str]
    :param value_col: Column with the true entity indices for the group.
    :type value_col: str
    :return: (groups, pairs) where:
        - groups maps key -> (row indices in df, true entity indices in the group)
        - pairs is an array of unique keys as given by the key_cols
    :rtype: tuple[dict[tuple[int, int], tuple[np.ndarray, np.ndarray]], np.ndarray]
    """
    grouped = df.groupby(key_cols)
    group_indices = grouped.indices
    group_values = {key: series.to_numpy(dtype=np.int32, copy=False) for key, series in grouped[value_col]}
    groups = {key: (indices, group_values[key]) for key, indices in group_indices.items()}
    pairs = np.array(list(groups.keys()), dtype=np.int32)

    return groups, pairs


def build_filter_map(
    df: pd.DataFrame,
    key_cols: list[str],
    value_col: str,
) -> dict[tuple[int, int], np.ndarray]:
    """
    Build a filtered evaluation map from all known triples.

    :param df: Input DataFrame containing all known triples.
    :type df: pd.DataFrame
    :param key_cols: Columns defining a group key.
    :type key_cols: list[str]
    :param value_col: Column with entity indices to filter.
    :type value_col: str
    :return: Map from group key to filtered entity indices.
    :rtype: dict[tuple[int, int], np.ndarray]
    """
    grouped = df.groupby(key_cols)[value_col]
    return {key: series.to_numpy(dtype=np.int32, copy=False) for key, series in grouped}


def score_grouped_pairs(
    model: BaseKGE,
    pairs: np.ndarray,
    groups: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    filter_map: dict[tuple[int, int], np.ndarray],
    corruption_side: Literal["head", "tail"],
    num_entities: int,
    eval_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Score unique pairs in batches and assign ranks to all triples in each group.

    :param model: KGE model with score_hrt.
    :type model: BaseKGE
    :param pairs: Unique group keys to score, shape [G, 2].
    :type pairs: np.ndarray
    :param groups: Map from key -> (row indices in eval_df, true entity indices for that key).
    :type groups: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]
    :param filter_map: Map from key to filtered entity indices (may be empty).
    :type filter_map: dict[tuple[int, int], np.ndarray]
    :param corruption_side: Which side to corrupt ("head" or "tail").
    :type corruption_side: Literal["head", "tail"]
    :param num_entities: Total number of entities in the graph.
    :type num_entities: int
    :param eval_batch_size: Batch size for scoring unique pairs.
    :type eval_batch_size: int
    :return: Tuple of ranks and true-entity scores aligned to original eval DataFrame rows.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    total_triples = sum(len(indices) for indices, _ in groups.values())
    ranks_out = np.empty(total_triples, dtype=np.int32)
    scores_out = np.empty(total_triples, dtype=np.float32)
    processed = 0

    # Loop over the set of partial triple groups
    for start in range(0, len(pairs), eval_batch_size):
        batch_pairs = pairs[start : start + eval_batch_size]
        dummy = np.zeros(len(batch_pairs), dtype=np.int32)
        if corruption_side == "tail":
            triples = np.column_stack([batch_pairs[:, 0], batch_pairs[:, 1], dummy])
        else:
            triples = np.column_stack([dummy, batch_pairs[:, 0], batch_pairs[:, 1]])

        # Score every entity used to complete the partial triples
        scores = score_all_entities_batch(
            model,
            jnp.asarray(triples, dtype=jnp.int32),
            num_entities,
            corruption_side=corruption_side,
        )

        # Compute the rank of the true entity - looping per group
        for i, pair in enumerate(batch_pairs):
            key = (int(pair[0]), int(pair[1]))
            indices, target_ids = groups[key]
            filtered_indices = filter_map.get(key)

            # Compute ranks for this group and assign to output arrays
            ranks_out[indices] = compute_group_ranks(scores[i], target_ids, filtered_indices)
            scores_out[indices] = np.asarray(scores[i][jnp.asarray(target_ids, dtype=jnp.int32)])

            processed += len(indices)

        if processed % 100 == 0 or processed >= total_triples:
            print(f"  Processed {processed}/{total_triples} triples ({corruption_side})...")

    return ranks_out, scores_out
