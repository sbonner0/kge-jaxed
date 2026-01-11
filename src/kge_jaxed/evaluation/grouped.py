"""Grouped evaluation utilities for KGE ranking."""

from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
import pandas as pd

from kge_jaxed.evaluation.utils import compute_group_ranks, score_all_entities_batch


def build_group_maps(
    df: pd.DataFrame,
    key_cols: list[str],
    value_col: str,
) -> tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, int], np.ndarray], np.ndarray]:
    """
    Build grouping maps for evaluation.

    :param df: Input DataFrame containing triples.
    :type df: pd.DataFrame
    :param key_cols: Columns defining a group key (e.g., ["head", "relation"]).
    :type key_cols: list[str]
    :param value_col: Column with the true entity indices for the group.
    :type value_col: str
    :return: (group_indices, group_values, pairs) where:
        - group_indices maps key -> row indices in df
        - group_values maps key -> true entity indices in the group
        - pairs is an array of unique keys
    :rtype: tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, int], np.ndarray], np.ndarray]
    """
    grouped = df.groupby(key_cols)
    group_indices = grouped.indices
    group_values = {key: series.to_numpy(dtype=np.int32, copy=False) for key, series in grouped[value_col]}
    pairs = np.array(list(group_indices.keys()), dtype=np.int32)
    if pairs.size == 0:
        pairs = pairs.reshape(0, len(key_cols))
    return group_indices, group_values, pairs


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
    model,
    pairs: np.ndarray,
    group_indices: dict[tuple[int, int], np.ndarray],
    group_values: dict[tuple[int, int], np.ndarray],
    filter_map: dict[tuple[int, int], np.ndarray],
    corruption_side: Literal["head", "tail"],
    num_entities: int,
    eval_batch_size: int,
    total_triples: int,
    label: str,
    scores_out: np.ndarray | None = None,
) -> np.ndarray:
    """
    Score unique pairs in batches and assign ranks to all triples in each group.

    :param model: KGE model with score_hrt.
    :type model: Any
    :param pairs: Unique group keys to score, shape [G, 2].
    :type pairs: np.ndarray
    :param group_indices: Map from key to row indices in the eval DataFrame.
    :type group_indices: dict[tuple[int, int], np.ndarray]
    :param group_values: Map from key to true entity indices for that key.
    :type group_values: dict[tuple[int, int], np.ndarray]
    :param filter_map: Map from key to filtered entity indices (may be empty).
    :type filter_map: dict[tuple[int, int], np.ndarray]
    :param corruption_side: Which side to corrupt ("head" or "tail").
    :type corruption_side: Literal["head", "tail"]
    :param num_entities: Total number of entities in the graph.
    :type num_entities: int
    :param eval_batch_size: Batch size for scoring unique pairs.
    :type eval_batch_size: int
    :param total_triples: Total number of eval triples (for progress logging).
    :type total_triples: int
    :param label: Progress label string.
    :type label: str
    :param scores_out: Optional array to store true entity scores aligned to eval rows.
    :type scores_out: np.ndarray | None
    :return: Ranks aligned to the original eval DataFrame rows.
    :rtype: np.ndarray
    """
    ranks_out = np.empty(total_triples, dtype=np.int32)
    processed = 0
    for start in range(0, len(pairs), eval_batch_size):
        batch_pairs = pairs[start : start + eval_batch_size]
        dummy = np.zeros(len(batch_pairs), dtype=np.int32)
        if corruption_side == "tail":
            triples = np.column_stack([batch_pairs[:, 0], batch_pairs[:, 1], dummy])
        else:
            triples = np.column_stack([dummy, batch_pairs[:, 0], batch_pairs[:, 1]])
        scores = score_all_entities_batch(
            model,
            jnp.asarray(triples, dtype=jnp.int32),
            num_entities,
            corruption_side=corruption_side,
        )

        for i, pair in enumerate(batch_pairs):
            key = (int(pair[0]), int(pair[1]))
            indices = group_indices[key]
            values = group_values[key]
            filtered_indices = filter_map.get(key)
            ranks = compute_group_ranks(scores[i], values, filtered_indices)
            ranks_out[indices] = np.asarray(ranks)
            if scores_out is not None:
                true_scores = scores[i][jnp.asarray(values, dtype=jnp.int32)]
                scores_out[indices] = np.asarray(true_scores)
            processed += len(indices)

        if processed % 100 == 0 or processed >= total_triples:
            print(f"  Processed {processed}/{total_triples} triples ({label})...")

    return ranks_out
