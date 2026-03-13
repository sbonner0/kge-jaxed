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
    """Group evaluation triples by query and keep the matching row positions."""
    grouped = df.groupby(key_cols, sort=False)
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
    """Map each query to the set of known true entities used for filtering."""
    grouped = df.groupby(key_cols, sort=False)[value_col]
    return {key: np.unique(series.to_numpy(dtype=np.int32, copy=False)) for key, series in grouped}


def score_grouped_pairs(
    model: BaseKGE,
    pairs: np.ndarray,
    groups: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    filter_map: dict[tuple[int, int], np.ndarray],
    corruption_side: Literal["head", "tail"],
    num_entities: int,
    eval_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Score one corruption side query-by-query and align results to `eval_df` rows."""
    total_triples = sum(len(indices) for indices, _ in groups.values())
    ranks_out = np.empty(total_triples, dtype=np.int32)
    scores_out = np.empty(total_triples, dtype=np.float32)

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
        scores = np.asarray(scores)

        for i, pair in enumerate(batch_pairs):
            key = (int(pair[0]), int(pair[1]))
            indices, target_ids = groups[key]
            filtered_indices = filter_map.get(key)

            ranks_out[indices] = compute_group_ranks(scores[i], target_ids, filtered_indices)
            scores_out[indices] = scores[i][target_ids]

    return ranks_out, scores_out
