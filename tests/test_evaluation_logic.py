import numpy as np
import pandas as pd

from kge_jaxed.evaluation.grouped_ranking import build_filter_map
from kge_jaxed.evaluation.utils import compute_group_ranks


def test_compute_group_ranks_filters_other_true_entities() -> None:
    all_scores = np.array([0.4, 0.9, 0.8, 0.1], dtype=np.float32)

    unfiltered_ranks = compute_group_ranks(all_scores, target_ids=np.array([1, 2], dtype=np.int32))
    filtered_ranks = compute_group_ranks(
        all_scores,
        target_ids=np.array([1, 2], dtype=np.int32),
        filtered_ids=np.array([1, 2], dtype=np.int32),
    )

    np.testing.assert_array_equal(unfiltered_ranks, np.array([1, 2], dtype=np.int32))
    np.testing.assert_array_equal(filtered_ranks, np.array([1, 1], dtype=np.int32))


def test_compute_group_ranks_deduplicates_filtered_entities() -> None:
    all_scores = np.array([0.95, 0.9, 0.5, 0.8], dtype=np.float32)

    ranks = compute_group_ranks(
        all_scores,
        target_ids=np.array([2], dtype=np.int32),
        filtered_ids=np.array([1, 1], dtype=np.int32),
    )

    np.testing.assert_array_equal(ranks, np.array([3], dtype=np.int32))


def test_build_filter_map_returns_unique_entities_per_query() -> None:
    triples = pd.DataFrame(
        [
            (0, 0, 1),
            (0, 0, 1),
            (0, 0, 2),
            (1, 0, 3),
        ],
        columns=["head", "relation", "tail"],
        dtype="int32",
    )

    filter_map = build_filter_map(triples, ["head", "relation"], "tail")

    np.testing.assert_array_equal(filter_map[(0, 0)], np.array([1, 2], dtype=np.int32))
    np.testing.assert_array_equal(filter_map[(1, 0)], np.array([3], dtype=np.int32))
