"""High-level helpers for ranking-based evaluation."""

from typing import Literal

import numpy as np
import pandas as pd

from kge_jaxed.datasets.base import BaseDataset
from kge_jaxed.evaluation.grouped_ranking import build_filter_map, build_group_maps, score_grouped_pairs
from kge_jaxed.evaluation.validation import validate_eval_df
from kge_jaxed.models.base_kge import BaseKGE

CorruptionSide = Literal["head", "tail"]


def resolve_eval_dataframe(
    dataset: BaseDataset,
    split: str | None,
    eval_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, str]:
    """Return the evaluation triples and a label for reporting."""
    if eval_df is not None:
        if split is not None:
            raise ValueError("Provide only split or eval_df, not both")
        validate_eval_df(eval_df, dataset.num_entities, dataset.num_relations)
        return eval_df.reset_index(drop=True), "custom"

    if split is None:
        raise ValueError("split must be provided when eval_df is not set")

    split_map = {
        "train": dataset.train_df,
        "valid": dataset.val_df,
        "test": dataset.test_df,
    }
    if split not in split_map:
        raise ValueError(f"Invalid split: {split}. Expected one of {list(split_map.keys())}.")

    return split_map[split].reset_index(drop=True), split


def build_eval_filter_maps(
    dataset: BaseDataset,
    filtered: bool,
) -> tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, int], np.ndarray]]:
    """Build filter maps for tail and head prediction."""
    if not filtered:
        return {}, {}

    filter_triples = pd.concat(
        [dataset.train_df, dataset.val_df, dataset.test_df],
        ignore_index=True,
    )
    tail_filter_map = build_filter_map(filter_triples, ["head", "relation"], "tail")
    head_filter_map = build_filter_map(filter_triples, ["relation", "tail"], "head")
    return tail_filter_map, head_filter_map


def evaluate_corruption_side(
    model: BaseKGE,
    eval_df: pd.DataFrame,
    *,
    group_cols: list[str],
    value_col: str,
    filter_map: dict[tuple[int, int], np.ndarray],
    corruption_side: CorruptionSide,
    num_entities: int,
    eval_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ranks and true-triple scores for one corruption side."""
    groups, pairs = build_group_maps(eval_df, group_cols, value_col)
    return score_grouped_pairs(
        model,
        pairs,
        groups,
        filter_map,
        corruption_side,
        num_entities,
        eval_batch_size,
    )
