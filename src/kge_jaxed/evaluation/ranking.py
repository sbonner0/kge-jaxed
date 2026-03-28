"""High-level helpers for ranking-based evaluation."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd  # type: ignore[import]

from kge_jaxed.datasets.base import BaseDataset
from kge_jaxed.evaluation.grouped_ranking import build_filter_map, build_group_maps, score_grouped_pairs
from kge_jaxed.evaluation.validation import validate_eval_df
from kge_jaxed.models.base_kge import BaseKGE

EvalSplit = Literal["train", "valid", "test"]
EvalSourceLabel = EvalSplit | Literal["custom"]
CorruptionSide = Literal["head", "tail"]
QueryKey = tuple[int, int]
EntityFilterMap = dict[QueryKey, np.ndarray]


@dataclass(frozen=True)
class EvalFilterMaps:
    """Named filter maps for head and tail corruption queries."""

    tail: EntityFilterMap
    head: EntityFilterMap


def _grouping_for_side(corruption_side: CorruptionSide) -> tuple[list[str], str]:
    """Return the query grouping columns and target entity column for one side."""
    if corruption_side == "tail":
        return ["head", "relation"], "tail"
    return ["relation", "tail"], "head"


def resolve_eval_dataframe(
    dataset: BaseDataset,
    split: EvalSplit | None,
    eval_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, EvalSourceLabel]:
    """
    Resolve the triples used for ranking evaluation.

    This helper accepts either a named dataset split or a caller-provided evaluation DataFrame, validates the input
    combination, and returns a re-indexed DataFrame together with the split label used in reporting.

    :param dataset: Dataset providing the built-in train, validation, and test splits.
    :type dataset: BaseDataset
    :param split: Dataset split to evaluate when ``eval_df`` is not provided. Must be one of ``"train"``, ``"valid"``,
        or ``"test"``.
    :type split: EvalSplit | None
    :param eval_df: Optional custom evaluation triples with columns ``["head", "relation", "tail"]``.
    :type eval_df: pd.DataFrame | None
    :returns: A copy of the selected evaluation triples with a reset integer index, and the reporting label for that
        source. Custom data uses the label ``"custom"``.
    :rtype: tuple[pd.DataFrame, EvalSourceLabel]
    :raises ValueError: If both ``split`` and ``eval_df`` are provided, if neither  source is provided, or if ``split``
        is not one of the supported names.
    """
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
) -> EvalFilterMaps:
    """
    Build lookup tables of known true entities for filtered ranking metrics.

    When filtered evaluation is enabled, all triples from the train, validation, and test splits are combined so that
    candidate entities that already form a known true triple can be masked during rank computation.
    Separate maps are returned for tail prediction and head prediction queries.

    :param dataset: Dataset whose triples define the known true facts used for filtering.
    :type dataset: BaseDataset
    :param filtered: Whether filtered ranking evaluation is enabled.
    :type filtered: bool
    :returns: Named dictionaries keyed by grouped query pairs. ``tail`` maps
        ``(head, relation)`` to known true tails and ``head`` maps
        ``(relation, tail)`` to known true heads. Empty dictionaries are
        returned when filtered evaluation is disabled.
    :rtype: EvalFilterMaps
    """
    if not filtered:
        return EvalFilterMaps(tail={}, head={})

    filter_triples = pd.concat(
        [dataset.train_df, dataset.val_df, dataset.test_df],
        ignore_index=True,
    )
    tail_filter_map = build_filter_map(filter_triples, ["head", "relation"], "tail")
    head_filter_map = build_filter_map(filter_triples, ["relation", "tail"], "head")
    return EvalFilterMaps(tail=tail_filter_map, head=head_filter_map)


def evaluate_corruption_side(
    model: BaseKGE,
    eval_df: pd.DataFrame,
    *,
    filter_map: EntityFilterMap,
    corruption_side: CorruptionSide,
    num_entities: int,
    eval_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Score one corruption direction and recover row-aligned ranks.

    The evaluation triples are first grouped by the fixed query columns for the requested corruption side so repeated
    queries can be scored once against all entities. The grouped queries are then scored in batches, filtered if
    requested, and expanded back into arrays aligned with the original ``eval_df`` rows.

    :param model: Knowledge graph embedding model used to score candidate entities.
    :type model: BaseKGE
    :param eval_df: Evaluation triples with columns ``["head", "relation", "tail"]``.
    :type eval_df: pd.DataFrame
    :param filter_map: Optional mapping from grouped query keys to known true entity ids that should be filtered out
        before ranking.
    :type filter_map: EntityFilterMap
    :param corruption_side: Which entity position to corrupt during ranking, either ``"head"`` or ``"tail"``.
    :type corruption_side: CorruptionSide
    :param num_entities: Total number of candidate entities scored per query.
    :type num_entities: int
    :param eval_batch_size: Number of grouped queries to score per batch.
    :type eval_batch_size: int
    :returns: Two arrays aligned with ``eval_df`` rows: integer ranks for the true entities and their raw model scores.
    :rtype: tuple[np.ndarray, np.ndarray]
    :raises ValueError: If ``eval_batch_size`` is not a positive integer.
    """
    if eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be a positive integer")

    group_cols, value_col = _grouping_for_side(corruption_side)
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
