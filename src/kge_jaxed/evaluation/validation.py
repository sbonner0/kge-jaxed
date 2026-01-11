"""Validation utilities for evaluation inputs."""

import numpy as np
import pandas as pd


def validate_eval_df(eval_df: pd.DataFrame, num_entities: int, num_relations: int) -> None:
    """
    Validate that eval_df uses the expected schema and id ranges.

    :param eval_df: Evaluation DataFrame with columns [head, relation, tail].
    :type eval_df: pd.DataFrame
    :param num_entities: Total number of entities in the dataset.
    :type num_entities: int
    :param num_relations: Total number of relations in the dataset.
    :type num_relations: int
    :raises ValueError: If required columns are missing or ids are out of range.
    """
    missing_cols = {"head", "relation", "tail"} - set(eval_df.columns)
    if missing_cols:
        raise ValueError(f"eval_df missing required columns: {sorted(missing_cols)}")

    eval_array = eval_df[["head", "relation", "tail"]].to_numpy(dtype=np.int64, copy=False)
    if eval_array.size == 0:
        return
    if eval_array.min() < 0:
        raise ValueError("eval_df contains negative entity or relation ids")
    if eval_array[:, 0].max() >= num_entities or eval_array[:, 2].max() >= num_entities:
        raise ValueError("eval_df contains entity ids outside dataset range")
    if eval_array[:, 1].max() >= num_relations:
        raise ValueError("eval_df contains relation ids outside dataset range")
