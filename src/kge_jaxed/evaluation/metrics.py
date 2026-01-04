"""Ranking Metrics."""

from collections.abc import Sequence

import jax.numpy as jnp
import pandas as pd


def compute_metrics_from_ranks(ranks: jnp.ndarray, ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict[str, float]:
    """
    Compute standard KGE metrics from a list of ranks.

    :param ranks: Array of ranks [N]
    :param ks: Hit thresholds to compute (e.g., (1, 3, 5, 10))
    :return: Dictionary with MRR, MR, and Hits@K for each k in ks
    """
    ranks = jnp.array(ranks)

    metrics = {
        "mrr": float(jnp.mean(1.0 / ranks)),
        "mr": float(jnp.mean(ranks)),
    }
    for k in ks:
        metrics[f"hits@{k}"] = float(jnp.mean(ranks <= k))

    return metrics


def compute_metrics_dataframe(
    head_ranks: jnp.ndarray | Sequence[int],
    tail_ranks: jnp.ndarray | Sequence[int],
) -> pd.DataFrame:
    """
    Compute metrics per side (head/tail) and an averaged column.

    :param head_ranks: Ranks for head corruption
    :param tail_ranks: Ranks for tail corruption
    :return: DataFrame indexed by metric with columns [head, tail, avg]
    """
    head_metrics = compute_metrics_from_ranks(jnp.asarray(head_ranks))
    tail_metrics = compute_metrics_from_ranks(jnp.asarray(tail_ranks))

    df = pd.DataFrame({"head": head_metrics, "tail": tail_metrics})
    df["avg"] = df[["head", "tail"]].mean(axis=1)

    return df
