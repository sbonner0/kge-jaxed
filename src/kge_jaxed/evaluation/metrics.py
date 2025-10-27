"""Ranking Metrics."""

import jax.numpy as jnp


def compute_metrics_from_ranks(ranks: jnp.ndarray) -> dict[str, float]:
    """
    Compute standard KGE metrics from a list of ranks.

    :param ranks: Array of ranks [N]
    :return: Dictionary with MRR, MR, Hits@1, Hits@3, Hits@10
    """
    ranks = jnp.array(ranks)

    return {
        "mrr": float(jnp.mean(1.0 / ranks)),
        "mr": float(jnp.mean(ranks)),
        "hits@1": float(jnp.mean(ranks <= 1)),
        "hits@3": float(jnp.mean(ranks <= 3)),
        "hits@10": float(jnp.mean(ranks <= 10)),
    }
