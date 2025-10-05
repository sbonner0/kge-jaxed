from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("k",))
def uniform_balanced_sampler(
    triples: jnp.ndarray,
    num_entities: int,
    k: int,
    key: jax.Array,
) -> jax.Array:
    """
    Balanced uniform negative sampling for knowledge graph triples.

    :param triples: Input triples of shape [B, 3] where B is the batch size.
    :type triples: jnp.ndarray
    :param num_entities: Total number of entities in the knowledge graph.
    :type num_entities: int
    :param k: Number of negative samples per positive triple.
    :type k: int
    :param key: JAX random key for reproducibility.
    :type key: jax.Array
    :return: The generated negative samples of shape [B, K, 3]
    :rtype: jax.Array
    """
    B = triples.shape[0]
    h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]

    # side mask: first floor(k/2) are heads, rest tails
    k_head = k // 2
    side = jnp.arange(k) < k_head  # [K] True=head, False=tail

    # sample K entity replacements per row
    raw = jax.random.randint(key, (B, k), 0, num_entities - 1)  # [B, K]
    originals = jnp.where(side[None, :], h[:, None], t[:, None])  # [B, K]
    neg_ent = jnp.where(raw >= originals, raw + 1, raw).astype(triples.dtype)

    # build corrupted triples
    neg_h = jnp.where(side[None, :], neg_ent, h[:, None])  # [B, K]
    neg_t = jnp.where(side[None, :], t[:, None], neg_ent)  # [B, K]
    neg_r = jnp.broadcast_to(r[:, None], (B, k))  # [B, K]

    # stack into triples
    neg = jnp.stack((neg_h, neg_r, neg_t), axis=-1)  # [B, K, 3]

    return neg


if __name__ == "__main__":

    # Example usage
    triples = jnp.array([[0, 0, 1], [2, 1, 3], [4, 0, 5]], dtype=jnp.int32)
    num_entities = 6
    k = 4
    key = jax.random.PRNGKey(42)

    neg_samples = uniform_balanced_sampler(triples, num_entities, k, key)
    print("Positive Triples:\n", triples)
    print("Negative Samples:\n", neg_samples)
    print("Negative Samples Shape:", neg_samples.shape)
