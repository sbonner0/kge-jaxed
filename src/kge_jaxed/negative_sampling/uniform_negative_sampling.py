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
    This sampler creates K negatives per positive by randomly choosing, for each
    negative sample, whether to corrupt the head or the tail. It then replaces
    that entity with a uniformly sampled different entity (via an index shift
    to avoid resampling the original).

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

    key_side, key_ent = jax.random.split(key, 2)
    # sample head/tail corruption independently per negative sample
    side = jax.random.bernoulli(key_side, 0.5, (B, k))  # [B, K] True=head, False=tail

    # sample K entity replacements per row
    raw = jax.random.randint(key_ent, (B, k), 0, num_entities - 1)  # [B, K]
    originals = jnp.where(side, h[:, None], t[:, None])  # [B, K]
    neg_ent = jnp.where(raw >= originals, raw + 1, raw).astype(triples.dtype)

    # build corrupted triples
    neg_h = jnp.where(side, neg_ent, h[:, None])  # [B, K]
    neg_t = jnp.where(side, t[:, None], neg_ent)  # [B, K]
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
