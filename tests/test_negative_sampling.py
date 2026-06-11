import jax
import jax.numpy as jnp
import numpy as np

from kge_jaxed.negative_sampling.uniform_negative_sampling import (
    uniform_balanced_sampler,
    uniform_corruptions,
)

TRIPLES = jnp.array([[0, 0, 1], [2, 1, 3], [4, 0, 5], [5, 2, 0]], dtype=jnp.int32)
NUM_ENTITIES = 6


def test_output_shape_and_dtype() -> None:
    neg = uniform_balanced_sampler(TRIPLES, NUM_ENTITIES, k=7, key=jax.random.PRNGKey(0))
    assert neg.shape == (TRIPLES.shape[0], 7, 3)
    assert neg.dtype == TRIPLES.dtype


def test_exactly_one_side_corrupted_and_relation_preserved() -> None:
    k = 50
    neg = np.asarray(uniform_balanced_sampler(TRIPLES, NUM_ENTITIES, k=k, key=jax.random.PRNGKey(1)))
    pos = np.asarray(TRIPLES)[:, None, :]  # [B, 1, 3]

    assert np.all(neg[:, :, 1] == pos[:, :, 1])

    head_changed = neg[:, :, 0] != pos[:, :, 0]
    tail_changed = neg[:, :, 2] != pos[:, :, 2]
    assert np.all(head_changed ^ tail_changed)


def test_corruption_is_balanced_and_uniform_over_other_entities() -> None:
    # Single fixed triple so each side has a known original entity.
    triple = jnp.array([[2, 0, 4]], dtype=jnp.int32)
    k = 2000
    neg = np.asarray(uniform_balanced_sampler(triple, NUM_ENTITIES, k=k, key=jax.random.PRNGKey(2)))[0]

    head_corrupted = neg[:, 0] != 2
    replacements = np.where(head_corrupted, neg[:, 0], neg[:, 2])

    # The corrupted side never resamples its original entity.
    assert np.all(neg[head_corrupted, 0] != 2)
    assert np.all(neg[~head_corrupted, 2] != 4)

    # Roughly half of the negatives corrupt the head.
    assert 0.45 <= head_corrupted.mean() <= 0.55

    # Each side draws uniformly from the 5 entities excluding its original, so
    # the side-specific originals (2 and 4) are reachable from one side only
    # (~k/10 draws each) and every other entity from both sides (~k/5 draws).
    counts = np.bincount(replacements, minlength=NUM_ENTITIES)
    expected = np.array([k / 5, k / 5, k / 10, k / 5, k / 10, k / 5])
    np.testing.assert_allclose(counts, expected, rtol=0.25)


def test_uniform_corruptions_never_resamples_original() -> None:
    side, neg_ent = uniform_corruptions(TRIPLES, NUM_ENTITIES, k=50, key=jax.random.PRNGKey(6))
    side, neg_ent = np.asarray(side), np.asarray(neg_ent)
    pos = np.asarray(TRIPLES)

    assert side.shape == neg_ent.shape == (TRIPLES.shape[0], 50)
    originals = np.where(side, pos[:, 0:1], pos[:, 2:3])
    assert np.all(neg_ent != originals)


def test_same_key_is_deterministic() -> None:
    first = uniform_balanced_sampler(TRIPLES, NUM_ENTITIES, k=8, key=jax.random.PRNGKey(3))
    second = uniform_balanced_sampler(TRIPLES, NUM_ENTITIES, k=8, key=jax.random.PRNGKey(3))
    other_key = uniform_balanced_sampler(TRIPLES, NUM_ENTITIES, k=8, key=jax.random.PRNGKey(4))

    assert jnp.array_equal(first, second)
    assert not jnp.array_equal(first, other_key)
