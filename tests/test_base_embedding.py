import jax.numpy as jnp
from flax import nnx
from flax.nnx import initializers as nnx_initializers

from kge_jaxed.models.base_embedding import BaseEmbedding


def _get_embedding_weights(emb: BaseEmbedding):
    return nnx.state(emb)["emb"]["embedding"][...]


def test_embedding_shape():
    emb = BaseEmbedding(num_embeddings=10, embedding_dim=4, seed=0)
    weights = _get_embedding_weights(emb)
    assert weights.shape == (10, 4)


def test_embedding_init_uniform_range():
    emb = BaseEmbedding(
        num_embeddings=10,
        embedding_dim=4,
        seed=0,
        embedding_init="uniform",
        embedding_init_kwargs={"scale": 0.05},
    )
    weights = _get_embedding_weights(emb)
    assert jnp.max(jnp.abs(weights)) <= 0.05 + 1e-6


def test_embedding_init_normal_stddev():
    emb = BaseEmbedding(
        num_embeddings=10,
        embedding_dim=4,
        seed=0,
        embedding_init="normal",
        embedding_init_kwargs={"stddev": 0.02},
    )
    weights = _get_embedding_weights(emb)
    std = float(weights.std())
    assert 0.005 < std < 0.05


def test_embedding_init_callable_respected():
    emb_uniform = BaseEmbedding(
        num_embeddings=10,
        embedding_dim=4,
        seed=0,
        embedding_init="uniform",
    )
    emb_callable = BaseEmbedding(
        num_embeddings=10,
        embedding_dim=4,
        seed=0,
        embedding_init=nnx_initializers.normal(stddev=0.02),
    )
    weights_uniform = _get_embedding_weights(emb_uniform)
    weights_callable = _get_embedding_weights(emb_callable)
    assert not jnp.allclose(weights_uniform, weights_callable)


def test_embedding_init_xavier_uniform_norm_unit_rows():
    emb = BaseEmbedding(num_embeddings=10, embedding_dim=4, seed=0, embedding_init="xavier_uniform_norm")
    weights = _get_embedding_weights(emb)
    norms = jnp.linalg.norm(weights, axis=1)
    assert jnp.allclose(norms, jnp.ones_like(norms), atol=1e-6)


def test_embedding_init_phases_unit_modulus():
    emb = BaseEmbedding(
        num_embeddings=10,
        embedding_dim=4,
        seed=0,
        embedding_init="init_phases",
        param_dtype=jnp.complex64,
    )
    weights = _get_embedding_weights(emb)
    assert jnp.iscomplexobj(weights)
    assert jnp.allclose(jnp.abs(weights), jnp.ones_like(jnp.abs(weights)), atol=1e-6)


def test_embedding_init_phases_requires_complex_dtype():
    try:
        BaseEmbedding(
            num_embeddings=10,
            embedding_dim=4,
            seed=0,
            embedding_init="init_phases",
            param_dtype=jnp.float32,
        )
    except TypeError as exc:
        assert "requires a complex dtype" in str(exc)
    else:
        raise AssertionError("Expected TypeError for init_phases with non-complex dtype")


def test_dropout_deterministic_path():
    emb = BaseEmbedding(num_embeddings=10, embedding_dim=4, seed=0, dropout_rate=0.5)
    x = jnp.array([0, 1, 2], dtype=jnp.int32)
    out = emb(x, rngs=None)
    expected = emb.emb(x)
    assert jnp.allclose(out, expected)


def test_dropout_stochastic_path_changes_output():
    emb = BaseEmbedding(num_embeddings=10, embedding_dim=4, seed=0, dropout_rate=0.5)
    x = jnp.array([0, 1, 2], dtype=jnp.int32)
    out_a = emb(x, rngs=nnx.Rngs(0))
    out_b = emb(x, rngs=nnx.Rngs(1))
    expected = emb.emb(x)
    assert not jnp.allclose(out_a, expected) or not jnp.allclose(out_b, expected)
    assert not jnp.allclose(out_a, out_b)


def test_unknown_embedding_init_raises():
    try:
        BaseEmbedding(num_embeddings=10, embedding_dim=4, seed=0, embedding_init="not-a-real-init")
    except ValueError as exc:
        assert "Unknown embedding_init" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown embedding_init")


def test_invalid_embedding_init_type_raises():
    try:
        BaseEmbedding(num_embeddings=10, embedding_dim=4, seed=0, embedding_init=123)
    except TypeError as exc:
        assert "embedding_init must be a string" in str(exc)
    else:
        raise AssertionError("Expected TypeError for invalid embedding_init type")


def test_missing_rngs_and_seed_raises():
    try:
        BaseEmbedding(num_embeddings=10, embedding_dim=4, rngs=None, seed=None)
    except ValueError as exc:
        assert "requires rngs or seed" in str(exc)
    else:
        raise AssertionError("Expected ValueError when rngs and seed are missing")


def test_xavier_variance_scaling_kwargs_change_distribution():
    emb_default = BaseEmbedding(num_embeddings=10, embedding_dim=4, seed=0, embedding_init="xavier")
    emb_scaled = BaseEmbedding(
        num_embeddings=10,
        embedding_dim=4,
        seed=0,
        embedding_init="xavier",
        embedding_init_kwargs={"scale": 2.0, "mode": "fan_in", "distribution": "uniform"},
    )
    weights_default = _get_embedding_weights(emb_default)
    weights_scaled = _get_embedding_weights(emb_scaled)
    assert not jnp.allclose(weights_default, weights_scaled)
