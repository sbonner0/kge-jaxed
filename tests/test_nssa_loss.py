import jax
import jax.numpy as jnp

from kge_jaxed.loss_functions.losses import self_adversarial_negative_sampling_loss
from kge_jaxed.registries import get_loss, list_losses


def test_nssa_matches_rotate_style_reference_without_margin() -> None:
    pos_scores = jnp.array([0.2, -0.3], dtype=jnp.float32)
    neg_scores = jnp.array([[0.1, -0.5, 0.7], [-0.2, 0.0, 0.3]], dtype=jnp.float32)
    temperature = 1.5

    weights = jax.nn.softmax(temperature * jax.lax.stop_gradient(neg_scores), axis=-1)
    pos_term = -jnp.mean(jax.nn.log_sigmoid(pos_scores))
    neg_term = -jnp.mean(jnp.sum(weights * jax.nn.log_sigmoid(-neg_scores), axis=-1))
    expected = 0.5 * (pos_term + neg_term)

    actual = self_adversarial_negative_sampling_loss(
        pos_scores,
        neg_scores,
        adversarial_temperature=temperature,
        margin=0.0,
    )
    assert jnp.allclose(actual, expected, atol=1e-7)


def test_nssa_matches_margin_formulation() -> None:
    pos_scores = jnp.array([0.4, -0.1], dtype=jnp.float32)
    neg_scores = jnp.array([[0.2, -0.4], [0.0, 0.3]], dtype=jnp.float32)
    temperature = 0.75
    margin = 9.0

    weights = jax.nn.softmax(temperature * jax.lax.stop_gradient(neg_scores), axis=-1)
    pos_term = -jnp.mean(jax.nn.log_sigmoid(margin + pos_scores))
    neg_term = -jnp.mean(jnp.sum(weights * jax.nn.log_sigmoid(-(margin + neg_scores)), axis=-1))
    expected = 0.5 * (pos_term + neg_term)

    actual = self_adversarial_negative_sampling_loss(
        pos_scores,
        neg_scores,
        adversarial_temperature=temperature,
        margin=margin,
    )
    assert jnp.allclose(actual, expected, atol=1e-7)


def test_nssa_gradient_matches_constant_weight_reference() -> None:
    pos_scores = jnp.array([0.1, -0.2], dtype=jnp.float32)
    neg_scores = jnp.array([[0.3, -0.1, 0.5], [-0.2, 0.4, 0.0]], dtype=jnp.float32)
    temperature = 2.0

    fixed_weights = jax.nn.softmax(temperature * jax.lax.stop_gradient(neg_scores), axis=-1)

    def nssa_loss(neg: jnp.ndarray) -> jnp.ndarray:
        return self_adversarial_negative_sampling_loss(
            pos_scores,
            neg,
            adversarial_temperature=temperature,
        )

    def reference_loss(neg: jnp.ndarray) -> jnp.ndarray:
        pos_term = -jax.nn.log_sigmoid(pos_scores)
        neg_term_unreduced = -jax.nn.log_sigmoid(-neg)
        neg_term = jnp.sum(fixed_weights * neg_term_unreduced, axis=-1)
        return 0.5 * (jnp.mean(pos_term) + jnp.mean(neg_term))

    grad_nssa = jax.grad(nssa_loss)(neg_scores)
    grad_reference = jax.grad(reference_loss)(neg_scores)

    assert jnp.all(jnp.isfinite(grad_nssa))
    assert jnp.all(jnp.isfinite(grad_reference))
    assert jnp.allclose(grad_nssa, grad_reference, atol=1e-7)


def test_registry_binds_nssa_kwargs() -> None:
    assert "nssa" in list_losses()

    pos_scores = jnp.array([0.2], dtype=jnp.float32)
    neg_scores = jnp.array([[0.1, 0.3]], dtype=jnp.float32)
    kwargs = {"adversarial_temperature": 0.5, "margin": 1.0}

    loss_fn = get_loss("nssa", **kwargs)
    expected = self_adversarial_negative_sampling_loss(pos_scores, neg_scores, **kwargs)
    actual = loss_fn(pos_scores, neg_scores)

    assert jnp.allclose(actual, expected, atol=1e-7)
