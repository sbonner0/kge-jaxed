import jax.numpy as jnp
import pytest

from kge_jaxed.constraints.constrainers import clip, max_norm, unit_modulus
from kge_jaxed.constraints.registry import get_constrainer
from kge_jaxed.regularization.registry import get_regularizer


def test_pykeen_constrainer_aliases_resolve() -> None:
    assert get_constrainer("normalize") is get_constrainer("unit_norm")
    assert get_constrainer("clamp") is get_constrainer("clip")
    assert get_constrainer("clamp_norm") is get_constrainer("max_norm")
    assert get_constrainer("complex_normalize") is get_constrainer("unit_modulus")


def test_unit_modulus_projects_zero_to_unit_value() -> None:
    x = jnp.array([0.0 + 0.0j, 3.0 + 4.0j], dtype=jnp.complex64)
    constrained = unit_modulus()(x)

    assert jnp.allclose(jnp.abs(constrained), jnp.ones_like(jnp.abs(constrained)), atol=1e-6)
    assert constrained[0] == jnp.array(1.0 + 0.0j, dtype=jnp.complex64)


def test_constrainer_factories_validate_arguments() -> None:
    with pytest.raises(ValueError, match="max_value"):
        max_norm(max_value=0.0)
    with pytest.raises(ValueError, match="min_value"):
        clip(min_value=2.0, max_value=1.0)


def test_powersum_regularizer_alias_and_normalization() -> None:
    regularizer = get_regularizer("powersum")(p=2.0, normalize=True)
    weights = jnp.array([[3.0, 4.0], [1.0, 1.0]], dtype=jnp.float32)

    assert float(regularizer(weights)) == pytest.approx(6.75)


def test_n3_regularizer_alias_uses_np_regularizer_defaults() -> None:
    regularizer = get_regularizer("n3")()
    weights = jnp.array([[1.0, 2.0]], dtype=jnp.float32)

    assert float(regularizer(weights)) == pytest.approx(9.0)
