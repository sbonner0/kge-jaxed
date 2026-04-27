import subprocess
import sys

from kge_jaxed.loss_functions.losses import self_adversarial_negative_sampling_loss
from kge_jaxed.models.initializers import resolve_embedding_init
from kge_jaxed.models.transe import TransE
from kge_jaxed.registry import constrainers, initializers, losses, models, optimizers, regularizers


def test_registry_facade_resolves_builtin_components() -> None:
    assert models.get("transe") is TransE
    assert losses.build("nssa") is self_adversarial_negative_sampling_loss
    assert "adam" in optimizers.names()


def test_registry_aliases_resolve_to_same_component() -> None:
    assert constrainers.get("normalize") is constrainers.get("unit_norm")
    assert constrainers.get("complex_normalize") is constrainers.get("unit_modulus")
    assert regularizers.get("n3") is regularizers.get("np")
    assert regularizers.get("powersum") is regularizers.get("np")


def test_initializer_registry_matches_resolver() -> None:
    assert initializers.build("default") is None
    assert resolve_embedding_init("default", None) is None
    assert callable(resolve_embedding_init("normal_norm", {"stddev": 0.1}))


def test_registry_build_binds_loss_kwargs() -> None:
    bound_loss = losses.build("nssa", adversarial_temperature=0.5, margin=1.0)
    assert callable(bound_loss)


def test_registry_import_order_avoids_circular_imports() -> None:
    code = """
from kge_jaxed.models.transe import TransE
from kge_jaxed.registry import models
assert models.get("transe") is TransE
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
