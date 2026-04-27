"""Register built-in components lazily."""

from kge_jaxed.registry.builtins.constrainers import register_constrainers
from kge_jaxed.registry.builtins.initializers import register_initializers
from kge_jaxed.registry.builtins.losses import register_losses
from kge_jaxed.registry.builtins.models import register_models
from kge_jaxed.registry.builtins.optimizers import register_optimizers
from kge_jaxed.registry.builtins.regularizers import register_regularizers
from kge_jaxed.registry.builtins.samplers import register_samplers

_REGISTERED = False


def ensure_builtins_registered() -> None:
    """Populate the shared registries with built-in components."""
    global _REGISTERED
    if _REGISTERED:
        return

    register_initializers()
    register_constrainers()
    register_regularizers()
    register_losses()
    register_samplers()
    register_optimizers()
    register_models()

    _REGISTERED = True
