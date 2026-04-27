"""Canonical registries for user-selectable KGE-JAXed components."""

from kge_jaxed.registry.builtins import ensure_builtins_registered
from kge_jaxed.registry.core import (
    constrainers,
    initializers,
    losses,
    models,
    negative_samplers,
    optimizers,
    regularizers,
)

for _registry in (
    constrainers,
    initializers,
    losses,
    models,
    negative_samplers,
    optimizers,
    regularizers,
):
    _registry.set_loader(ensure_builtins_registered)

__all__ = [
    "constrainers",
    "ensure_builtins_registered",
    "initializers",
    "losses",
    "models",
    "negative_samplers",
    "optimizers",
    "regularizers",
]
