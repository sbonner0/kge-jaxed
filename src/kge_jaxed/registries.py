"""Central registry for all models, losses, samplers, and optimizers."""

import optax

from kge_jaxed.loss_functions.losses import bce_loss, margin_ranking_loss
from kge_jaxed.models.distmult import DistMult
from kge_jaxed.models.transe import TransE

# Import samplers
from kge_jaxed.negative_sampling.uniform_negative_sampling import uniform_balanced_sampler

# ============================================
# Model Registry
# ============================================
MODELS = {
    "transe": TransE,
    "distmult": DistMult,
}


# ============================================
# Loss Registry
# ============================================
LOSSES = {
    "mrl": margin_ranking_loss,
    "bce": bce_loss,
}


# ============================================
# Sampler Registry
# ============================================
NEGATIVE_SAMPLERS = {
    "uniform": uniform_balanced_sampler,
}

# ============================================
# Optimizer Registry
# ============================================
OPTIMIZERS = {
    "adam": optax.adam,
    "adamw": optax.adamw,
    "sgd": optax.sgd,
    "adagrad": optax.adagrad,
    "rmsprop": optax.rmsprop,
    "adadelta": optax.adadelta,
    "adamax": optax.adamax,
}


# ============================================
# Helper Functions
# ============================================
def get_model(name: str):
    """Get model class by name."""
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODELS.keys())}")
    return MODELS[name]


def get_loss(name: str):
    """Get loss function by name."""
    if name not in LOSSES:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(LOSSES.keys())}")
    return LOSSES[name]


def get_sampler(name: str):
    """Get negative sampler by name."""
    if name not in NEGATIVE_SAMPLERS:
        raise ValueError(f"Unknown sampler '{name}'. Available: {list(NEGATIVE_SAMPLERS.keys())}")
    return NEGATIVE_SAMPLERS[name]


def get_optimizer(name: str):
    """Get optimizer factory by name."""
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer '{name}'. Available: {list(OPTIMIZERS.keys())}")
    return OPTIMIZERS[name]


def list_models():
    """Return list of available model names."""
    return list(MODELS.keys())


def list_losses():
    """Return list of available loss names."""
    return list(LOSSES.keys())


def list_samplers():
    """Return list of available sampler names."""
    return list(NEGATIVE_SAMPLERS.keys())


def list_optimizers():
    """Return list of available optimizer names."""
    return list(OPTIMIZERS.keys())
