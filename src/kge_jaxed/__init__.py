"""KGE-Jaxed: Knowledge Graph Embedding library in JAX."""

from kge_jaxed.pipeline import KGEPipeline
from kge_jaxed.rngs import RngManager, make_model_rngs

# Export main user-facing API
__all__ = [
    "KGEPipeline",
    "RngManager",
    "make_model_rngs",
]
