from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
from flax import nnx
from flax.typing import Dtype
from jax import Array

from kge_jaxed.models.initializers import resolve_embedding_init
from kge_jaxed.rngs import make_model_rngs


class BaseEmbedding(nnx.Module):
    """
    Base class for embeddings. Allows for easy extension of embeddings with dropout and custom
    initializers.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dropout_rate: float = 0.0,
        embedding_init: str | Callable | None = None,
        embedding_init_kwargs: dict | None = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the embedding.

        :param num_embeddings: How many embeddings to learn.
        :type num_embeddings: int
        :param embedding_dim: Dimensionality of the embeddings.
        :type embedding_dim: int
        :param dropout_rate: Dropout rate, defaults to 0.0
        :type dropout_rate: float, optional
        :param embedding_init: Initializer name or callable for embedding weights
        :type embedding_init: str | Callable | None, optional
        :param embedding_init_kwargs: Optional kwargs for string-based initializers
        :type embedding_init_kwargs: dict | None, optional
        :param param_dtype: Data type for the parameters, defaults to jnp.float32
        :type param_dtype: Dtype, optional
        :param rngs: RNGs for the module, required unless a seed is provided
        :type rngs: nnx.Rngs, optional
        :param seed: Seed to initialize RNG streams if rngs is not provided
        :type seed: int, optional
        """

        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        if rngs is None:
            if seed is None:
                raise ValueError("BaseEmbedding requires rngs or seed to be provided.")
            rngs = make_model_rngs(seed)

        embedding_init_fn = resolve_embedding_init(embedding_init, embedding_init_kwargs)

        embed_kwargs: dict[str, Any] = {}
        if embedding_init_fn is not None:
            embed_kwargs["embedding_init"] = embedding_init_fn

        self.emb = nnx.Embed(
            num_embeddings=self.num_embeddings,
            features=self.embedding_dim,
            param_dtype=param_dtype,
            **embed_kwargs,
            rngs=rngs,
            **kwargs,
        )
        self.dropout = nnx.Dropout(rate=self.dropout_rate, rngs=rngs)

    def __call__(self, x, *, rngs: nnx.Rngs | None = None):
        x = self.emb(x)
        if self.dropout_rate <= 0:
            return x
        if rngs is None:
            # None rngs => deterministic path (e.g., evaluation).
            return self.dropout(x, deterministic=True)
        return self.dropout(x, rngs=rngs)

    def weights(self) -> jnp.ndarray:
        return jnp.asarray(self.emb.embedding[...])

    def apply_constrainer(self, constrainer: Callable[[Array], Array] | None) -> None:
        if constrainer is None:
            return
        constrained = constrainer(jnp.asarray(self.emb.embedding))
        self.emb.embedding.set_value(constrained)
