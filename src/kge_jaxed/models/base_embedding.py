import jax.numpy as jnp
from flax import nnx
from flax.typing import Dtype


class BaseEmbedding(nnx.Module):
    """
    Base class for embeddings. Allows for easy extension of embeddings with dropout.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dropout_rate: float = 0.0,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(0),
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
        :param param_dtype: Data type for the parameters, defaults to jnp.float32
        :type param_dtype: Dtype, optional
        :param rngs: RNGs for the module, defaults to nnx.Rngs(0)
        :type rngs: nnx.Rngs, optional
        """

        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate

        self.emb = nnx.Embed(
            num_embeddings=self.num_embeddings,
            features=self.embedding_dim,
            param_dtype=param_dtype,
            rngs=rngs,
            **kwargs,
        )
        self.dropout = nnx.Dropout(rate=self.dropout_rate, rngs=rngs)

    def __call__(self, x):
        x = self.emb(x)
        x = self.dropout(x)
        return x
