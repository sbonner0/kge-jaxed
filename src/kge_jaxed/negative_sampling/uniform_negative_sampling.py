import tensorflow as tf  # type: ignore


class UniformNegativeSampling:
    """
    A simplified class for generating negative samples for knowledge graph triples
    in a fully vectorized way, supporting any k >= 1.
    """

    def __init__(self, num_entities: int, k: int = 1) -> None:
        """
        Initialize the negative sampler.

        :param num_entities: Total number of entities in the knowledge graph.
        :type num_entities: int
        :param k: Total number of negatives per triple.
        :type k: int
        """
        self.num_entities = num_entities
        self.k = k

    def generate_negatives_batch(self, positives: tf.Tensor) -> tf.Tensor:
        """
        Generate negatives for a batch of positive triples.

        :param positives: Tensor of shape (batch_size, 3)
        :type positives: tf.Tensor
        :return: negatives Tensor of shape (batch_size, k, 3)
        :type positives: tf.Tensor
        """
        batch_size = tf.shape(positives)[0]
        k = self.k

        # Extract positive triple components.
        heads = positives[:, 0]  # shape: (batch_size,)
        relations = positives[:, 1]  # shape: (batch_size,)
        tails = positives[:, 2]  # shape: (batch_size,)

        # Tile positive components to shape (batch_size, k)
        heads_tiled = tf.tile(tf.expand_dims(heads, axis=-1), [1, k])
        relations_tiled = tf.tile(tf.expand_dims(relations, axis=-1), [1, k])
        tails_tiled = tf.tile(tf.expand_dims(tails, axis=-1), [1, k])

        # For each negative sample decide whether to corrupt the head or the tail.
        # coin_flip is True: corrupt head, False: corrupt tail.
        coin_flip = tf.random.uniform(shape=(batch_size, k), minval=0, maxval=1) < 0.5

        # Generate random values for head and tail.
        random_heads = tf.random.uniform(shape=(batch_size, k), minval=0, maxval=self.num_entities, dtype=tf.int32)
        random_tails = tf.random.uniform(shape=(batch_size, k), minval=0, maxval=self.num_entities, dtype=tf.int32)

        # Use random head when coin_flip is True; else keep the original head.
        new_heads = tf.where(coin_flip, random_heads, heads_tiled)
        # Use random tail when coin_flip is False; else keep the original tail.
        new_tails = tf.where(~coin_flip, random_tails, tails_tiled)

        # Construct negative triples: (new_head, original relation, new_tail).
        negatives = tf.stack([new_heads, relations_tiled, new_tails], axis=-1)
        return negatives
