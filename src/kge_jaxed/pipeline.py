# kge_jaxed/trainer/pipeline.py
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from ..config import TrainConfig
from ..registries import LOSSES, MODELS, SAMPLERS


class Pipeline:
    def __init__(self, cfg: TrainConfig, key: jax.random.PRNGKey):
        self.cfg = cfg
        self.key = nnx.Rngs(jax.random.PRNGKey(cfg.seed))
        Model = MODELS.get(cfg.model)
        self.model = Model(cfg.num_entities, cfg.num_relations, cfg.embedding_dim, self.key)
        self.optimizer = nnx.Optimizer(self.model, nnx.adam(cfg.learning_rate), wrt=nnx.Param)
        self.metrics = nnx.MultiMetric(loss=nnx.metrics.Average())

        self.loss_fn = LOSSES.get(cfg.loss)
        self.sampler = SAMPLERS.get(cfg.sampler)

    def _score_batch(self, model, batch):
        return model.score(batch[:, 0], batch[:, 1], batch[:, 2])

    def _loss(self, model, batch, key):
        # positives
        pos_scores = self._score_batch(model, batch)  # [B]
        # negatives (on device)
        neg_triples, _ = self.sampler(batch, self.cfg.num_entities, self.cfg.num_negatives, key)
        B, K, _ = neg_triples.shape
        neg_scores = self._score_batch(model, neg_triples.reshape(B * K, 3)).reshape(B, K)
        # loss
        if self.cfg.loss == "mrl":
            return self.loss_fn(pos_scores, neg_scores, margin=self.cfg.margin)
        return self.loss_fn(pos_scores, neg_scores)

    @nnx.jit  # compiles the step incl. sampler
    def train_step(self, model, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch, key):
        loss, grads = nnx.value_and_grad(self._loss)(model, batch, key)
        metrics.update(loss=loss)
        optimizer.update(model, grads)
        return loss

    def fit(self, tf_dataset):
        # iterate positives from tf.data; everything else stays on device
        for epoch in range(self.cfg.epochs):
            for batch_np in tf_dataset:  # batch_np: (B,3) numpy from CPU
                batch = jnp.asarray(batch_np)  # move to device once
                self.key, k = self.key.split()
                self.train_step(self.model, self.optimizer, self.metrics, batch, k)
            print(f"epoch {epoch+1}: loss={self.metrics['loss'].compute():.4f}")
            self.metrics.reset()
