# kge_jaxed/trainer/pipeline.py
# import flax.nnx as nnx
# import jax
# import jax.numpy as jnp
# import numpy as np

# from ..config import TrainConfig
# from ..registries import LOSSES, MODELS, SAMPLERS


# class Pipeline:
#     def __init__(self, cfg: TrainConfig, key: jax.random.PRNGKey):
#         self.cfg = cfg
#         self.key = nnx.Rngs(jax.random.PRNGKey(cfg.seed))
#         Model = MODELS.get(cfg.model)
#         self.model = Model(cfg.num_entities, cfg.num_relations, cfg.embedding_dim, self.key)
#         self.optimizer = nnx.Optimizer(self.model, nnx.adam(cfg.learning_rate), wrt=nnx.Param)
#         self.metrics = nnx.MultiMetric(loss=nnx.metrics.Average())

#         self.loss_fn = LOSSES.get(cfg.loss)
#         self.sampler = SAMPLERS.get(cfg.sampler)

#     def _score_batch(self, model, batch):
#         return model.score(batch[:, 0], batch[:, 1], batch[:, 2])

#     def _loss(self, model, batch, key):
#         # positives
#         pos_scores = self._score_batch(model, batch)  # [B]
#         # negatives (on device)
#         neg_triples, _ = self.sampler(batch, self.cfg.num_entities, self.cfg.num_negatives, key)
#         B, K, _ = neg_triples.shape
#         neg_scores = self._score_batch(model, neg_triples.reshape(B * K, 3)).reshape(B, K)
#         # loss
#         if self.cfg.loss == "mrl":
#             return self.loss_fn(pos_scores, neg_scores, margin=self.cfg.margin)
#         return self.loss_fn(pos_scores, neg_scores)

#     @nnx.jit  # compiles the step incl. sampler
#     def train_step(self, model, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch, key):
#         loss, grads = nnx.value_and_grad(self._loss)(model, batch, key)
#         metrics.update(loss=loss)
#         optimizer.update(model, grads)
#         return loss

#     def fit(self, tf_dataset):
#         # iterate positives from tf.data; everything else stays on device
#         for epoch in range(self.cfg.epochs):
#             for batch_np in tf_dataset:  # batch_np: (B,3) numpy from CPU
#                 batch = jnp.asarray(batch_np)  # move to device once
#                 self.key, k = self.key.split()
#                 self.train_step(self.model, self.optimizer, self.metrics, batch, k)
#             print(f"epoch {epoch+1}: loss={self.metrics['loss'].compute():.4f}")
#             self.metrics.reset()


import time

# kge_jaxed/trainer.py
from dataclasses import dataclass
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from kge_jaxed.data.adapters import numpy_batch_to_device, tf_dataset_to_numpy_iterator
from kge_jaxed.registries import LOSSES, MODELS, SAMPLERS


@dataclass
class TrainerConfig:
    model_name: str = "transe"
    loss_name: str = "mrl"
    sampler_name: str = "uniform_balanced"
    embedding_dim: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    k_negatives: int = 64
    num_entities: int = 0
    num_relations: int = 0
    seed: int = 0
    # JIT friendliness: keep k static
    jit_static_k: bool = True


class Trainer:
    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg

        self.rngs = nnx.Rngs(nnx.make_rng("params", seed=cfg.seed), nnx.make_rng("dropout", seed=cfg.seed + 1))
        ModelCls = MODELS.get(cfg.model_name)
        self.model = ModelCls(
            rngs=self.rngs,
            num_entities=cfg.num_entities,
            num_relations=cfg.num_relations,
            embedding_dim=cfg.embedding_dim,
        )

        # Optax optimizer
        tx = optax.chain(
            optax.add_decayed_weights(cfg.weight_decay) if cfg.weight_decay > 0 else optax.identity(),
            optax.adam(cfg.learning_rate),
        )
        # NNX stateful optimizer wrapper
        self.opt = nnx.Optimizer(self.model, tx)

        self.loss_fn = LOSSES.get(cfg.loss_name)
        self.sampler_fn = SAMPLERS.get(cfg.sampler_name)

        # Compile steps
        self._compile_train_step()

    def _compile_train_step(self):
        k = self.cfg.k_negatives
        sampler_fn = self.sampler_fn
        loss_fn = self.loss_fn

        def loss_for_batch(model, pos_triples, neg_triples):
            return loss_fn(model, pos_triples, neg_triples)

        def train_step(opt: nnx.Optimizer, pos_triples: jnp.ndarray, key: jax.Array):
            model = opt.target
            # negatives on device
            neg_triples = sampler_fn(pos_triples, self.cfg.num_entities, k, key)

            # grads
            def _loss_wrap(model_):
                return loss_for_batch(model_, pos_triples, neg_triples)

            loss, grads = nnx.value_and_grad(_loss_wrap)(model)
            opt.update(grads)  # in-place update of model params
            return opt, loss

        # JIT compile (k is static inside sampler; we don’t need it static here)
        self.train_step = jax.jit(train_step)

    def fit(self, train_ds, val_ds=None, epochs: int = 1, log_every: int = 50):
        """
        train_ds: tf.data.Dataset yielding [B,3] int arrays (your BaseTFDataset pipelines).
        """
        step = 0
        rng = jax.random.PRNGKey(self.cfg.seed + 42)

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            losses = []
            for batch_np in tf_dataset_to_numpy_iterator(train_ds):
                pos = numpy_batch_to_device(batch_np)  # jnp[int32] [B,3]
                rng, sub = jax.random.split(rng)
                self.opt, loss = self.train_step(self.opt, pos, sub)
                losses.append(float(loss))
                step += 1
                if step % log_every == 0:
                    print(f"epoch {epoch} step {step} loss {sum(losses)/len(losses):.4f}")
            print(f"epoch {epoch} done in {time.time()-t0:.1f}s; mean loss={sum(losses)/max(1,len(losses)):.4f}")

            # if val_ds is not None:
            #     val_loss = self.evaluate(val_ds)
            #     print(f"[val] epoch {epoch} loss={val_loss:.4f}")

    # @jax.jit
    # def _eval_step(self, model, pos_triples):
    #     # minimal: score positives; if you want val negatives, replicate train path.
    #     return jnp.mean(model.score_batch(pos_triples))

    # def evaluate(self, val_ds) -> float:
    #     scores = []
    #     for batch_np in tf_dataset_to_numpy_iterator(val_ds):
    #         pos = numpy_batch_to_device(batch_np)
    #         scores.append(float(self._eval_step(self.model, pos)))
    #     # higher is better for raw scores; invert if you want a "loss-like" number
    #     return -(sum(scores) / max(1, len(scores)))

    # # --- persistence stubs (swap for Orbax/Flax checkpoints as needed) ---
    # def state_dict(self) -> Dict[str, Any]:
    #     # NNX has .state_dict()
    #     return {"model": nnx.state(self.model), "opt": nnx.state(self.opt)}

    # def load_state_dict(self, state):
    #     nnx.update(self.model, state["model"])
    #     nnx.update(self.opt, state["opt"])
