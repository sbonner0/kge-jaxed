from functools import partial
from typing import Any, Dict

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from kge_jaxed.datasets.base import BaseDataset
from kge_jaxed.datasets.pykeen_datasets import PyKEENDataset
from kge_jaxed.loss_functions.losses import bce_loss, margin_ranking_loss
from kge_jaxed.models.transe import TransE
from kge_jaxed.negative_sampling.uniform_negative_sampling import (
    uniform_balanced_sampler,
)
from kge_jaxed.registries import LOSSES, MODELS

# ----------------------------- #
# JIT-friendly step functions   #
# ----------------------------- #


@partial(nnx.jit, static_argnames=("negative_samples", "num_entities", "loss_fn"))
def train_step_fn(model, optimizer, neg_key, batch, negative_samples, num_entities, loss_fn) -> float:
    """Train step that takes a raw JAX key instead of nnx.Rngs"""

    def loss_on_model(m):
        neg = uniform_balanced_sampler(batch, num_entities, negative_samples, neg_key)
        neg = neg.reshape(-1, 3)

        return loss_fn(m, batch, neg)

    loss, grads = nnx.value_and_grad(loss_on_model)(model)
    optimizer.update(model, grads)
    return loss


@partial(nnx.jit, static_argnames=("negative_samples", "num_entities", "loss_fn"))
def eval_step_fn(model, neg_key, batch, *, negative_samples, num_entities, loss_fn):
    """Eval step that takes a raw JAX key"""
    neg = uniform_balanced_sampler(batch, num_entities, negative_samples, neg_key).reshape(-1, 3)
    return loss_fn(model, batch, neg)


# ----------------------------- #
# Pipeline class                #
# ----------------------------- #
class KGEPipeline:
    """Simple pipeline for training Knowledge Graph Embedding models (JAX + Flax NNX)."""

    def __init__(
        self,
        model_name: str,
        loss_name: str,
        dataset: BaseDataset | str,
        train_batch_size: int = 32,
        embedding_dim: int = 100,
        negative_samples: int = 1,
        learning_rate: float = 1e-3,
        seed: int = 42,
        **model_kwargs: Any,
    ):
        """Initialize the KGE training pipeline."""
        self.negative_samples = int(negative_samples)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)

        # Sort out dataset input
        if isinstance(dataset, str):
            self.dataset = PyKEENDataset(dataset_name=dataset, batch_size=train_batch_size)
        elif isinstance(dataset, BaseDataset):
            self.dataset = dataset

        # Get model and loss from registries
        model_cls = MODELS.get(model_name)
        self.loss_fn = LOSSES.get(loss_name)

        # Keys: base -> split for init vs training
        self.base_key = self._make_base_key(self.seed)
        self.init_key, self.train_key = jax.random.split(self.base_key)

        # Dedicated init RNGs (separate from training steps)
        init_rngs = self._make_init_rngs(self.init_key)

        # Build model
        self.model = model_cls(
            num_entities=self.dataset.num_entities,
            num_relations=self.dataset.num_relations,
            embedding_dim=embedding_dim,
            rngs=init_rngs,
            **model_kwargs,
        )

        # Optimizer bound to NNX params
        self.optimizer = nnx.Optimizer(self.model, optax.adam(self.learning_rate), wrt=nnx.Param)

    # -------- RNG helpers -------- #

    @staticmethod
    def _make_base_key(seed: int) -> jax.Array:
        """Seeded, multi-host-safe base key."""
        base = jax.random.PRNGKey(seed)
        return jax.random.fold_in(base, jax.process_index())

    @staticmethod
    def _make_init_rngs(init_key: jax.Array) -> nnx.Rngs:
        """One-off RNGs for parameter initialization (kept disjoint from training)."""
        k_params, k_dropout, k_neg = jax.random.split(init_key, 3)
        return nnx.Rngs(params=k_params, dropout=k_dropout, neg=k_neg)

    def _make_step_key(self, step: int, phase: int = 0) -> jax.Array:
        """
        Generate a JAX key for a specific step.
        phase=0 → train; phase=1 → eval (keeps streams disjoint).
        """
        k = jax.random.fold_in(self.train_key, phase)
        return jax.random.fold_in(k, int(step))

    # -------- Training / eval loops -------- #

    def train(self, epochs: int = 100, log_every: int = 10) -> Dict[str, Any]:
        """
        Train loop. Deterministic RNGs derived from (seed, process, phase=0, global_step).
        """
        train_losses: list[float] = []
        global_step = 0

        print(f"Starting training for {epochs} epochs...")

        for epoch in range(int(epochs)):
            epoch_losses = []

            for batch in self.dataset.iter_batches("train"):
                batch = jnp.array(batch)

                # Generate key for this step
                step_key = self._make_step_key(global_step, phase=0)

                # JITed step (mutates model via optimizer in-place)
                loss = train_step_fn(
                    self.model,
                    self.optimizer,
                    step_key,
                    batch,
                    self.negative_samples,
                    self.dataset.num_entities,
                    self.loss_fn,
                )

                epoch_losses.append(float(loss))
                global_step += 1

            avg_loss = float(jnp.mean(jnp.array(epoch_losses)))
            train_losses.append(avg_loss)

            if (epoch + 1) % int(log_every) == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        return {
            "train_losses": train_losses,
            "seed": self.seed,
        }

    def evaluate(self, split: str = "valid", max_batches: int | None = None) -> float:
        """
        Simple evaluation over a split using disjoint RNG phase (=1).
        Typically deterministic (no dropout).
        """
        losses = []
        for i, batch in enumerate(self.dataset.iter_batches(split)):
            if max_batches is not None and i >= max_batches:
                break
            batch = jnp.array(batch)

            # Generate key for eval step
            step_key = self._make_step_key(i, phase=1)

            loss = eval_step_fn(
                self.model,
                step_key,
                batch,
                negative_samples=self.negative_samples,
                num_entities=self.dataset.num_entities,
                loss_fn=self.loss_fn,
            )
            losses.append(float(loss))
        return float(jnp.mean(jnp.array(losses))) if losses else float("nan")
