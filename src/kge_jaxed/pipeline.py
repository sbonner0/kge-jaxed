from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import optax
import pandas as pd
from flax import nnx

from kge_jaxed.datasets.base import BaseDataset
from kge_jaxed.datasets.pykeen_datasets import PyKEENDataset
from kge_jaxed.evaluation.metrics import compute_metrics_dataframe
from kge_jaxed.evaluation.utils import rank_triple
from kge_jaxed.negative_sampling.uniform_negative_sampling import (
    uniform_balanced_sampler,
)
from kge_jaxed.registries import LOSSES, MODELS
from kge_jaxed.rngs import RngManager, make_model_rngs

# ----------------------------- #
# JIT-friendly step function   #
# ----------------------------- #


@partial(nnx.jit, static_argnames=("num_negative_samples", "num_entities", "loss_fn"))
def train_step_fn(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    step_key: jax.Array,
    batch: jnp.ndarray,
    num_negative_samples: int,
    num_entities: int,
    loss_fn: Any,
) -> jnp.ndarray:
    """
    A JIT-optimised function to take a training step for KGE model. This function
    generates negative samples, computes the loss, and updates model parameters.


    :param model: KGE model
    :type model: nnx.Module
    :param optimizer: Optimizer bound to model parameters
    :type optimizer: nnx.Optimizer
    :param step_key: JAX random key for this training step
    :type step_key: jax.Array
    :param batch: Input batch of positive triples
    :type batch: jnp.ndarray
    :param num_negative_samples: Number of negative samples per positive
    :type num_negative_samples: int
    :param num_entities: Total number of entities in the knowledge graph
    :type num_entities: int
    :param loss_fn: Loss function to use
    :type loss_fn: Any
    :return: Computed loss value
    :rtype: jnp.ndarray
    """

    def loss_on_model(m: nnx.Module) -> jnp.ndarray:
        neg_key, dropout_key = jax.random.split(step_key, 2)
        dropout_rngs = nnx.Rngs(dropout=dropout_key)
        neg = uniform_balanced_sampler(
            triples=batch,
            num_entities=num_entities,
            k=num_negative_samples,
            key=neg_key,
        )
        neg = neg.reshape(-1, 3)

        return loss_fn(m, batch, neg, dropout_rngs=dropout_rngs)

    loss, grads = nnx.value_and_grad(loss_on_model)(model)
    optimizer.update(model, grads)

    return loss


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
        model_seed: int | None = None,
        dataset_seed: int | None = None,
        **model_kwargs: Any,
    ) -> None:
        """Initialize the KGE training pipeline."""

        self.negative_samples = int(negative_samples)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.model_seed = None if model_seed is None else int(model_seed)
        self.dataset_seed = int(self.seed if dataset_seed is None else dataset_seed)

        # Sort out dataset input
        if isinstance(dataset, str):
            self.dataset = PyKEENDataset(
                dataset_name=dataset,
                batch_size=train_batch_size,
                seed=self.dataset_seed,
            )
        elif isinstance(dataset, BaseDataset):
            self.dataset = dataset

        # Get model and loss from registries
        model_cls = MODELS[model_name]
        self.loss_fn = LOSSES[loss_name]

        # Keys: base -> split for init vs training
        # Dedicated RNGs (separate model init from training steps)
        self.rng_manager = RngManager(self.seed)
        init_rngs = self.rng_manager.init_rngs() if self.model_seed is None else make_model_rngs(self.model_seed)

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

    def _make_step_key(self, step: int, phase: int = 0) -> jax.Array:
        """
        Generate a JAX key for a specific step.
        phase=0 → train; phase=1 → eval (keeps streams disjoint).
        """
        return self.rng_manager.step_key(step, phase=phase)

    # -------- Training / eval loops -------- #

    def train(self, epochs: int = 100, log_every: int = 10) -> dict[str, Any]:
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

                # JITed train step
                loss = train_step_fn(
                    self.model,
                    self.optimizer,
                    step_key,
                    batch,
                    self.negative_samples,
                    self.dataset.num_entities,
                    self.loss_fn,
                )  # ignore: type

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

    def evaluate(
        self,
        split: str = "test",
        filtered: bool = True,
        max_triples: int | None = None,
        return_ranks_df: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """
        Evaluate model using ranking metrics (MRR, MR, Hits@K).

        :param split: Which split to evaluate on ('train', 'valid', 'test')
        :param filtered: Use filtered evaluation (exclude other known true triples)
        :param max_triples: Limit number of triples to evaluate (for speed)
        :param return_ranks_df: Return the evaluation triples with head/tail ranks
        :return: Metrics DataFrame, and optionally the ranked triples DataFrame
        """
        # Get test triples
        split_map = {
            "train": self.dataset.train_df,
            "valid": self.dataset.val_df,
            "test": self.dataset.test_df,
        }
        test_df = split_map[split]
        eval_df = test_df if max_triples is None else test_df.iloc[:max_triples]

        test_triples = jnp.array(eval_df.to_numpy(), dtype=jnp.int32)

        # Get all known triples for filtering
        filter_triples = None
        if filtered:
            all_triples = jnp.concatenate(
                [
                    jnp.array(self.dataset.train_df.to_numpy(), dtype=jnp.int32),
                    jnp.array(self.dataset.val_df.to_numpy(), dtype=jnp.int32),
                    jnp.array(self.dataset.test_df.to_numpy(), dtype=jnp.int32),
                ]
            )
            filter_triples = all_triples

        # Always evaluate both sides (head and tail)
        sides = ["tail", "head"]

        print(f"Evaluating on {len(test_triples)} triples (corruption: both, filtered: {filtered})...")

        head_ranks: list[int] = []
        tail_ranks: list[int] = []

        # Process each triple and compute ranks
        for i, triple in enumerate(test_triples):
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(test_triples)} triples...")

            for side in sides:
                rank = rank_triple(
                    self.model,
                    triple,
                    self.dataset.num_entities,
                    corruption_side=side,
                    filter_triples=filter_triples,
                )
                if side == "head":
                    head_ranks.append(int(rank))
                else:
                    tail_ranks.append(int(rank))

        # Compute metrics from ranks (per side + average)
        metrics_df = compute_metrics_dataframe(head_ranks, tail_ranks)

        print(f"\nRanking Results ({split} set, both corruption):")
        print(metrics_df)

        if not return_ranks_df:
            return metrics_df

        ranks_df = eval_df.copy()
        ranks_df["rank_head"] = head_ranks
        ranks_df["rank_tail"] = tail_ranks

        return metrics_df, ranks_df
