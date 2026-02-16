from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax import nnx

from kge_jaxed.datasets.base import BaseDataset
from kge_jaxed.datasets.pykeen_datasets import PyKEENDataset
from kge_jaxed.evaluation.grouped import (
    build_filter_map,
    build_group_maps,
    score_grouped_pairs,
)
from kge_jaxed.evaluation.metrics import compute_metrics_dataframe
from kge_jaxed.evaluation.validation import validate_eval_df
from kge_jaxed.models.base_kge import BaseKGE
from kge_jaxed.negative_sampling.uniform_negative_sampling import (
    uniform_balanced_sampler,
)
from kge_jaxed.registries import get_loss, get_model, get_optimizer
from kge_jaxed.rngs import RngManager
from kge_jaxed.training import checkpointing as ckpt

# ----------------------------- #
# JIT-friendly step function   #
# ----------------------------- #


def _score_pos_neg(
    model: BaseKGE,
    pos_batch: jnp.ndarray,
    neg_batch: jnp.ndarray,
    *,
    dropout_rngs: nnx.Rngs | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    if dropout_rngs is None:
        pos_scores = model.score_hrt(pos_batch)
    else:
        pos_scores = model.score_hrt(pos_batch, dropout_rngs=dropout_rngs)
    neg_flat = neg_batch.reshape(-1, 3)
    if dropout_rngs is None:
        neg_scores = model.score_hrt(neg_flat)
    else:
        neg_scores = model.score_hrt(neg_flat, dropout_rngs=dropout_rngs)
    neg_scores = neg_scores.reshape(neg_batch.shape[0], neg_batch.shape[1])
    return pos_scores, neg_scores


@partial(
    nnx.jit,
    static_argnames=(
        "num_negative_samples",
        "num_entities",
        "loss_fn",
    ),
)
def train_step_fn(
    model: BaseKGE,
    optimizer: nnx.Optimizer,
    step_key: jax.Array,
    batch: jnp.ndarray,
    num_negative_samples: int,
    num_entities: int,
    loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
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
    :param loss_fn: Loss function that consumes (pos_scores, neg_scores). The
        score convention is ``higher is better`` for both arrays.
    :type loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    :return: Computed loss value
    :rtype: jnp.ndarray
    """

    def loss_on_model(m: BaseKGE) -> jnp.ndarray:
        use_dropout = bool(getattr(m, "uses_dropout", False))
        if use_dropout:
            neg_key, dropout_key = jax.random.split(step_key, 2)
            dropout_rngs = nnx.Rngs(dropout=dropout_key)
        else:
            neg_key = step_key
            dropout_rngs = None

        # Generate negative samples
        neg = uniform_balanced_sampler(
            triples=batch,
            num_entities=num_entities,
            k=num_negative_samples,
            key=neg_key,
        )
        # Compute scores and loss
        pos_scores, neg_scores = _score_pos_neg(m, batch, neg, dropout_rngs=dropout_rngs)
        loss = loss_fn(pos_scores, neg_scores)

        # Add regularization loss if applicable
        loss = loss + m.regularization_loss()

        return loss

    loss, grads = nnx.value_and_grad(loss_on_model)(model)
    optimizer.update(model, grads)
    model.apply_constraints()

    return loss


# ----------------------------- #
# Pipeline class                #
# ----------------------------- #
class KGEPipeline:
    """Simple pipeline for training Knowledge Graph Embedding models (JAX + Flax NNX)."""

    def __init__(
        self,
        model: str | BaseKGE,
        dataset: str | BaseDataset,
        loss_name: str,
        model_kwargs: dict[str, Any] | None = None,
        dataset_kwargs: dict[str, Any] | None = None,
        embedding_dim: int = 128,
        negative_samples: int = 1,
        learning_rate: float = 1e-3,
        optimizer_name: str = "adam",
        optimizer_kwargs: dict[str, Any] | None = None,
        seed: int = 42,
        loss_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the KGE training pipeline."""

        model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
        dataset_kwargs = {} if dataset_kwargs is None else dict(dataset_kwargs)
        loss_kwargs = {} if loss_kwargs is None else dict(loss_kwargs)
        self.negative_samples = int(negative_samples)
        self.learning_rate = float(learning_rate)
        self.optimizer_name = str(optimizer_name)
        self.optimizer_kwargs = {} if optimizer_kwargs is None else dict(optimizer_kwargs)
        self.seed = int(seed)
        self.loss_kwargs = loss_kwargs

        # Training state (checkpoint-resumable)
        self.epoch = 0
        self.global_step = 0

        self.dataset, self.dataset_name = self._resolve_dataset(dataset, dataset_kwargs)

        # Loss from registry
        self.loss_fn = get_loss(loss_name, **self.loss_kwargs)

        self.rng_manager = RngManager(self.seed)

        self.model, self.model_name, self.embedding_dim, self.model_kwargs = self._resolve_model(
            model=model,
            model_kwargs=model_kwargs,
            embedding_dim=embedding_dim,
        )

        # Optimizer bound to NNX params
        self.optimizer = self._build_optimizer(self.model)

    def _resolve_dataset(self, dataset: str | BaseDataset, dataset_kwargs: dict[str, Any]) -> tuple[BaseDataset, str]:
        if isinstance(dataset, str):
            resolved_dataset_kwargs = dict(dataset_kwargs)
            resolved_dataset_kwargs.setdefault("seed", self.seed)
            resolved_dataset = PyKEENDataset(
                dataset_name=dataset,
                **resolved_dataset_kwargs,
            )
            return resolved_dataset, dataset
        if isinstance(dataset, BaseDataset):
            if dataset_kwargs:
                raise ValueError("dataset_kwargs is only supported when dataset is a string name")
            dataset_name = getattr(dataset, "dataset_name", "custom_dataset")
            return dataset, dataset_name
        raise TypeError("dataset must be a dataset name string or BaseDataset instance")

    def _resolve_model(
        self,
        model: str | BaseKGE,
        model_kwargs: dict[str, Any],
        embedding_dim: int,
    ) -> tuple[BaseKGE, str, int, dict[str, Any]]:
        if isinstance(model, str):
            model_name = model
            resolved_embedding_dim = int(embedding_dim)
            model_cls = get_model(model_name)
            resolved_model = model_cls(
                num_entities=self.dataset.num_entities,
                num_relations=self.dataset.num_relations,
                entity_embedding_dim=resolved_embedding_dim,
                rngs=self.rng_manager.init_rngs(),
                **model_kwargs,
            )
            return resolved_model, model_name, resolved_embedding_dim, model_kwargs

        if isinstance(model, BaseKGE):
            if model_kwargs:
                raise ValueError("model_kwargs is only supported when model is a string name")
            if getattr(model, "num_entities", self.dataset.num_entities) != self.dataset.num_entities:
                raise ValueError("Provided model num_entities does not match dataset.num_entities")
            if getattr(model, "num_relations", self.dataset.num_relations) != self.dataset.num_relations:
                raise ValueError("Provided model num_relations does not match dataset.num_relations")
            model_name = model.__class__.__name__.lower()
            resolved_embedding_dim = int(getattr(model, "entity_embedding_dim", embedding_dim))
            return model, model_name, resolved_embedding_dim, {}

        raise TypeError("model must be either a model name string or a BaseKGE instance")

    def _build_optimizer(self, model: BaseKGE) -> nnx.Optimizer:
        optimizer_factory = get_optimizer(self.optimizer_name)
        optimizer_transform = optimizer_factory(self.learning_rate, **self.optimizer_kwargs)
        return nnx.Optimizer(model, optimizer_transform, wrt=nnx.Param)

    # -------- RNG helpers -------- #

    def _make_step_key(self, step: int, phase: int = 0) -> jax.Array:
        """
        Generate a JAX key for a specific step.
        phase=0 → train; phase=1 → eval (keeps streams disjoint).
        """
        return self.rng_manager.step_key(step, phase=phase)

    # -------- Checkpointing -------- #

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "model_kwargs": self.model_kwargs,
            "dataset_name": self.dataset_name,
            "num_entities": self.dataset.num_entities,
            "num_relations": self.dataset.num_relations,
            "learning_rate": self.learning_rate,
            "optimizer_name": self.optimizer_name,
            "optimizer_kwargs": self.optimizer_kwargs,
        }

    def save_checkpoint(
        self,
        checkpoint_path: str,
        *,
        epoch: int | None = None,
        global_step: int | None = None,
    ) -> None:
        """
        Save model parameters and optimizer state to an Orbax checkpoint directory.

        :param checkpoint_path: Target checkpoint directory.
        :type checkpoint_path: str
        :param epoch: Optional current epoch to store in metadata.
        :type epoch: int | None
        :param global_step: Optional global step to store in metadata.
        :type global_step: int | None
        """
        metadata = self._checkpoint_metadata()
        if epoch is not None:
            metadata["epoch"] = int(epoch)
        if global_step is not None:
            metadata["global_step"] = int(global_step)
        ckpt.save_checkpoint(
            checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            metadata=metadata,
        )

    def load_checkpoint(self, checkpoint_path: str) -> dict[str, Any] | None:
        """
        Restore model parameters and optimizer state from an Orbax checkpoint directory.

        :param checkpoint_path: Source checkpoint directory.
        :type checkpoint_path: str
        :return: Stored metadata dict when present; otherwise None.
        :rtype: dict[str, Any] | None
        """

        def rebuild_optimizer(model: BaseKGE) -> nnx.Optimizer:
            return self._build_optimizer(model)

        self.model, self.optimizer, metadata = ckpt.load_checkpoint(
            checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            rebuild_optimizer=rebuild_optimizer,
            expected_metadata=self._checkpoint_metadata(),
            warn_metadata_keys={"learning_rate", "optimizer_name", "optimizer_kwargs"},
        )
        if metadata is not None:
            self.epoch = int(metadata.get("epoch", 0))
            self.global_step = int(metadata.get("global_step", 0))
        return metadata

    # -------- Training / eval loops -------- #

    def train(
        self,
        epochs: int = 100,
        log_every: int = 10,
        *,
        save_checkpoint_dir: str | None = None,
        save_every: int | None = None,
    ) -> dict[str, Any]:
        """
        Run the training loop for a fixed number of epochs.

        This loop uses deterministic RNGs derived from (seed, process, phase=0, global_step).
        It optionally saves checkpoints during training and always saves a final checkpoint
        when ``save_checkpoint_dir`` is provided. Saved checkpoints include the current
        epoch and global step in metadata. If a checkpoint was loaded, training resumes
        from the stored epoch and global step.

        :param epochs: Number of epochs to train for.
        :type epochs: int
        :param log_every: Print loss every N epochs.
        :type log_every: int
        :param save_checkpoint_dir: Directory to save checkpoints. If set, the checkpoint is overwritten
            in this directory.
        :type save_checkpoint_dir: str | None
        :param save_every: Save every N epochs. Requires ``save_checkpoint_dir`` to be set.
            When provided, the same directory is overwritten each time.
        :type save_every: int | None
        :return: Training summary including per-epoch loss and the RNG seed.
        :rtype: dict[str, Any]
        """
        if log_every <= 0:
            raise ValueError("log_every must be a positive integer")
        if save_every is not None and save_every <= 0:
            raise ValueError("save_every must be a positive integer")
        if save_every is not None and save_checkpoint_dir is None:
            raise ValueError("save_checkpoint_dir must be set when save_every is provided")

        checkpoint_path = Path(save_checkpoint_dir) if save_checkpoint_dir is not None else None
        train_losses: list[float] = []
        start_epoch = int(self.epoch)
        global_step = int(self.global_step)

        print(f"Starting training for {epochs} epochs (resume from epoch {start_epoch})...")

        # Training loop over epochs
        for epoch in range(start_epoch, start_epoch + int(epochs)):
            epoch_losses = []

            # Loop over training batches
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
                )  # type: ignore[call-arg]

                loss_value = float(jnp.asarray(loss))
                epoch_losses.append(loss_value)
                global_step += 1

            if epoch_losses:
                avg_loss = float(jnp.mean(jnp.array(epoch_losses)))
                train_losses.append(avg_loss)
            else:
                avg_loss = float("nan")

            if (epoch + 1) % int(log_every) == 0:
                print(f"Epoch {epoch + 1}/{start_epoch + epochs}, Loss: {avg_loss:.4f}")
            if checkpoint_path is not None and save_every is not None and (epoch + 1) % int(save_every) == 0:
                self.save_checkpoint(str(checkpoint_path), epoch=epoch + 1, global_step=global_step)

            self.epoch = epoch + 1
            self.global_step = global_step

        if checkpoint_path is not None:
            self.save_checkpoint(str(checkpoint_path), epoch=self.epoch, global_step=self.global_step)

        return {
            "train_losses": train_losses,
            "seed": self.seed,
        }

    def evaluate(
        self,
        split: str | None = "test",
        eval_df: pd.DataFrame | None = None,
        filtered: bool = True,
        eval_batch_size: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Evaluate model using ranking metrics (MRR, MR, Hits@K).

        This evaluation groups triples by shared (head, relation) and (relation, tail)
        pairs so each unique pair is scored once against all entities. The resulting
        score vectors are reused to compute ranks for all true tails/heads in that
        group, reducing redundant scoring. When filtered=True, known positives are
        removed from the ranking via precomputed filter maps.

        :param split: Which split to evaluate on ('train', 'valid', 'test'); set to None when eval_df is provided
        :type split: str | None
        :param eval_df: Optional evaluation DataFrame override (mutually exclusive with split)
        :type eval_df: pd.DataFrame | None
        :param filtered: Use filtered evaluation (exclude other known true triples)
        :type filtered: bool
        :param eval_batch_size: Batch size for grouped evaluation
        :type eval_batch_size: int | None
        :return: Metrics DataFrame and the ranked triples DataFrame
        :rtype: tuple[pd.DataFrame, pd.DataFrame]
        """
        # Get triples to evaluate on (validate if eval_df is provided, otherwise select by split)
        if eval_df is None:
            if split is None:
                raise ValueError("split must be provided when eval_df is not set")
            split_map = {
                "train": self.dataset.train_df,
                "valid": self.dataset.val_df,
                "test": self.dataset.test_df,
            }
            if split not in split_map:
                raise ValueError(f"Invalid split: {split}. Expected one of {list(split_map.keys())}.")
            eval_df = split_map[split]
        else:
            if split is not None:
                raise ValueError("Provide only split or eval_df, not both")
            validate_eval_df(eval_df, self.dataset.num_entities, self.dataset.num_relations)
        eval_df = eval_df.reset_index(drop=True)

        if eval_batch_size is None:
            eval_batch_size = self.dataset.batch_size

        # Build eval groups (unique keys -> row indices + true entities)
        tail_groups, tail_pairs = build_group_maps(eval_df, ["head", "relation"], "tail")
        head_groups, head_pairs = build_group_maps(eval_df, ["relation", "tail"], "head")

        # Build filter maps from all triples (for filtered evaluation)
        tail_filter_map: dict[tuple[int, int], np.ndarray] = {}
        head_filter_map: dict[tuple[int, int], np.ndarray] = {}
        if filtered:
            filter_triples = pd.concat(
                [self.dataset.train_df, self.dataset.val_df, self.dataset.test_df],
                ignore_index=True,
            )
            tail_filter_map = build_filter_map(filter_triples, ["head", "relation"], "tail")
            head_filter_map = build_filter_map(filter_triples, ["relation", "tail"], "head")

        # Score all unique pairs and assign ranks to true entities in each group for head and tail corruption
        tail_ranks, tail_scores = score_grouped_pairs(
            self.model,
            tail_pairs,
            tail_groups,
            tail_filter_map,
            "tail",
            self.dataset.num_entities,
            eval_batch_size,
        )
        head_ranks, head_scores = score_grouped_pairs(
            self.model,
            head_pairs,
            head_groups,
            head_filter_map,
            "head",
            self.dataset.num_entities,
            eval_batch_size,
        )

        # Compute metrics from ranks (per side + average)
        metrics_df = compute_metrics_dataframe(head_ranks, tail_ranks)

        print(f"\nRanking Results ({split} set, both corruption):")
        print(metrics_df)

        # Create a DataFrame with ranks and scores for all evaluated triples
        ranks_df = eval_df.copy()
        ranks_df["rank_head"] = head_ranks
        ranks_df["rank_tail"] = tail_ranks
        ranks_df["score_head"] = head_scores
        ranks_df["score_tail"] = tail_scores

        return metrics_df, ranks_df
