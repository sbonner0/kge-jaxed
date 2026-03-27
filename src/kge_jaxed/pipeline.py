"""Public training pipeline for KGE models."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import pandas as pd  # type: ignore[import]
from flax import nnx

from kge_jaxed.datasets.base import BaseDataset
from kge_jaxed.evaluation.metrics import compute_metrics_dataframe
from kge_jaxed.evaluation.ranking import (
    build_eval_filter_maps,
    evaluate_corruption_side,
    resolve_eval_dataframe,
)
from kge_jaxed.models.base_kge import BaseKGE
from kge_jaxed.registries import get_loss
from kge_jaxed.rngs import RngManager
from kge_jaxed.training import checkpointing as ckpt
from kge_jaxed.training.setup_training import (
    build_checkpoint_metadata,
    build_optimizer,
    resolve_dataset,
    resolve_model,
)
from kge_jaxed.training.steps import train_step_fn


class KGEPipeline:
    """
    Pipeline for training and evaluating Knowledge Graph Embedding models.

    The pipeline owns the dataset, model, optimizer, RNG manager, and resumable training state.
    It provides a small high-level API for training, evaluation, and checkpointing while delegating model construction
    and the JIT-compiled training step to lower-level helpers.
    """

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
        """
        Initialize the KGE training pipeline.

        ``model`` and ``dataset`` can each be provided either as prebuilt  instances or as string identifiers that are
        resolved by the library. The resulting pipeline keeps enough configuration state to rebuild optimizer
        state around checkpoints and to validate that a loaded checkpoint is compatible with the current setup.

        :param model: Model name to resolve from the registry, or a prebuilt
            ``BaseKGE`` instance.
        :type model: str | BaseKGE
        :param dataset: Dataset name to resolve through ``PyKEENDataset``, or a
            prebuilt ``BaseDataset`` instance.
        :type dataset: str | BaseDataset
        :param loss_name: Registered loss name used to create ``self.loss_fn``.
        :type loss_name: str
        :param model_kwargs: Keyword arguments forwarded when ``model`` is given
            as a string name.
        :type model_kwargs: dict[str, Any] | None, optional
        :param dataset_kwargs: Keyword arguments forwarded when ``dataset`` is
            given as a string name.
        :type dataset_kwargs: dict[str, Any] | None, optional
        :param embedding_dim: Entity embedding dimension used when constructing a
            model from a string name.
        :type embedding_dim: int, optional
        :param negative_samples: Number of negative samples to generate per
            positive triple during training.
        :type negative_samples: int, optional
        :param learning_rate: Optimizer learning rate.
        :type learning_rate: float, optional
        :param optimizer_name: Registered optimizer name used to build the NNX
            optimizer wrapper.
        :type optimizer_name: str, optional
        :param optimizer_kwargs: Extra keyword arguments forwarded to the
            optimizer factory.
        :type optimizer_kwargs: dict[str, Any] | None, optional
        :param seed: Base random seed used for model initialization and per-step
            training RNGs.
        :type seed: int, optional
        :param loss_kwargs: Extra keyword arguments forwarded when constructing
            the configured loss function.
        :type loss_kwargs: dict[str, Any] | None, optional
        """

        model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
        dataset_kwargs = {} if dataset_kwargs is None else dict(dataset_kwargs)
        loss_kwargs = {} if loss_kwargs is None else dict(loss_kwargs)

        self.negative_samples = int(negative_samples)
        self.learning_rate = float(learning_rate)
        self.optimizer_name = str(optimizer_name)
        self.optimizer_kwargs = {} if optimizer_kwargs is None else dict(optimizer_kwargs)
        self.seed = int(seed)
        self.loss_name = str(loss_name)
        self.loss_kwargs = loss_kwargs

        self.epoch = 0
        self.global_step = 0

        self.dataset, self.dataset_name = resolve_dataset(dataset, dataset_kwargs, seed=self.seed)
        self.loss_fn = get_loss(self.loss_name, **self.loss_kwargs)
        self.rng_manager = RngManager(self.seed)
        self.model, self.model_name, self.embedding_dim, self.model_kwargs = resolve_model(
            model,
            model_kwargs,
            embedding_dim,
            dataset=self.dataset,
            rng_manager=self.rng_manager,
        )
        self.optimizer = build_optimizer(
            self.model,
            optimizer_name=self.optimizer_name,
            learning_rate=self.learning_rate,
            optimizer_kwargs=self.optimizer_kwargs,
        )

    def _checkpoint_metadata(self) -> dict[str, Any]:
        """
        Build the configuration metadata stored alongside checkpoints.

        This metadata is used on load to verify that the current pipeline is compatible with the saved checkpoint, while
        still allowing selected optimizer hyperparameters to produce warnings instead of hard failures.
        """
        return build_checkpoint_metadata(
            model_name=self.model_name,
            embedding_dim=self.embedding_dim,
            model_kwargs=self.model_kwargs,
            dataset=self.dataset,
            dataset_name=self.dataset_name,
            learning_rate=self.learning_rate,
            optimizer_name=self.optimizer_name,
            optimizer_kwargs=self.optimizer_kwargs,
            loss_name=self.loss_name,
            loss_kwargs=self.loss_kwargs,
            negative_samples=self.negative_samples,
        )

    def save_checkpoint(
        self,
        checkpoint_path: str,
        *,
        epoch: int | None = None,
        global_step: int | None = None,
    ) -> None:
        """
        Save model parameters, optimizer state, and pipeline metadata.

        The checkpoint contains both the Orbax model/optimizer state and a JSON metadata file describing the pipeline
        configuration. When provided,``epoch`` and ``global_step`` are persisted into that metadata so a later call to
        ``load_checkpoint()`` can resume training progress counters.

        :param checkpoint_path: Target checkpoint directory.
        :type checkpoint_path: str
        :param epoch: Optional epoch value to write into checkpoint metadata.
        :type epoch: int | None
        :param global_step: Optional global training step to write into
            checkpoint metadata.
        :type global_step: int | None
        :return: None
        :rtype: None
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
        Restore model parameters, optimizer state, and resumable counters.

        Checkpoint metadata is validated against the current pipeline configuration before state is restored.
        If metadata is present, ``self.epoch`` and ``self.global_step`` are updated from the stored values so
        subsequent calls to ``train()`` resume from the checkpoint.

        :param checkpoint_path: Source checkpoint directory.
        :type checkpoint_path: str
        :return: The stored checkpoint metadata when present, otherwise ``None``.
        :rtype: dict[str, Any] | None
        """

        def rebuild_optimizer(model: BaseKGE) -> nnx.Optimizer:
            return build_optimizer(
                model,
                optimizer_name=self.optimizer_name,
                learning_rate=self.learning_rate,
                optimizer_kwargs=self.optimizer_kwargs,
            )

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

    def train(
        self,
        epochs: int = 100,
        log_every: int = 10,
        *,
        save_checkpoint_dir: str | None = None,
        save_every: int | None = None,
    ) -> dict[str, Any]:
        """
        Train the model for a fixed number of additional epochs.

        Training resumes from the pipeline's current ``epoch`` and ``global_step`` counters, so a pipeline that has
        loaded a checkpoint continues from the restored state rather than starting from zero again. Each training batch
        is converted to a JAX array, scored via the JIT-compiled training step, and aggregated into a mean loss for the
        epoch. When checkpointing is enabled, an intermediate checkpoint can be written every ``save_every`` epochs and
        a final checkpoint is always written at the end of training.

        :param epochs: Number of additional epochs to run.
        :type epochs: int
        :param log_every: Print the average epoch loss every N epochs.
        :type log_every: int
        :param save_checkpoint_dir: Optional checkpoint directory. If provided,
            the same directory is overwritten on each save.
        :type save_checkpoint_dir: str | None
        :param save_every: Optional checkpoint cadence in epochs. Requires
            ``save_checkpoint_dir`` to be set.
        :type save_every: int | None
        :return: Dictionary containing the mean loss for each completed epoch in
            ``train_losses`` and the pipeline RNG ``seed`` used for the run.
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
        epochs = int(epochs)
        start_epoch = int(self.epoch)
        global_step = int(self.global_step)

        print(f"Starting training for {epochs} epochs (resume from epoch {start_epoch})...")

        for epoch in range(start_epoch, start_epoch + epochs):
            loss_sum = jnp.array(0.0)
            num_batches = 0

            for batch in self.dataset.iter_batches("train", shuffle=True, seed=self.seed + epoch):
                batch_array = jnp.asarray(batch)
                step_key = self.rng_manager.step_key(global_step, phase=0)
                loss = train_step_fn(
                    self.model,
                    self.optimizer,
                    step_key,
                    batch_array,
                    self.negative_samples,
                    self.dataset.num_entities,
                    self.loss_fn,
                )  # type: ignore[call-arg]
                loss_sum = loss_sum + loss
                num_batches += 1
                global_step += 1

            if num_batches:
                avg_loss = float(jax.device_get(loss_sum / num_batches))
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
        ks: Sequence[int] = (1, 3, 5, 10),
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Evaluate the model with standard link-prediction ranking metrics.

        Evaluation scores both tail prediction and head prediction, computes ranks for each query,
        and then summarizes those ranks as MR, MRR, and Hits@K. You can evaluate one of the built-in dataset splits
        (``"train"``, ``"valid"``, or ``"test"``) or provide a custom dataframe of triples through ``eval_df``.
        In filtered mode, other known positive triples from the dataset are removed from the candidate rankings before
        ranks are computed.

        :param split: Dataset split to evaluate when ``eval_df`` is not
            provided.
        :type split: str | None
        :param eval_df: Optional dataframe of triples with columns ``head``,
            ``relation``, and ``tail``. When this is provided, ``split`` must be
            ``None``.
        :type eval_df: pd.DataFrame | None
        :param filtered: Whether to exclude other known true triples from the
            ranking candidates.
        :type filtered: bool
        :param eval_batch_size: Number of grouped evaluation queries to score at
            once. Defaults to ``dataset.batch_size``.
        :type eval_batch_size: int | None
        :param ks: Positive Hit@K thresholds to include in the metrics table.
        :type ks: Sequence[int]
        :return: Tuple ``(metrics_df, ranks_df)`` where ``metrics_df`` is indexed
            by metric with columns ``head``, ``tail``, and ``avg``, and
            ``ranks_df`` contains the evaluated triples plus ``rank_head``,
            ``rank_tail``, ``score_head``, and ``score_tail``.
        :rtype: tuple[pd.DataFrame, pd.DataFrame]
        """
        eval_df, split_label = resolve_eval_dataframe(self.dataset, split, eval_df)
        eval_batch_size = self.dataset.batch_size if eval_batch_size is None else int(eval_batch_size)
        ks = tuple(int(k) for k in ks)
        if not ks:
            raise ValueError("ks must contain at least one value")
        if any(k <= 0 for k in ks):
            raise ValueError("ks values must be positive integers")

        tail_filter_map, head_filter_map = build_eval_filter_maps(self.dataset, filtered)

        tail_ranks, tail_scores = evaluate_corruption_side(
            self.model,
            eval_df,
            group_cols=["head", "relation"],
            value_col="tail",
            filter_map=tail_filter_map,
            corruption_side="tail",
            num_entities=self.dataset.num_entities,
            eval_batch_size=eval_batch_size,
        )
        head_ranks, head_scores = evaluate_corruption_side(
            self.model,
            eval_df,
            group_cols=["relation", "tail"],
            value_col="head",
            filter_map=head_filter_map,
            corruption_side="head",
            num_entities=self.dataset.num_entities,
            eval_batch_size=eval_batch_size,
        )

        metrics_df = compute_metrics_dataframe(head_ranks, tail_ranks, ks=ks)
        ranks_df = eval_df.copy()
        ranks_df["rank_head"] = head_ranks
        ranks_df["rank_tail"] = tail_ranks
        ranks_df["score_head"] = head_scores
        ranks_df["score_tail"] = tail_scores

        print(f"\nRanking Results ({split_label} set, both corruption):")
        print(metrics_df)
        return metrics_df, ranks_df
