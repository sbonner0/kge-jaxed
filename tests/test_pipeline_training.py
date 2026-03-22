import json
import warnings

import jax
import jax.numpy as jnp
import pandas as pd
import pytest
from flax import nnx

from kge_jaxed.datasets.base import BaseDataset
from kge_jaxed.pipeline import KGEPipeline


class TrainingDataset(BaseDataset):
    def __init__(self) -> None:
        super().__init__(batch_size=2, shuffle=False, seed=0)
        self.train_df = (
            self._frame(
                [
                    (0, 0, 1),
                    (1, 0, 2),
                    (2, 0, 3),
                ]
            )
        )
        self.val_df = self._frame([(0, 0, 2)])
        self.test_df = self._frame([(1, 0, 3)])
        self.num_entities = 4
        self.num_relations = 1

    @staticmethod
    def _frame(rows: list[tuple[int, int, int]]):
        return pd.DataFrame(rows, columns=["head", "relation", "tail"]).astype("int32")

    def load_data(self) -> None:
        return None


def _param_leaves(model: nnx.Module) -> list[jax.Array]:
    state = nnx.state(model, nnx.Param)
    return jax.tree_util.tree_leaves(state)


def _optimizer_state_leaves(optimizer: nnx.Optimizer) -> list[jax.Array]:
    opt_state = nnx.state(optimizer, nnx.OptState)
    pure_opt_state = nnx.to_pure_dict(opt_state)
    return jax.tree_util.tree_leaves(pure_opt_state)


def _assert_array_leaves_equal(actual: list[jax.Array], expected: list[jax.Array]) -> None:
    assert len(actual) == len(expected)
    assert all(jnp.array_equal(a, b) for a, b in zip(actual, expected))


def test_pipeline_train_validates_arguments(tmp_path) -> None:
    pipeline = KGEPipeline(
        model="transe",
        loss_name="mrl",
        dataset=TrainingDataset(),
        embedding_dim=8,
        seed=0,
    )

    with pytest.raises(ValueError, match="log_every"):
        pipeline.train(log_every=0)
    with pytest.raises(ValueError, match="save_every"):
        pipeline.train(save_checkpoint_dir=str(tmp_path / "ckpt"), save_every=0)
    with pytest.raises(ValueError, match="save_checkpoint_dir"):
        pipeline.train(save_every=1)


def test_pipeline_train_updates_epoch_and_global_step() -> None:
    pipeline = KGEPipeline(
        model="transe",
        loss_name="mrl",
        dataset=TrainingDataset(),
        embedding_dim=8,
        seed=0,
    )

    summary = pipeline.train(epochs=2, log_every=10)

    assert pipeline.epoch == 2
    assert pipeline.global_step == 4
    assert len(summary["train_losses"]) == 2
    assert summary["seed"] == 0


def test_pipeline_checkpoint_metadata_includes_resume_config(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    pipeline = KGEPipeline(
        model="transe",
        loss_name="nssa",
        dataset=TrainingDataset(),
        embedding_dim=8,
        negative_samples=3,
        learning_rate=0.05,
        loss_kwargs={"adversarial_temperature": 1.2, "margin": 9.0},
        seed=0,
    )

    pipeline.save_checkpoint(str(checkpoint_path), epoch=4, global_step=9)

    metadata = json.loads((checkpoint_path / "metadata.json").read_text(encoding="utf-8"))

    assert metadata["loss_name"] == "nssa"
    assert metadata["loss_kwargs"] == {"adversarial_temperature": 1.2, "margin": 9.0}
    assert metadata["negative_samples"] == 3
    assert metadata["epoch"] == 4
    assert metadata["global_step"] == 9


def test_pipeline_checkpoint_rejects_resume_config_mismatch(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    pipeline = KGEPipeline(
        model="transe",
        loss_name="mrl",
        dataset=TrainingDataset(),
        embedding_dim=8,
        negative_samples=2,
        seed=0,
    )
    pipeline.save_checkpoint(str(checkpoint_path))

    mismatch_pipeline = KGEPipeline(
        model="transe",
        loss_name="mrl",
        dataset=TrainingDataset(),
        embedding_dim=8,
        negative_samples=3,
        seed=0,
    )

    with pytest.raises(ValueError, match="negative_samples"):
        mismatch_pipeline.load_checkpoint(str(checkpoint_path))


def test_pipeline_checkpoint_load_restores_model_optimizer_and_counters(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    pipeline = KGEPipeline(
        model="transe",
        loss_name="mrl",
        dataset=TrainingDataset(),
        embedding_dim=8,
        seed=0,
    )
    pipeline.train(epochs=1, log_every=10)
    pipeline.save_checkpoint(str(checkpoint_path), epoch=pipeline.epoch, global_step=pipeline.global_step)

    saved_param_leaves = _param_leaves(pipeline.model)
    saved_opt_state_leaves = _optimizer_state_leaves(pipeline.optimizer)
    saved_epoch = pipeline.epoch
    saved_global_step = pipeline.global_step

    pipeline.train(epochs=1, log_every=10)
    assert any(not jnp.array_equal(a, b) for a, b in zip(_param_leaves(pipeline.model), saved_param_leaves))

    restored_pipeline = KGEPipeline(
        model="transe",
        loss_name="mrl",
        dataset=TrainingDataset(),
        embedding_dim=8,
        seed=0,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        metadata = restored_pipeline.load_checkpoint(str(checkpoint_path))

    assert metadata is not None
    assert restored_pipeline.epoch == saved_epoch
    assert restored_pipeline.global_step == saved_global_step
    assert not any("Sharding info not provided when restoring" in str(w.message) for w in caught)
    _assert_array_leaves_equal(_param_leaves(restored_pipeline.model), saved_param_leaves)
    _assert_array_leaves_equal(_optimizer_state_leaves(restored_pipeline.optimizer), saved_opt_state_leaves)


def test_pipeline_checkpoint_resume_matches_uninterrupted_training(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    uninterrupted_pipeline = KGEPipeline(
        model="transe",
        loss_name="mrl",
        dataset=TrainingDataset(),
        embedding_dim=8,
        seed=0,
    )
    uninterrupted_pipeline.train(epochs=2, log_every=10)

    resumed_source_pipeline = KGEPipeline(
        model="transe",
        loss_name="mrl",
        dataset=TrainingDataset(),
        embedding_dim=8,
        seed=0,
    )
    resumed_source_pipeline.train(epochs=1, log_every=10, save_checkpoint_dir=str(checkpoint_path))

    resumed_pipeline = KGEPipeline(
        model="transe",
        loss_name="mrl",
        dataset=TrainingDataset(),
        embedding_dim=8,
        seed=0,
    )
    resumed_pipeline.load_checkpoint(str(checkpoint_path))
    resumed_pipeline.train(epochs=1, log_every=10)

    assert resumed_pipeline.epoch == uninterrupted_pipeline.epoch == 2
    assert resumed_pipeline.global_step == uninterrupted_pipeline.global_step == 4
    _assert_array_leaves_equal(_param_leaves(resumed_pipeline.model), _param_leaves(uninterrupted_pipeline.model))
    _assert_array_leaves_equal(
        _optimizer_state_leaves(resumed_pipeline.optimizer),
        _optimizer_state_leaves(uninterrupted_pipeline.optimizer),
    )


def test_pipeline_checkpoint_load_warns_on_learning_rate_mismatch(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    pipeline = KGEPipeline(
        model="transe",
        loss_name="mrl",
        dataset=TrainingDataset(),
        embedding_dim=8,
        learning_rate=0.01,
        seed=0,
    )
    pipeline.train(epochs=1, log_every=10)
    pipeline.save_checkpoint(str(checkpoint_path), epoch=pipeline.epoch, global_step=pipeline.global_step)

    saved_param_leaves = _param_leaves(pipeline.model)

    restored_pipeline = KGEPipeline(
        model="transe",
        loss_name="mrl",
        dataset=TrainingDataset(),
        embedding_dim=8,
        learning_rate=0.02,
        seed=0,
    )

    with pytest.warns(UserWarning, match="learning_rate"):
        metadata = restored_pipeline.load_checkpoint(str(checkpoint_path))

    assert metadata is not None
    _assert_array_leaves_equal(_param_leaves(restored_pipeline.model), saved_param_leaves)


def test_pipeline_checkpoint_load_without_metadata_restores_state(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    pipeline = KGEPipeline(
        model="transe",
        loss_name="mrl",
        dataset=TrainingDataset(),
        embedding_dim=8,
        seed=0,
    )
    pipeline.train(epochs=1, log_every=10)
    pipeline.save_checkpoint(str(checkpoint_path), epoch=pipeline.epoch, global_step=pipeline.global_step)

    saved_param_leaves = _param_leaves(pipeline.model)
    metadata_path = checkpoint_path / "metadata.json"
    metadata_path.unlink()

    restored_pipeline = KGEPipeline(
        model="transe",
        loss_name="mrl",
        dataset=TrainingDataset(),
        embedding_dim=8,
        seed=0,
    )
    metadata = restored_pipeline.load_checkpoint(str(checkpoint_path))

    assert metadata is None
    assert restored_pipeline.epoch == 0
    assert restored_pipeline.global_step == 0
    _assert_array_leaves_equal(_param_leaves(restored_pipeline.model), saved_param_leaves)
