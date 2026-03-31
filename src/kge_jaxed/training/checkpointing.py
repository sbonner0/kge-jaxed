"""Orbax checkpoint helpers for NNX models."""

import json
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
from flax import nnx
from orbax import checkpoint as ocp  # type: ignore[import]

type CheckpointMetadata = dict[str, Any]
type PureStateDict = dict[str, Any]

METADATA_FILENAME = "metadata.json"


def _make_restore_arg(value: Any) -> ocp.RestoreArgs:
    """
    Build the Orbax restore arguments for a single checkpoint leaf.

    :param value: Example leaf from the current in-memory state tree.
    :type value: Any
    :return: Restore arguments describing how Orbax should materialize the leaf.
    :rtype: ocp.RestoreArgs
    """
    if isinstance(value, jax.Array):
        if type(value).__name__ == "PRNGKeyArray":
            storage_shape = jax.random.key_data(value).shape
            return ocp.ArrayRestoreArgs(
                restore_type=type(value),
                dtype=value.dtype,
                sharding=value.sharding,
                global_shape=storage_shape,
                shape=storage_shape,
                strict=False,
            )
        return ocp.ArrayRestoreArgs(
            restore_type=jax.Array,
            dtype=value.dtype,
            sharding=value.sharding,
            global_shape=value.shape,
            shape=value.shape,
        )
    return ocp.RestoreArgs(restore_type=type(value), dtype=getattr(value, "dtype", None))


def _build_restore_spec(state: nnx.State) -> tuple[PureStateDict, PureStateDict]:
    """
    Create the abstract tree and restore-args tree needed for Orbax restore.

    :param state: Current NNX state used as the restore template.
    :type state: nnx.State
    :return: Tuple of abstract state and restore-args trees.
    :rtype: tuple[dict[str, Any], dict[str, Any]]
    """
    pure_state = nnx.to_pure_dict(state)
    abstract_state = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=getattr(x, "sharding", None)),
        pure_state,
    )
    restore_args = jax.tree_util.tree_map(_make_restore_arg, pure_state)
    return abstract_state, restore_args


def _write_metadata(checkpoint_dir: Path, metadata: CheckpointMetadata) -> None:
    """
    Persist JSON metadata alongside a checkpoint.

    :param checkpoint_dir: Checkpoint directory that will contain the metadata file.
    :type checkpoint_dir: Path
    :param metadata: Serializable metadata dictionary to write.
    :type metadata: dict[str, Any]
    :return: None
    :rtype: None
    """
    metadata_path = checkpoint_dir / METADATA_FILENAME
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2, sort_keys=True)


def _read_metadata(checkpoint_dir: Path) -> CheckpointMetadata | None:
    """
    Read JSON metadata stored alongside a checkpoint.

    :param checkpoint_dir: Checkpoint directory containing the metadata file.
    :type checkpoint_dir: Path
    :return: Loaded metadata when present, otherwise ``None``.
    :rtype: dict[str, Any] | None
    """
    metadata_path = checkpoint_dir / METADATA_FILENAME
    if not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _validate_metadata(
    metadata: CheckpointMetadata,
    expected: CheckpointMetadata,
    *,
    warn_keys: set[str] | None = None,
) -> None:
    """
    Validate stored metadata against the current pipeline configuration.

    :param metadata: Metadata loaded from the checkpoint.
    :type metadata: dict[str, Any]
    :param expected: Metadata describing the current in-memory configuration.
    :type expected: dict[str, Any]
    :param warn_keys: Metadata keys that should emit warnings instead of errors on mismatch.
    :type warn_keys: set[str] | None
    :raises ValueError: If any non-warning metadata entries do not match.
    :return: None
    :rtype: None
    """
    mismatches: list[str] = []
    warnings_list: list[str] = []
    for key, expected_value in expected.items():
        actual_value = metadata.get(key)
        if actual_value != expected_value:
            if warn_keys is not None and key in warn_keys:
                warnings_list.append(f"{key}: expected {expected_value}, got {actual_value}")
            else:
                mismatches.append(f"{key}: expected {expected_value}, got {actual_value}")
    if mismatches:
        msg = "Checkpoint metadata does not match current pipeline configuration:\n" + "\n".join(mismatches)
        raise ValueError(msg)
    if warnings_list:
        warnings.warn(
            "Checkpoint metadata differs from current pipeline configuration:\n" + "\n".join(warnings_list),
            stacklevel=2,
        )


def write_checkpoint(
    checkpoint_path: str,
    *,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    metadata: CheckpointMetadata,
) -> None:
    """
    Save model parameters, optimizer state, and checkpoint metadata.

    :param checkpoint_path: Target Orbax checkpoint directory.
    :type checkpoint_path: str
    :param model: NNX model whose parameters should be saved.
    :type model: nnx.Module
    :param optimizer: Optimizer whose ``OptState`` should be saved.
    :type optimizer: nnx.Optimizer
    :param metadata: Serializable metadata to store next to the checkpoint.
    :type metadata: dict[str, Any]
    :return: None
    :rtype: None
    """
    checkpoint_dir = Path(checkpoint_path).resolve()
    checkpointer = ocp.PyTreeCheckpointer()
    _, model_state = nnx.split(model)
    optimizer_state = nnx.state(optimizer, nnx.OptState)
    item = {
        "model": nnx.to_pure_dict(model_state),
        "optimizer": nnx.to_pure_dict(optimizer_state),
    }
    checkpointer.save(str(checkpoint_dir), item, force=True)
    _write_metadata(checkpoint_dir, metadata)


def restore_checkpoint[TModel: nnx.Module](
    checkpoint_path: str,
    *,
    model: TModel,
    optimizer: nnx.Optimizer | None = None,
    rebuild_optimizer: Callable[[TModel], nnx.Optimizer] | None = None,
    restore_optimizer_state: bool = True,
    expected_metadata: CheckpointMetadata | None = None,
    warn_metadata_keys: set[str] | None = None,
) -> tuple[TModel, nnx.Optimizer | None, CheckpointMetadata | None]:
    """
    Restore model parameters and, optionally, optimizer state from an Orbax checkpoint directory.

    :param checkpoint_path: Source Orbax checkpoint directory.
    :type checkpoint_path: str
    :param model: Model instance whose graph structure defines the restore target.
    :type model: TModel
    :param optimizer: Optional optimizer instance used as the template for optimizer-state restoration.
    :type optimizer: nnx.Optimizer | None
    :param rebuild_optimizer: Factory used to bind a fresh optimizer to the restored model before injecting restored
        optimizer state or to rebuild a fresh optimizer after model-only restoration.
    :type rebuild_optimizer: Callable[[TModel], nnx.Optimizer] | None
    :param restore_optimizer_state: Whether to restore the saved optimizer state. When ``False``, only model weights
        are restored unless both ``optimizer`` and ``rebuild_optimizer`` are provided, in which case a fresh optimizer
        is rebuilt around the restored model.
    :type restore_optimizer_state: bool
    :param expected_metadata: Optional metadata dict to validate against the stored metadata.
    :type expected_metadata: dict[str, Any] | None
    :param warn_metadata_keys: Metadata keys that should warn instead of raising on mismatch.
    :type warn_metadata_keys: set[str] | None
    :return: Restored model, restored optimizer when requested or rebuilt, and stored metadata when present.
    :rtype: tuple[TModel, nnx.Optimizer | None, dict[str, Any] | None]
    """
    checkpoint_dir = Path(checkpoint_path).resolve()
    metadata = _read_metadata(checkpoint_dir)
    if metadata is not None and expected_metadata is not None:
        _validate_metadata(metadata, expected_metadata, warn_keys=warn_metadata_keys)
    if restore_optimizer_state and (optimizer is None or rebuild_optimizer is None):
        raise ValueError("optimizer and rebuild_optimizer are required when restore_optimizer_state=True")

    checkpointer = ocp.PyTreeCheckpointer()
    graphdef, state = nnx.split(model)
    model_item, model_restore_args = _build_restore_spec(state)
    item = {"model": model_item}
    restore_args = {"model": model_restore_args}
    if restore_optimizer_state and optimizer is not None:
        optimizer_state = nnx.state(optimizer, nnx.OptState)
        optimizer_item, optimizer_restore_args = _build_restore_spec(optimizer_state)
        item["optimizer"] = optimizer_item
        restore_args["optimizer"] = optimizer_restore_args

    restored = checkpointer.restore(
        str(checkpoint_dir),
        ocp.args.PyTreeRestore(
            item=item,
            restore_args=restore_args,
            partial_restore=not restore_optimizer_state,
        ),
    )
    restored_state = nnx.restore_int_paths(restored["model"])

    nnx.replace_by_pure_dict(state, restored_state)
    model = nnx.merge(graphdef, state)
    restored_optimizer: nnx.Optimizer | None = None
    if rebuild_optimizer is not None and (restore_optimizer_state or optimizer is not None):
        restored_optimizer = rebuild_optimizer(model)
    if restore_optimizer_state and restored_optimizer is not None:
        restored_opt_state = nnx.restore_int_paths(restored["optimizer"])
        optimizer_state = nnx.state(restored_optimizer, nnx.OptState)
        nnx.replace_by_pure_dict(optimizer_state, restored_opt_state)
        nnx.update(restored_optimizer, optimizer_state)

    return model, restored_optimizer, metadata
