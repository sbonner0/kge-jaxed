"""Orbax checkpoint helpers for NNX models."""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import jax
from flax import nnx
from orbax import checkpoint as ocp


def _abstract_pure_dict(state: nnx.State) -> dict[str, Any]:
    pure = nnx.to_pure_dict(state)
    return jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
        pure,
    )


def _write_metadata(checkpoint_path: str, metadata: dict[str, Any]) -> None:
    path = Path(checkpoint_path).resolve() / "metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def _read_metadata(checkpoint_path: str) -> dict[str, Any] | None:
    path = Path(checkpoint_path).resolve() / "metadata.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _validate_metadata(
    metadata: dict[str, Any],
    expected: dict[str, Any],
    *,
    warn_keys: set[str] | None = None,
) -> None:
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


def save_checkpoint(
    checkpoint_path: str,
    *,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    metadata: dict[str, Any],
) -> None:
    """Save model parameters and optimizer state to an Orbax checkpoint directory."""
    checkpoint_path = str(Path(checkpoint_path).resolve())
    checkpointer = ocp.PyTreeCheckpointer()
    _, state = nnx.split(model)
    model_state = nnx.to_pure_dict(state)
    opt_state = nnx.state(optimizer, nnx.OptState)
    opt_state = nnx.to_pure_dict(opt_state)
    checkpointer.save(checkpoint_path, {"model": model_state, "optimizer": opt_state}, force=True)
    _write_metadata(checkpoint_path, metadata)


TModel = TypeVar("TModel", bound=nnx.Module)


def load_checkpoint(
    checkpoint_path: str,
    *,
    model: TModel,
    optimizer: nnx.Optimizer,
    rebuild_optimizer: Callable[[TModel], nnx.Optimizer],
    expected_metadata: dict[str, Any] | None = None,
    warn_metadata_keys: set[str] | None = None,
) -> tuple[TModel, nnx.Optimizer, dict[str, Any] | None]:
    """
    Restore model parameters and optimizer state from an Orbax checkpoint directory.

    Returns the restored model, optimizer, and any stored metadata if present.
    """
    checkpoint_path = str(Path(checkpoint_path).resolve())
    metadata = _read_metadata(checkpoint_path)
    if metadata is not None and expected_metadata is not None:
        _validate_metadata(metadata, expected_metadata, warn_keys=warn_metadata_keys)

    checkpointer = ocp.PyTreeCheckpointer()
    graphdef, state = nnx.split(model)
    abstract_state = _abstract_pure_dict(state)
    opt_state = nnx.state(optimizer, nnx.OptState)
    abstract_opt_state = _abstract_pure_dict(opt_state)

    restored = checkpointer.restore(
        checkpoint_path,
        {"model": abstract_state, "optimizer": abstract_opt_state},
    )
    restored_state = nnx.restore_int_paths(restored["model"])
    restored_opt_state = nnx.restore_int_paths(restored["optimizer"])

    nnx.replace_by_pure_dict(state, restored_state)
    model = nnx.merge(graphdef, state)
    optimizer = rebuild_optimizer(model)
    new_opt_state = nnx.state(optimizer, nnx.OptState)
    nnx.replace_by_pure_dict(new_opt_state, restored_opt_state)

    return model, optimizer, metadata
