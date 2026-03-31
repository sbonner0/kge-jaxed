"""Setup helpers for pipeline construction and checkpoint metadata."""

from typing import Any

from flax import nnx

from kge_jaxed.datasets.base import BaseDataset
from kge_jaxed.datasets.pykeen_datasets import PyKEENDataset
from kge_jaxed.models.base_kge import BaseKGE
from kge_jaxed.registries import get_model, get_optimizer
from kge_jaxed.rngs import RngManager


def resolve_dataset(dataset: str | BaseDataset, dataset_kwargs: dict[str, Any]) -> tuple[BaseDataset, str | None]:
    """Resolve a dataset input into a dataset instance and label."""
    if isinstance(dataset, str):
        resolved_dataset_kwargs = dict(dataset_kwargs)
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


def resolve_model(
    model: str | BaseKGE,
    model_kwargs: dict[str, Any],
    embedding_dim: int,
    *,
    dataset: BaseDataset,
    rng_manager: RngManager,
) -> tuple[BaseKGE, str, int, dict[str, Any]]:
    """Resolve a model input into a model instance plus its tracked config."""
    if isinstance(model, str):
        model_name = model
        resolved_embedding_dim = int(embedding_dim)
        model_cls = get_model(model_name)
        resolved_model = model_cls(
            num_entities=dataset.num_entities,
            num_relations=dataset.num_relations,
            entity_embedding_dim=resolved_embedding_dim,
            rngs=rng_manager.init_rngs(),
            **model_kwargs,
        )
        return resolved_model, model_name, resolved_embedding_dim, dict(model_kwargs)

    if isinstance(model, BaseKGE):
        if model_kwargs:
            raise ValueError("model_kwargs is only supported when model is a string name")
        if getattr(model, "num_entities", dataset.num_entities) != dataset.num_entities:
            raise ValueError("Provided model num_entities does not match dataset.num_entities")
        if getattr(model, "num_relations", dataset.num_relations) != dataset.num_relations:
            raise ValueError("Provided model num_relations does not match dataset.num_relations")
        model_name = model.__class__.__name__.lower()
        resolved_embedding_dim = int(getattr(model, "entity_embedding_dim", embedding_dim))
        return model, model_name, resolved_embedding_dim, {}

    raise TypeError("model must be either a model name string or a BaseKGE instance")


def build_optimizer(
    model: BaseKGE,
    *,
    optimizer_name: str,
    learning_rate: float,
    optimizer_kwargs: dict[str, Any],
) -> nnx.Optimizer:
    """Create an optimizer bound to a model's parameters."""
    optimizer_factory = get_optimizer(optimizer_name)
    optimizer_transform = optimizer_factory(learning_rate, **optimizer_kwargs)
    return nnx.Optimizer(model, optimizer_transform, wrt=nnx.Param)


def build_checkpoint_metadata(
    *,
    model_name: str,
    embedding_dim: int,
    model_kwargs: dict[str, Any],
    dataset: BaseDataset,
    dataset_name: str | None,
    learning_rate: float,
    optimizer_name: str,
    optimizer_kwargs: dict[str, Any],
    loss_name: str,
    loss_kwargs: dict[str, Any],
    negative_samples: int,
) -> dict[str, Any]:
    """Assemble checkpoint metadata for config validation on load."""
    return {
        "model_name": model_name,
        "embedding_dim": int(embedding_dim),
        "model_kwargs": dict(model_kwargs),
        "dataset_name": dataset_name,
        "num_entities": dataset.num_entities,
        "num_relations": dataset.num_relations,
        "learning_rate": float(learning_rate),
        "optimizer_name": str(optimizer_name),
        "optimizer_kwargs": dict(optimizer_kwargs),
        "loss_name": str(loss_name),
        "loss_kwargs": dict(loss_kwargs),
        "negative_samples": int(negative_samples),
    }
