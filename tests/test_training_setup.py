import pandas as pd
import pytest

import kge_jaxed.training.setup_training as training_setup
from kge_jaxed.datasets.base import BaseDataset
from kge_jaxed.models.transe import TransE
from kge_jaxed.rngs import RngManager, make_model_rngs
from kge_jaxed.training.setup_training import resolve_dataset, resolve_model


class DummyDataset(BaseDataset):
    def __init__(self) -> None:
        super().__init__(batch_size=2, shuffle=False, seed=0)
        self.train_df = pd.DataFrame([(0, 0, 1)], columns=["head", "relation", "tail"]).astype("int32")
        self.val_df = pd.DataFrame([(1, 0, 2)], columns=["head", "relation", "tail"]).astype("int32")
        self.test_df = pd.DataFrame([(2, 0, 3)], columns=["head", "relation", "tail"]).astype("int32")
        self.num_entities = 4
        self.num_relations = 1

    def load_data(self) -> None:
        return None


def test_resolve_dataset_forwards_seed_and_kwargs(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class StubPyKEENDataset(DummyDataset):
        def __init__(self, dataset_name: str, batch_size: int = 32, shuffle: bool = True, seed: int = 0) -> None:
            super().__init__()
            captured["dataset_name"] = dataset_name
            captured["batch_size"] = batch_size
            captured["shuffle"] = shuffle
            captured["seed"] = seed
            self.dataset_name = dataset_name
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.seed = seed

    monkeypatch.setattr(training_setup, "PyKEENDataset", StubPyKEENDataset)

    dataset, dataset_name = resolve_dataset(
        "dummy",
        {"batch_size": 7, "shuffle": False},
        seed=11,
    )

    assert captured == {
        "dataset_name": "dummy",
        "batch_size": 7,
        "shuffle": False,
        "seed": 11,
    }
    assert dataset_name == "dummy"
    assert dataset.batch_size == 7


def test_resolve_dataset_rejects_kwargs_with_dataset_instance() -> None:
    with pytest.raises(ValueError, match="dataset_kwargs"):
        resolve_dataset(DummyDataset(), {"batch_size": 7}, seed=0)


def test_resolve_model_rejects_kwargs_with_prebuilt_model() -> None:
    dataset = DummyDataset()
    prebuilt_model = TransE(
        num_entities=dataset.num_entities,
        num_relations=dataset.num_relations,
        entity_embedding_dim=8,
        rngs=make_model_rngs(0),
    )

    with pytest.raises(ValueError, match="model_kwargs"):
        resolve_model(
            prebuilt_model,
            {"foo": "bar"},
            8,
            dataset=dataset,
            rng_manager=RngManager(0),
        )


def test_resolve_model_rejects_incompatible_prebuilt_model() -> None:
    dataset = DummyDataset()
    wrong_model = TransE(
        num_entities=dataset.num_entities + 1,
        num_relations=dataset.num_relations,
        entity_embedding_dim=8,
        rngs=make_model_rngs(0),
    )

    with pytest.raises(ValueError, match="num_entities"):
        resolve_model(
            wrong_model,
            {},
            8,
            dataset=dataset,
            rng_manager=RngManager(0),
        )


def test_resolve_model_rejects_relation_mismatch() -> None:
    dataset = DummyDataset()
    wrong_model = TransE(
        num_entities=dataset.num_entities,
        num_relations=dataset.num_relations + 1,
        entity_embedding_dim=8,
        rngs=make_model_rngs(0),
    )

    with pytest.raises(ValueError, match="num_relations"):
        resolve_model(
            wrong_model,
            {},
            8,
            dataset=dataset,
            rng_manager=RngManager(0),
        )
