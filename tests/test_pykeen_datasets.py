import numpy as np

import kge_jaxed.datasets.pykeen_datasets as pykeen_datasets
from kge_jaxed.datasets.pykeen_datasets import PyKEENDataset


class DummyMappedTriples:
    def __init__(self, triples: np.ndarray) -> None:
        self._triples = triples

    def numpy(self) -> np.ndarray:
        return self._triples


class DummyTriplesFactory:
    def __init__(self, triples: np.ndarray) -> None:
        self.mapped_triples = DummyMappedTriples(triples)


class DummyPyKEENDataset:
    def __init__(self) -> None:
        self.training = DummyTriplesFactory(np.array([[0, 0, 1]], dtype=np.int32))
        self.validation = DummyTriplesFactory(np.array([[1, 0, 2]], dtype=np.int32))
        self.testing = DummyTriplesFactory(np.array([[2, 0, 3]], dtype=np.int32))
        self.num_entities = 4
        self.num_relations = 1
        self.entity_to_id = {"a": 0, "b": 1, "c": 2, "d": 3}
        self.relation_to_id = {"r": 0}


def test_pykeen_dataset_passes_name_and_dataset_kwargs(monkeypatch) -> None:
    captured: dict[str, object] = {}
    pykeen_ds = DummyPyKEENDataset()

    def fake_get_dataset(*, dataset: str | None, dataset_kwargs: dict[str, object]) -> DummyPyKEENDataset:
        captured["dataset"] = dataset
        captured["dataset_kwargs"] = dataset_kwargs
        return pykeen_ds

    monkeypatch.setattr(pykeen_datasets, "get_dataset", fake_get_dataset)

    dataset = PyKEENDataset(
        dataset_name="nations",
        batch_size=7,
        shuffle=False,
        seed=11,
        pykeen_dataset_kwargs={"create_inverse_triples": True},
    )

    assert captured == {
        "dataset": "nations",
        "dataset_kwargs": {"create_inverse_triples": True},
    }
    assert dataset.dataset_name == "nations"
    assert dataset.batch_size == 7
    assert dataset.shuffle is False
    assert dataset.seed == 11
    assert dataset.num_entities == 4
    assert dataset.num_relations == 1
    assert dataset.id_to_entity == {0: "a", 1: "b", 2: "c", 3: "d"}
    assert dataset.id_to_relation == {0: "r"}
