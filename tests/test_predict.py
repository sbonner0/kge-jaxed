from pathlib import Path

import jax.numpy as jnp
import pandas as pd
import pytest

import kge_jaxed.predict as predict_module
from kge_jaxed.datasets.base import BaseDataset
from kge_jaxed.predict import KGEPredict


class PredictionDataset(BaseDataset):
    def __init__(self) -> None:
        super().__init__(batch_size=2, shuffle=False, seed=0)
        self.train_df = self._frame([(0, 0, 1)])
        self.val_df = self._frame([(2, 0, 1)])
        self.test_df = self._frame([(0, 0, 2)])
        self.num_entities = 3
        self.num_relations = 1

        # Keep insertion order intentionally different from ID order so the
        # test catches regressions in entity labeling.
        self.entity_to_id = {
            "entity_2": 2,
            "entity_0": 0,
            "entity_1": 1,
        }
        self.id_to_entity = {idx: entity for entity, idx in self.entity_to_id.items()}
        self.relation_to_id = {"likes": 0}
        self.id_to_relation = {0: "likes"}
        self.split_array_calls = {"train": 0, "val": 0, "test": 0}

    @staticmethod
    def _frame(rows: list[tuple[int, int, int]]) -> pd.DataFrame:
        return pd.DataFrame(rows, columns=["head", "relation", "tail"]).astype("int32")

    def load_data(self) -> None:
        return None

    def split_array(self, split="train", *, columns=("head", "relation", "tail")):
        self.split_array_calls[split] += 1
        return super().split_array(split, columns=columns)


class DummyModel:
    def score_hrt(self, triples):
        triples = jnp.asarray(triples, dtype=jnp.float32)
        return triples[:, 0] * 10.0 + triples[:, 2]


@pytest.fixture
def predictor(monkeypatch) -> tuple[KGEPredict, PredictionDataset]:
    dataset = PredictionDataset()
    model = DummyModel()
    metadata = {
        "dataset_name": "dummy",
        "dataset_kwargs": {},
        "model_name": "transe",
        "model_kwargs": {},
        "embedding_dim": 8,
    }

    monkeypatch.setattr(predict_module.ckpt, "_read_metadata", lambda checkpoint_dir: metadata)
    monkeypatch.setattr(predict_module, "resolve_dataset", lambda dataset_name, dataset_kwargs: (dataset, dataset_name))
    monkeypatch.setattr(
        predict_module,
        "resolve_model",
        lambda model_name, model_kwargs, embedding_dim, *, dataset, rng_manager: (
            model,
            model_name,
            embedding_dim,
            model_kwargs,
        ),
    )
    monkeypatch.setattr(
        predict_module.ckpt,
        "restore_checkpoint",
        lambda checkpoint_path, *, model, restore_optimizer_state: (model, None, metadata),
    )

    return KGEPredict(Path("unused-checkpoint")), dataset


def test_predict_tail_orders_entities_by_id_and_marks_split_membership(predictor) -> None:
    helper, _ = predictor

    result = helper.predict("entity_0", "likes", "tail")

    assert list(result["entity"]) == ["entity_2", "entity_1", "entity_0"]
    assert list(result["score"]) == pytest.approx([2.0, 1.0, 0.0])
    assert list(result["in_train"]) == [False, True, False]
    assert list(result["in_test"]) == [True, False, False]
    assert list(result["in_val"]) == [False, False, False]


def test_predict_uses_precomputed_split_arrays(predictor) -> None:
    helper, dataset = predictor

    assert dataset.split_array_calls == {"train": 1, "val": 1, "test": 1}

    helper.predict("entity_0", "likes", "tail")
    helper.predict("entity_1", "likes", "head")

    assert dataset.split_array_calls == {"train": 1, "val": 1, "test": 1}


def test_predict_rejects_invalid_predict_side(predictor) -> None:
    helper, _ = predictor

    with pytest.raises(ValueError, match="predict_side"):
        helper.predict("entity_0", "likes", "middle")
