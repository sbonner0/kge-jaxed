import jax.numpy as jnp
import pandas as pd
import pytest

import kge_jaxed.pipeline as pipeline_module
from kge_jaxed.datasets.base import BaseDataset
from kge_jaxed.loss_functions.losses import self_adversarial_negative_sampling_loss
from kge_jaxed.models.transe import TransE
from kge_jaxed.pipeline import KGEPipeline
from kge_jaxed.rngs import make_model_rngs


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


def test_pipeline_accepts_prebuilt_model():
    dataset = DummyDataset()
    prebuilt_model = TransE(
        num_entities=dataset.num_entities,
        num_relations=dataset.num_relations,
        entity_embedding_dim=8,
        rngs=make_model_rngs(0),
    )

    pipeline = KGEPipeline(
        model=prebuilt_model,
        loss_name="mrl",
        dataset=dataset,
    )

    assert pipeline.model is prebuilt_model
    assert pipeline.model_name == "transe"


def test_pipeline_rejects_incompatible_prebuilt_model():
    dataset = DummyDataset()
    wrong_model = TransE(
        num_entities=dataset.num_entities + 1,
        num_relations=dataset.num_relations,
        entity_embedding_dim=8,
        rngs=make_model_rngs(0),
    )

    with pytest.raises(ValueError, match="num_entities"):
        KGEPipeline(
            model=wrong_model,
            loss_name="mrl",
            dataset=dataset,
        )


def test_pipeline_rejects_model_kwargs_with_instance():
    dataset = DummyDataset()
    prebuilt_model = TransE(
        num_entities=dataset.num_entities,
        num_relations=dataset.num_relations,
        entity_embedding_dim=8,
        rngs=make_model_rngs(0),
    )

    with pytest.raises(ValueError, match="model_kwargs"):
        KGEPipeline(
            model=prebuilt_model,
            loss_name="mrl",
            dataset=dataset,
            model_kwargs={"foo": "bar"},
        )


def test_pipeline_accepts_model_name_string():
    dataset = DummyDataset()

    pipeline = KGEPipeline(
        model="transe",
        loss_name="mrl",
        dataset=dataset,
        embedding_dim=8,
    )

    assert pipeline.model_name == "transe"


def test_pipeline_forwards_dataset_kwargs_for_dataset_name(monkeypatch):
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

    monkeypatch.setattr(pipeline_module, "PyKEENDataset", StubPyKEENDataset)

    pipeline = KGEPipeline(
        model="transe",
        loss_name="mrl",
        dataset="dummy",
        dataset_kwargs={"batch_size": 7, "shuffle": False},
        seed=11,
    )

    assert captured["dataset_name"] == "dummy"
    assert captured["batch_size"] == 7
    assert captured["shuffle"] is False
    assert captured["seed"] == 11
    assert pipeline.dataset.batch_size == 7


def test_pipeline_rejects_dataset_kwargs_with_dataset_instance():
    dataset = DummyDataset()

    with pytest.raises(ValueError, match="dataset_kwargs"):
        KGEPipeline(
            model="transe",
            loss_name="mrl",
            dataset=dataset,
            dataset_kwargs={"batch_size": 7},
        )


def test_pipeline_accepts_nssa_loss_kwargs():
    dataset = DummyDataset()

    pipeline = KGEPipeline(
        model="transe",
        loss_name="nssa",
        dataset=dataset,
        embedding_dim=8,
        loss_kwargs={
            "adversarial_temperature": 1.2,
            "margin": 9.0,
        },
    )

    triples = jnp.array(dataset.train_df[["head", "relation", "tail"]].to_numpy())
    pos_scores = pipeline.model.score_hrt(triples)
    neg_scores = pos_scores[:, None]
    actual = pipeline.loss_fn(pos_scores, neg_scores)
    expected = self_adversarial_negative_sampling_loss(
        pos_scores,
        neg_scores,
        adversarial_temperature=1.2,
        margin=9.0,
    )
    assert actual == pytest.approx(float(expected))
