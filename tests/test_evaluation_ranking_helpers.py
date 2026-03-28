import jax.numpy as jnp
import jax.tree_util as tree_util
import numpy as np
import pandas as pd
import pytest

from kge_jaxed.datasets.base import BaseDataset
from kge_jaxed.evaluation.ranking import (
    EvalFilterMaps,
    build_eval_filter_maps,
    evaluate_corruption_side,
    resolve_eval_dataframe,
)


class DummyDataset(BaseDataset):
    def __init__(self) -> None:
        super().__init__(batch_size=2, shuffle=False, seed=0)
        self.train_df = pd.DataFrame(
            [(0, 0, 1), (0, 0, 2), (3, 1, 4)],
            columns=["head", "relation", "tail"],
        ).astype("int32")
        self.val_df = pd.DataFrame(
            [(1, 0, 1)],
            columns=["head", "relation", "tail"],
        ).astype("int32")
        self.test_df = pd.DataFrame(
            [(0, 0, 3), (2, 1, 4)],
            columns=["head", "relation", "tail"],
        ).astype("int32")
        self.num_entities = 5
        self.num_relations = 2

    def load_data(self) -> None:
        return None


@tree_util.register_pytree_node_class
class DummyModel:
    def score_hrt(self, triples: jnp.ndarray) -> jnp.ndarray:
        h = triples[:, 0]
        r = triples[:, 1]
        t = triples[:, 2]
        return -(jnp.abs(h - t) * 2 + r)

    def tree_flatten(self):
        return (), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls()


def test_resolve_eval_dataframe_returns_custom_df_with_reset_index() -> None:
    dataset = DummyDataset()
    custom_df = pd.DataFrame(
        [(0, 0, 1), (1, 0, 1)],
        columns=["head", "relation", "tail"],
        index=[3, 7],
        dtype="int32",
    )

    resolved_df, source_label = resolve_eval_dataframe(dataset, split=None, eval_df=custom_df)

    assert source_label == "custom"
    assert resolved_df.index.tolist() == [0, 1]
    pd.testing.assert_frame_equal(resolved_df, custom_df.reset_index(drop=True))


def test_resolve_eval_dataframe_rejects_invalid_argument_combinations() -> None:
    dataset = DummyDataset()

    with pytest.raises(ValueError, match="Provide only split or eval_df"):
        resolve_eval_dataframe(dataset, split="test", eval_df=dataset.test_df)

    with pytest.raises(ValueError, match="split must be provided"):
        resolve_eval_dataframe(dataset, split=None, eval_df=None)

    with pytest.raises(ValueError, match="Invalid split"):
        resolve_eval_dataframe(dataset, split="dev", eval_df=None)


def test_build_eval_filter_maps_returns_named_maps() -> None:
    dataset = DummyDataset()

    disabled_maps = build_eval_filter_maps(dataset, filtered=False)
    enabled_maps = build_eval_filter_maps(dataset, filtered=True)

    assert isinstance(disabled_maps, EvalFilterMaps)
    assert disabled_maps.tail == {}
    assert disabled_maps.head == {}
    np.testing.assert_array_equal(enabled_maps.tail[(0, 0)], np.array([1, 2, 3], dtype=np.int32))
    np.testing.assert_array_equal(enabled_maps.head[(0, 1)], np.array([0, 1], dtype=np.int32))


def test_evaluate_corruption_side_rejects_non_positive_batch_size() -> None:
    dataset = DummyDataset()

    with pytest.raises(ValueError, match="eval_batch_size must be a positive integer"):
        evaluate_corruption_side(
            DummyModel(),
            dataset.test_df,
            filter_map={},
            corruption_side="tail",
            num_entities=dataset.num_entities,
            eval_batch_size=0,
        )


def test_evaluate_corruption_side_keeps_repeated_queries_row_aligned() -> None:
    eval_df = pd.DataFrame(
        [(0, 0, 1), (0, 0, 2)],
        columns=["head", "relation", "tail"],
        dtype="int32",
    )

    ranks, scores = evaluate_corruption_side(
        DummyModel(),
        eval_df,
        filter_map={},
        corruption_side="tail",
        num_entities=5,
        eval_batch_size=1,
    )

    np.testing.assert_array_equal(ranks, np.array([2, 3], dtype=np.int32))
    np.testing.assert_allclose(scores, np.array([-2.0, -4.0], dtype=np.float32))
