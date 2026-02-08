import jax.numpy as jnp
import jax.tree_util as tree_util
import numpy as np
import pandas as pd

from kge_jaxed.datasets.base import BaseDataset
from kge_jaxed.pipeline import KGEPipeline

# These tests compare grouped evaluation ranks against a brute-force reference
# to ensure correctness for both filtered and unfiltered settings.


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


def _reference_ranks(
    model: DummyModel,
    eval_triples: np.ndarray,
    num_entities: int,
    all_triples: np.ndarray,
    *,
    filtered: bool,
) -> tuple[list[int], list[int]]:
    head_ranks: list[int] = []
    tail_ranks: list[int] = []

    for h, r, t in eval_triples:
        # Tail corruption
        tails = np.arange(num_entities, dtype=np.int32)
        tail_batch = np.stack([np.full(num_entities, h), np.full(num_entities, r), tails], axis=1).astype(np.int32)
        scores = np.asarray(model.score_hrt(jnp.asarray(tail_batch)))
        true_score = scores[t]
        rank = 1 + int(np.sum(scores > true_score))
        if filtered:
            mask = (all_triples[:, 0] == h) & (all_triples[:, 1] == r) & (all_triples[:, 2] != t)
            filtered_indices = all_triples[mask, 2]
            if filtered_indices.size:
                rank -= int(np.sum(scores[filtered_indices] > true_score))
        tail_ranks.append(rank)

        # Head corruption
        heads = np.arange(num_entities, dtype=np.int32)
        head_batch = np.stack([heads, np.full(num_entities, r), np.full(num_entities, t)], axis=1).astype(np.int32)
        scores = np.asarray(model.score_hrt(jnp.asarray(head_batch)))
        true_score = scores[h]
        rank = 1 + int(np.sum(scores > true_score))
        if filtered:
            mask = (all_triples[:, 1] == r) & (all_triples[:, 2] == t) & (all_triples[:, 0] != h)
            filtered_indices = all_triples[mask, 0]
            if filtered_indices.size:
                rank -= int(np.sum(scores[filtered_indices] > true_score))
        head_ranks.append(rank)

    return head_ranks, tail_ranks


def _make_pipeline() -> KGEPipeline:
    dataset = DummyDataset()
    pipeline = KGEPipeline(
        model="transe",
        loss_name="mrl",
        dataset=dataset,
        embedding_dim=4,
        seed=0,
    )
    pipeline.model = DummyModel()
    return pipeline


def test_grouped_eval_unfiltered_matches_reference():
    pipeline = _make_pipeline()
    eval_df = pipeline.dataset.test_df
    all_triples = pd.concat(
        [pipeline.dataset.train_df, pipeline.dataset.val_df, pipeline.dataset.test_df],
        ignore_index=True,
    ).to_numpy(dtype=np.int32)

    ref_head, ref_tail = _reference_ranks(
        pipeline.model, eval_df.to_numpy(dtype=np.int32), pipeline.dataset.num_entities, all_triples, filtered=False
    )

    _, ranks_df = pipeline.evaluate(
        split="test",
        filtered=False,
        eval_batch_size=2,
    )

    assert ranks_df["rank_head"].tolist() == ref_head
    assert ranks_df["rank_tail"].tolist() == ref_tail


def test_grouped_eval_filtered_matches_reference():
    pipeline = _make_pipeline()
    eval_df = pipeline.dataset.test_df
    all_triples = pd.concat(
        [pipeline.dataset.train_df, pipeline.dataset.val_df, pipeline.dataset.test_df],
        ignore_index=True,
    ).to_numpy(dtype=np.int32)

    ref_head, ref_tail = _reference_ranks(
        pipeline.model, eval_df.to_numpy(dtype=np.int32), pipeline.dataset.num_entities, all_triples, filtered=True
    )

    _, ranks_df = pipeline.evaluate(
        split="test",
        filtered=True,
        eval_batch_size=2,
    )

    assert ranks_df["rank_head"].tolist() == ref_head
    assert ranks_df["rank_tail"].tolist() == ref_tail
