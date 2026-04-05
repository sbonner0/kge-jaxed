"""Predict with a saved KGE model."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd  # type: ignore

from kge_jaxed.rngs import RngManager
from kge_jaxed.training import checkpointing as ckpt
from kge_jaxed.training.setup_training import resolve_dataset, resolve_model


class KGEPredict:
    """Helper class for making predictions with a saved KGE model."""

    def __init__(self, checkpoint_path: str | Path) -> None:
        """
        Initialize the prediction helper by loading a saved checkpoint.

        :param checkpoint_path: Path to the saved model checkpoint and metadata.
        :type checkpoint_path: str | Path
        """
        self.checkpoint_dir = Path(checkpoint_path).resolve()
        self.metadata = ckpt._read_metadata(self.checkpoint_dir)
        if self.metadata is None:
            raise ValueError(f"No metadata found in checkpoint directory {self.checkpoint_dir}")

        # Resolve and load the dataset
        # TODO: Maybe we should save the triples with the model and load them here instead of reconstructing the dataset
        dataset_name = self.metadata.get("dataset_name")
        dataset_kwargs = self.metadata.get("dataset_kwargs", {})
        if dataset_name is None:
            raise KeyError("Missing required key 'dataset_name' in metadata")
        self.dataset, self.dataset_name = resolve_dataset(dataset_name, dataset_kwargs)

        # Resolve the model architecture and config
        model_name = self.metadata.get("model_name")
        model_kwargs = self.metadata.get("model_kwargs", {})
        embedding_dim = self.metadata.get("embedding_dim")
        if model_name is None:
            raise KeyError("Missing required key 'model_name' in metadata")
        if embedding_dim is None:
            raise KeyError("Missing required key 'embedding_dim' in metadata")

        self.rng_manager = RngManager(0)
        self.model, self.model_name, self.embedding_dim, self.model_kwargs = resolve_model(
            model_name,
            model_kwargs,
            embedding_dim,
            dataset=self.dataset,
            rng_manager=self.rng_manager,
        )

        self.model, _, _ = ckpt.restore_checkpoint(
            str(self.checkpoint_dir),
            model=self.model,
            restore_optimizer_state=False,
        )

        self._train_triples = set(map(tuple, self.dataset.split_array("train").tolist()))
        self._test_triples = set(map(tuple, self.dataset.split_array("test").tolist()))
        self._val_triples = set(map(tuple, self.dataset.split_array("val").tolist()))

    def _lookup_query_indices(self, query_entity: str, relation: str) -> tuple[int, int]:
        """
        Look up the indices of the query entity and relation in the dataset.

        :param query_entity: The query entity of the triple.
        :type query_entity: str
        :param relation: The relation of the triple.
        :type relation: str
        :return: A tuple containing the indices of the head, relation, and tail.
        :rtype: tuple[int, int]
        """
        entity_idx = self.dataset.entity_to_id.get(query_entity)
        relation_idx = self.dataset.relation_to_id.get(relation)

        if entity_idx is None:
            raise ValueError(f"Query entity '{query_entity}' not found in dataset")
        if relation_idx is None:
            raise ValueError(f"Relation '{relation}' not found in dataset")

        return entity_idx, relation_idx

    def predict(
        self,
        query_entity: str,
        relation: str,
        predict_side: str,
    ) -> pd.DataFrame:
        """
        Make a prediction for a given triple.

        :param query_entity: The query entity of the triple.
        :type query_entity: str
        :param relation: The relation of the triple.
        :type relation: str
        :param predict_side: The side of the triple to predict ("head" or "tail").
        :type predict_side: str
        :return: A DataFrame containing the predicted scores and split membership for each entity.
        :rtype: pd.DataFrame
        """

        if predict_side not in {"head", "tail"}:
            raise ValueError(f"Invalid predict_side '{predict_side}', must be 'head' or 'tail'")

        entity_idx, relation_idx = self._lookup_query_indices(query_entity, relation)
        all_entities = jnp.arange(self.dataset.num_entities, dtype=jnp.int32)

        if predict_side == "tail":
            batch_h = jnp.broadcast_to(entity_idx, (1, self.dataset.num_entities))
            batch_r = jnp.broadcast_to(relation_idx, (1, self.dataset.num_entities))
            batch_t = jnp.broadcast_to(all_entities[None, :], (1, self.dataset.num_entities))
        elif predict_side == "head":
            batch_h = jnp.broadcast_to(all_entities[None, :], (1, self.dataset.num_entities))
            batch_r = jnp.broadcast_to(relation_idx, (1, self.dataset.num_entities))
            batch_t = jnp.broadcast_to(entity_idx, (1, self.dataset.num_entities))

        triples = jnp.stack([batch_h, batch_r, batch_t], axis=2).reshape(-1, 3)
        scores = self.model.score_hrt(triples)

        triple_tuples = list(map(tuple, np.asarray(triples, dtype=np.int32).tolist()))

        data = {
            "entity": [self.dataset.id_to_entity[i] for i in range(self.dataset.num_entities)],
            "score": np.asarray(scores).flatten(),
            "in_train": [triple in self._train_triples for triple in triple_tuples],
            "in_test": [triple in self._test_triples for triple in triple_tuples],
            "in_val": [triple in self._val_triples for triple in triple_tuples],
        }
        return pd.DataFrame(data).sort_values("score", ascending=False).reset_index(drop=True)
