"""Base class for creating Knowledge Graph datasets from pandas DataFrames."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd  # type: ignore
from numpy.typing import NDArray


class BaseDataset(ABC):
    def __init__(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        """
        Base class for creating Knowledge Graph datasets from pandas DataFrames.

        :param batch_size: Batch size, defaults to 32
        :type batch_size: int, optional
        :param shuffle: Whether to shuffle the dataset, defaults to True
        :type shuffle: bool, optional
        :param seed: Random seed for shuffling, defaults to 0
        :type seed: int, optional
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        self.train_df = pd.DataFrame()
        self.val_df = pd.DataFrame()
        self.test_df = pd.DataFrame()
        self._split_arrays: dict[str, NDArray[np.int32]] = {}

        self.dataset_name: str | None = None
        self.num_entities = 0
        self.num_relations = 0
        self.entity_to_id: dict[str, int] = {}
        self.relation_to_id: dict[str, int] = {}
        self.id_to_entity: dict[int, str] = {}
        self.id_to_relation: dict[int, str] = {}

    @abstractmethod
    def load_data(self) -> None:
        """Populate self.train_df / self.val_df / self.test_df."""
        raise NotImplementedError

    def split_array(
        self,
        split: Literal["train", "val", "test"] = "train",
        *,
        columns: Sequence[str] = ("head", "relation", "tail"),
    ) -> NDArray[np.int32]:
        """
        Return a cached contiguous NumPy array for the requested split.

        :param split: Which split to materialize, defaults to "train"
        :type split: Literal["train", "val", "test"], optional
        :param columns: Column order to materialize, defaults to ("head", "relation", "tail")
        :type columns: Sequence[str], optional
        :return: Array of triples with shape [N, 3] and dtype int32
        :rtype: NDArray[np.int32]
        """
        cached = self._split_arrays.get(split)
        if cached is not None:
            return cached

        df = {"train": self.train_df, "val": self.val_df, "test": self.test_df}[split]
        array = np.ascontiguousarray(df[list(columns)].to_numpy(dtype=np.int32, copy=True))
        self._split_arrays[split] = array
        return array

    def iter_batches(
        self,
        split: Literal["train", "val", "test"] = "train",
        *,
        batch_size: int | None = None,
        shuffle: bool | None = None,
        seed: int | None = None,
    ):
        """
        Yield batches from a cached NumPy array.

        :param split: Which split to use, defaults to "train"
        :type split: Literal["train", "val", "test"], optional
        :param batch_size: Batch size, defaults to self.batch_size
        :type batch_size: int | None, optional
        :param shuffle: Whether to shuffle rows before batching, defaults to self.shuffle
        :type shuffle: bool | None, optional
        :param seed: Seed used for shuffling when enabled
        :type seed: int | None, optional
        :yield: Batches of triples with shape [B, 3] and dtype int32
        :rtype: Iterator[NDArray[np.int32]]
        """
        if batch_size is None:
            batch_size = self.batch_size
        if shuffle is None:
            shuffle = self.shuffle

        array = self.split_array(split)
        if not len(array):
            return

        if shuffle:
            rng = np.random.default_rng(self.seed if seed is None else seed)
            indices = rng.permutation(len(array))
            for start in range(0, len(indices), batch_size):
                yield array[indices[start : start + batch_size]]
            return

        for start in range(0, len(array), batch_size):
            yield array[start : start + batch_size]

    def iter_eval_batches(
        self,
        split: Literal["train", "val", "test"] = "test",
        *,
        batch_size: int | None = None,
        df: pd.DataFrame | None = None,
    ):
        """
        Create an iterator for evaluation (no shuffling).

        :param split: Which split to use, defaults to "test"
        :type split: Literal[], optional
        :param batch_size: Batch size, defaults to self.batch_size
        :type batch_size: int | None, optional
        :param df: Optional DataFrame override for evaluation
        :type df: pd.DataFrame | None, optional
        :return: Iterator over batches of triples
        :rtype: Iterator[np.ndarray]
        """
        if df is None:
            yield from self.iter_batches(split, batch_size=batch_size, shuffle=False)
            return

        if batch_size is None:
            batch_size = self.batch_size
        array = np.ascontiguousarray(df[["head", "relation", "tail"]].to_numpy(dtype=np.int32, copy=True))
        for start in range(0, len(array), batch_size):
            yield array[start : start + batch_size]
