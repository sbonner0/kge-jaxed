"""Base class for creating Knowledge Graph grain datasets from pandas DataFrames."""

from abc import ABC, abstractmethod
from typing import Iterator, Literal

import grain  # type: ignore
import numpy as np
import pandas as pd  # type: ignore


class BaseDataset(ABC):
    def __init__(self, batch_size: int = 32, shuffle: bool = True, seed: int = 0) -> None:
        """
        Base class for creating Knowledge Graph grain datasets from pandas DataFrames.

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

        # same fields as your TF class
        self.train_df = pd.DataFrame()
        self.val_df = pd.DataFrame()
        self.test_df = pd.DataFrame()

    @abstractmethod
    def load_data(self) -> None:
        """Populate self.train_df / self.val_df / self.test_df."""
        raise NotImplementedError

    def _df_to_records(self, df: pd.DataFrame) -> np.ndarray:
        """
        Convert a DataFrame to a NumPy array of triples.

        :param df: Input DataFrame with columns ["head", "relation", "tail"]
        :type df: pd.DataFrame
        :return: NumPy array of shape [N, 3] with dtype int32
        :rtype: np.ndarray
        """
        # returns a NumPy array shape [N, 3] (head, relation, tail), int32
        arr = df[["head", "relation", "tail"]].to_numpy()
        if arr.dtype != np.int32:
            arr = arr.astype(np.int32, copy=False)
        return arr

    def _make_iter(self, df: pd.DataFrame) -> Iterator[np.ndarray]:
        """
        Create an iterator that yields batches of triples from the DataFrame.

        :param df: Input DataFrame with columns ["head", "relation", "tail"]
        :type df: pd.DataFrame
        :yield: Batches of triples of shape [B, 3] with dtype int32
        :rtype: Iterator[np.ndarray]
        """
        records = self._df_to_records(df)  # [N,3]
        ds = grain.MapDataset.source(records)
        if self.shuffle:
            ds = ds.shuffle(seed=self.seed)  # global shuffle
        ds = ds.map(lambda x: x.astype(np.int32, copy=False))
        ds = ds.batch(self.batch_size)  # -> yields [B,3] arrays

        # Optional: tune prefetch/threads via ReadOptions later if needed
        for batch in ds:
            # batch is already a NumPy array [B,3]
            yield batch

    def iter_batches(self, split: Literal["train", "val", "test"] = "train") -> Iterator[np.ndarray]:
        """
        Create an iterator that yields batches of triples from the specified split.

        :param split: Which split to use, defaults to "train"
        :type split: Literal[], optional
        :return: Iterator over batches of triples
        :rtype: Iterator[np.ndarray]
        """
        df = {"train": self.train_df, "val": self.val_df, "test": self.test_df}[split]
        return self._make_iter(df)
