"""Base class for creating Knowledge Graph grain datasets from pandas DataFrames."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Literal

import grain  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
from grain import DatasetIterator
from numpy.typing import NDArray


class PandasArraySource(grain.sources.RandomAccessDataSource):
    """
    Minimal random-access data source for Grain from a pandas DataFrame.
    """

    def __init__(self, df: pd.DataFrame, columns: Sequence[str] | None = None):
        """
        Minimal random-access data source for Grain from a pandas DataFrame.

        :param df: Input DataFrame
        :type df: pd.DataFrame
        :param columns: Columns to include, defaults to all columns
        :type columns: Optional[Sequence[str]], optional
        :raises KeyError: If any of the specified columns are not found in the DataFrame
        """
        self._df = df.reset_index(drop=True)
        self._cols = list(columns) if columns is not None else list(self._df.columns)
        missing = [c for c in self._cols if c not in self._df.columns]
        if missing:
            raise KeyError(f"columns not found: {missing}")
        self._df = self._df[self._cols]

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, index: int, /) -> NDArray[Any]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return self._df.iloc[index].to_numpy()


class BaseDataset(ABC):
    def __init__(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 0,
        num_threads: int = 16,
        prefetch_buffer_size: int = 500,
    ) -> None:
        """
        Base class for creating Knowledge Graph grain datasets from pandas DataFrames.

        :param batch_size: Batch size, defaults to 32
        :type batch_size: int, optional
        :param shuffle: Whether to shuffle the dataset, defaults to True
        :type shuffle: bool, optional
        :param seed: Random seed for shuffling, defaults to 0
        :type seed: int, optional
        :param num_threads: Number of threads for data loading, defaults to 16
        :type num_threads: int, optional
        :param prefetch_buffer_size: Size of the prefetch buffer, defaults to 500
        :type prefetch_buffer_size: int, optional
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.num_threads = num_threads
        self.prefetch_buffer_size = prefetch_buffer_size

        self.train_df = pd.DataFrame()
        self.val_df = pd.DataFrame()
        self.test_df = pd.DataFrame()

        self.num_entities = 0
        self.num_relations = 0

    @abstractmethod
    def load_data(self) -> None:
        """Populate self.train_df / self.val_df / self.test_df."""
        raise NotImplementedError

    def _make_iter(
        self,
        df: pd.DataFrame,
        *,
        shuffle: bool | None = None,
        batch_size: int | None = None,
    ) -> DatasetIterator:
        """
        Create an iterator that yields batches of triples from the DataFrame.

        :param df: Input DataFrame with columns ["head", "relation", "tail"]
        :type df: pd.DataFrame
        :param shuffle: Whether to shuffle the dataset, defaults to self.shuffle
        :type shuffle: bool | None, optional
        :param batch_size: Batch size, defaults to self.batch_size
        :type batch_size: int | None, optional
        :yield: Batches of triples of shape [B, 3] with dtype int32
        :rtype: Iterator[np.ndarray]
        """
        if shuffle is None:
            shuffle = self.shuffle
        if batch_size is None:
            batch_size = self.batch_size

        source = PandasArraySource(df)  # [N,3]
        ds = grain.MapDataset.source(source)
        if shuffle:
            ds = ds.shuffle(seed=self.seed)  # global shuffle
        ds = ds.map(lambda x: x.astype(np.int32, copy=False))
        ds = ds.batch(batch_size)  # -> yields [B,3] arrays

        iter_dataset = ds.to_iter_dataset(
            grain.ReadOptions(num_threads=self.num_threads, prefetch_buffer_size=self.prefetch_buffer_size)
        )
        return iter(iter_dataset)

    def iter_batches(self, split: Literal["train", "val", "test"] = "train") -> DatasetIterator:
        """
        Create an iterator that yields batches of triples from the specified split.

        :param split: Which split to use, defaults to "train"
        :type split: Literal[], optional
        :return: Iterator over batches of triples
        :rtype: Iterator[np.ndarray]
        """
        df = {"train": self.train_df, "val": self.val_df, "test": self.test_df}[split]
        return self._make_iter(df)

    def iter_eval_batches(
        self,
        split: Literal["train", "val", "test"] = "test",
        *,
        batch_size: int | None = None,
        df: pd.DataFrame | None = None,
    ) -> DatasetIterator:
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
            df = {"train": self.train_df, "val": self.val_df, "test": self.test_df}[split]
        return self._make_iter(df, shuffle=False, batch_size=batch_size)
