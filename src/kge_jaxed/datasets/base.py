"""Base class for creating Knowledge Graph TensorFlow-based datasets from pandas DataFrames."""

from abc import ABC, abstractmethod

import pandas as pd  # type: ignore
import tensorflow as tf  # type: ignore


class BaseTFDataset(ABC):
    def __init__(self, batch_size: int = 32, shuffle: bool = True) -> None:
        """
        Initialize the dataset.

        :param batch_size: Batch size for training.
        :type batch_size: int
        :param shuffle: Whether to shuffle the dataset.
        :type shuffle: bool
        """

        self.batch_size = batch_size
        self.shuffle = shuffle

        # Initialize the DataFrames
        self.train_df = pd.DataFrame()
        self.val_df = pd.DataFrame()
        self.test_df = pd.DataFrame()

    @abstractmethod
    def load_data(self):
        """
        Load the dataset from file.
        """
        pass

    def _dataframe_to_tf_dataset(self, df: pd.DataFrame):
        """
        Convert a pandas DataFrame to a TensorFlow Dataset.
        """

        dict_data = {key: df[key].values for key in df.columns}
        dataset = tf.data.Dataset.from_tensor_slices(dict_data)
        return dataset

    def _preprocess(self, example):
        """
        Preprocess the data by extracting the head, relation, and tail columns.
        """

        return tf.stack([example[col] for col in ["head", "relation", "tail"]], axis=-1)

    def _create_pipeline(self, df: pd.DataFrame):
        """
        Create a TensorFlow data pipeline for the given DataFrame.
        """
        dataset = self._dataframe_to_tf_dataset(df)
        dataset = dataset.map(self._preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        if self.shuffle:
            dataset = dataset.shuffle(len(df))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def get_train_dataset(self):
        return self._create_pipeline(self.train_df)

    def get_val_dataset(self):
        return self._create_pipeline(self.val_df)

    def get_test_dataset(self):
        return self._create_pipeline(self.test_df)
