"""Base class for creating Knowledge Graph TensorFlow-based datasets from pandas DataFrames."""

from abc import ABC, abstractmethod

import pandas as pd
import tensorflow as tf


class BaseTFDataset(ABC):
    def __init__(self, batch_size: int = 32, shuffle: bool = True) -> None:
        """
        Initialize the dataset.

        Args:
            data (pd.DataFrame): The dataset in pandas DataFrame format.
            feature_columns (list): List of column names to be used as features.
            label_column (str): Column name to be used as the label.
            test_size (float): Proportion of the dataset to include in the test split.
            val_size (float): Proportion of the remaining train data to include in the validation split.
            batch_size (int): Batch size for training.
            shuffle (bool): Whether to shuffle the dataset.
        """

        self.batch_size = batch_size
        self.shuffle = shuffle

    @abstractmethod
    def load_data(self):
        """Load the dataset from file."""
        pass

    def _dataframe_to_tf_dataset(self, df: pd.DataFrame):
        """
        Convert a pandas DataFrame to a TensorFlow Dataset.
        """

        dict_data = {key: df[key].values for key in df.columns}
        dataset = tf.data.Dataset.from_tensor_slices(dict_data)
        return dataset

    def _preprocess(self, example):
        """Preprocess the data by extracting the head, relation, and tail columns."""

        return tf.stack([example[col] for col in ["head", "relation", "tail"]], axis=-1)

    def _create_pipeline(self, df):
        """Create a TensorFlow data pipeline for the given DataFrame."""
        dataset = self._dataframe_to_tf_dataset(df)
        print(dataset)
        dataset = dataset.map(self._preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        if self.shuffle:
            dataset = dataset.shuffle(len(df))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def get_train_dataset(self):
        """Return the train dataset pipeline."""
        return self._create_pipeline(self.train_df)

    def get_val_dataset(self):
        """Return the validation dataset pipeline."""
        return self._create_pipeline(self.val_df)

    def get_test_dataset(self):
        """Return the test dataset pipeline."""
        return self._create_pipeline(self.test_df)
