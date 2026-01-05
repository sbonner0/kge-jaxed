"""Contains the PyKEEN dataset class. This allows for the use of PyKEEN's built-in datasets."""

import pandas as pd  # type: ignore
from pykeen.datasets import get_dataset

from kge_jaxed.datasets.base import BaseDataset  # type: ignore


class PyKEENDataset(BaseDataset):
    """
    Uses PyKEEN built-in datasets to create knowledge graphs. See https://pykeen.readthedocs.io/en/stable/api/pykeen.datasets.Dataset.html
    for all of the available datasets.
    """

    def __init__(
        self,
        dataset_name: str,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        """
        Uses PyKEEN built-in datasets to create knowledge graphs.

        :param dataset_name: Name of the PyKEEN dataset.
        :type dataset_name: str
        :param batch_size: Batch size for training, defaults to 32
        :type batch_size: int, optional
        :param shuffle: Whether to shuffle the dataset, defaults to True
        :type shuffle: bool, optional
        :param seed: Random seed for shuffling, defaults to 0
        :type seed: int, optional
        """
        super().__init__(batch_size=batch_size, shuffle=shuffle, seed=seed)
        self.dataset_name = dataset_name
        self.load_data()

    def load_data(self) -> None:
        """
        Load the dataset from PyKEEN.
        """

        # Load the dataset from PyKEEN
        pykeen_ds = get_dataset(dataset=self.dataset_name)

        # Extract the training, validation, and test triples
        self.train_df = pd.DataFrame(
            pykeen_ds.training.mapped_triples.numpy(), columns=["head", "relation", "tail"], dtype="int32"
        )
        self.val_df = pd.DataFrame(
            pykeen_ds.validation.mapped_triples.numpy(), columns=["head", "relation", "tail"], dtype="int32"
        )
        self.test_df = pd.DataFrame(
            pykeen_ds.testing.mapped_triples.numpy(), columns=["head", "relation", "tail"], dtype="int32"
        )

        # Set the number of entities and relations
        self.num_entities = pykeen_ds.num_entities
        self.num_relations = pykeen_ds.num_relations


if __name__ == "__main__":
    dataset = PyKEENDataset(dataset_name="nations", batch_size=32, shuffle=True)

    train_dataset = dataset.iter_batches("train")
    val_dataset = dataset.iter_batches("val")
    test_dataset = dataset.iter_batches("test")

    print(train_dataset)
    print(val_dataset)
    print(test_dataset)

    # Iterate through the train dataset
    print("Train Dataset:")
    for pos_batch in train_dataset:
        print("pos_batch:", pos_batch)
        print("pos_batch.shape:", pos_batch.shape)
        print("pos_batch.dtype:", pos_batch.dtype)
        break
