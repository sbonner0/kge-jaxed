"""Contains the PyKEEN dataset class. This allows for the use of PyKEEN's built-in datasets."""

import pandas as pd
from base import BaseTFDataset
from pykeen.datasets import get_dataset


class PyKEENDataset(BaseTFDataset):
    """
    Uses PyKEEN built-in datasets to create knowledge graphs.
    """

    def __init__(self, dataset_name: str, batch_size: int = 32, shuffle: bool = True) -> None:
        """
        Uses PyKEEN built-in datasets to create knowledge graphs.

        :param dataset_name: Name of the PyKEEN dataset.
        :type dataset_name: str
        :param batch_size: Batch size for training, defaults to 32
        :type batch_size: int, optional
        :param shuffle: Whether to shuffle the dataset, defaults to True
        :type shuffle: bool, optional
        """
        super().__init__(batch_size, shuffle)
        self.dataset_name = dataset_name
        self.load_data()

    def load_data(self) -> None:
        """
        Load the dataset from PyKEEN.
        """

        # Load the dataset from PyKEEN
        dataset = get_dataset(dataset=self.dataset_name)

        # Extract the training, validation, and test triples
        self.train_df = pd.DataFrame(dataset.training.mapped_triples.numpy(), columns=["head", "relation", "tail"])
        self.val_df = pd.DataFrame(dataset.validation.mapped_triples.numpy(), columns=["head", "relation", "tail"])
        self.test_df = pd.DataFrame(dataset.testing.mapped_triples.numpy(), columns=["head", "relation", "tail"])


if __name__ == "__main__":

    dataset = PyKEENDataset(dataset_name="nations", batch_size=32, shuffle=True)
    train_dataset = dataset.get_train_dataset()
    val_dataset = dataset.get_val_dataset()
    test_dataset = dataset.get_test_dataset()
    print(train_dataset)
    print(val_dataset)
    print(test_dataset)

    # Iterate through the train dataset
    print("Train Dataset:")
    for triple in train_dataset:
        print("Triple:", triple.numpy())
