import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


class BaseDataset:
    def __init__(self, data, feature_columns, label_column, test_size=0.2, val_size=0.1, batch_size=32, shuffle=True):
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
        self.data = data
        self.feature_columns = feature_columns
        self.label_column = label_column
        self.test_size = test_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Split the data
        self.train_data, self.test_data = train_test_split(self.data, test_size=self.test_size, random_state=42)
        self.train_data, self.val_data = train_test_split(self.train_data, test_size=self.val_size, random_state=42)

    def _dataframe_to_tf_dataset(self, df):
        """Convert a pandas DataFrame to a TensorFlow Dataset."""

        dict_data = {key: df[key].values for key in df.columns}
        dataset = tf.data.Dataset.from_tensor_slices(dict_data)
        return dataset

    def _preprocess(self, example):
        """Preprocess the data by extracting features and labels."""
        features = tf.stack([example[col] for col in self.feature_columns], axis=-1)
        label = example[self.label_column]
        return features, label

    def _create_pipeline(self, df):
        """Create a TensorFlow data pipeline for the given DataFrame."""
        dataset = self._dataframe_to_tf_dataset(df)
        dataset = dataset.map(self._preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        if self.shuffle:
            dataset = dataset.shuffle(len(df))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def get_train_dataset(self):
        """Return the train dataset pipeline."""
        return self._create_pipeline(self.train_data)

    def get_val_dataset(self):
        """Return the validation dataset pipeline."""
        return self._create_pipeline(self.val_data)

    def get_test_dataset(self):
        """Return the test dataset pipeline."""
        return self._create_pipeline(self.test_data)


# Example Usage
# Sample DataFrame
data = {
    "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    "feature2": [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
    "label": [0, 1, 0, 1, 0, 1, 0, 1],
}
df = pd.DataFrame(data)

# Instantiate the dataset class
dataset = BaseDataset(data=df, feature_columns=["feature1", "feature2"], label_column="label", batch_size=2)

# Get train, validation, and test datasets
train_ds = dataset.get_train_dataset()
val_ds = dataset.get_val_dataset()
test_ds = dataset.get_test_dataset()

# Iterate through the train dataset
print("Train Dataset:")
for batch_features, batch_labels in train_ds:
    print("Batch Features:", batch_features.numpy())
    print("Batch Labels:", batch_labels.numpy())
