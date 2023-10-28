from typing import Tuple

import numpy as np

from data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # set random state
    np.random.seed(random_state)
    # get dataset size
    n_samples = dataset.shape()[0]
    # get number of samples in the test set
    n_test = int(n_samples * test_size)
    # get the dataset permutations
    permutations = np.random.permutation(n_samples)
    # get samples in the test set
    test_idxs = permutations[:n_test]
    # get samples in the training set
    train_idxs = permutations[n_test:]
    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test

#exercise 6.1

def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42):
    np.random.seed(random_state)
    # classes = dataset.get_classes()
    # pandas = dataset.to_dataframe()
    # classes_tmp = pandas[dataset.label] #dataset.label é precisamente 'class' que é a coluna correspondente aos metadados
    # a, b, c = classes_tmp.value_counts()

    unique_classes, class_counts = np.unique(dataset.y, return_counts=True)
    
    train_indices, test_indices = [], []
    for index, label in enumerate(unique_classes):

        num_test_samples = int(class_counts[index] * test_size)
        indices = np.where(dataset.y == label)[0]
        np.random.shuffle(indices)
        
        test_indices.extend(indices[:num_test_samples])
        train_indices.extend(indices[num_test_samples:])

    train_dataset = Dataset(X=dataset.X[train_indices], y=dataset.y[train_indices])
    test_dataset = Dataset(X=dataset.X[test_indices], y=dataset.y[test_indices])
    
    return train_dataset, test_dataset