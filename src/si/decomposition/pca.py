import numpy as np

from data.dataset import Dataset
from statistics.f_classification import f_classification
# from io_.csv_file import read_csv

#exercise 5.1
class PCA:

    def __init__(self, n_components : int):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, dataset: Dataset):
        self.mean= np.mean(dataset.X)
        centered_data = dataset.X - self.mean

        u, s, v = np.linalg.svd(centered_data, full_matrices=False)
        self.components = v[:self.n_components]

        n_samples = dataset.X.shape[0]
        explained_variance = (s ** 2) / (n_samples - 1)

        self.explained_variance = explained_variance[:self.n_components]
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        centered_data = dataset.X - self.mean
        V = self.components.T #.T corresponde à transposta em numpy
        X_reduced = np.dot(centered_data, V)

        return X_reduced

    def fit_transform(self, dataset: Dataset) -> Dataset:
        self.fit(dataset)
        return self.transform(dataset)

#exercise 5.2

if __name__ == '__main__':

    dataset = Dataset(X=np.array([[3, 2, 0, 3],
                                  [4, 1, 4, 3],
                                  [2, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")
    selector = PCA(n_components=2)
    # selector = selector.fit(dataset)
    # print("PCA components:\n", selector.components)
    # print("Explained variance:\n", selector.explained_variance)
    sector = selector.fit_transform(dataset)
    print("\n",sector)