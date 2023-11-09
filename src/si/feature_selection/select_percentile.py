import numpy as np
import os
import sys

from data.dataset import Dataset
from statistics.f_classification import f_classification
# from io_.csv_file import read_csv

class SelectPercentile:

    def __init__(self, percentile: int, score_func=f_classification):
        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None
        

    def fit(self, dataset: Dataset):
        self.F, self.p = self.score_func(dataset)
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        X = dataset.X
        y = dataset.y

        num_selected_features = int((self.percentile*len(dataset.features))/100)
        sorted_indices = np.argsort(self.F)[::-1]
        selected_indices = sorted_indices[:num_selected_features]
        X_transformed = X[:, selected_indices]
        transformed_dataset = Dataset(X=X_transformed, y=y, features=[dataset.features[i] for i in selected_indices], label=dataset.label)
        
        return transformed_dataset

    def fit_transform(self, dataset: Dataset) -> Dataset:
        self.fit(dataset)
        return self.transform(dataset)
    
if __name__ == '__main__':

    dataset = Dataset(X=np.array([[3, 2, 0, 3],
                                  [4, 1, 4, 3],
                                  [2, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")
    selector = SelectPercentile(percentile=50)
    selector = selector.fit(dataset)
    print(selector.p)

    dataset = selector.transform(dataset)
    print(dataset.features)