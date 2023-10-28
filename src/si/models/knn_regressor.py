from typing import Callable, Union

import numpy as np

from data.dataset import Dataset
from statistics.euclidean_distance import euclidean_distance
from metrics.rmse import rmse

class KNNRegressor:

    def __init__(self, k: int = 1, distance: Callable = euclidean_distance):
        """
        Initialize the KNN regressor

        Parameters
        ----------
        k: int
            The number of nearest neighbors to use
        distance: Callable
            The distance function to use
        """
        # parameters
        self.k = k
        self.distance = distance

        # attributes
        self.dataset = None

    def fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        It fits the model to the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: KNNRegressor
            The fitted model
        """
        self.dataset = dataset
        return self

    def _get_closest_value(self, sample: np.ndarray) -> Union[int, str]:
        """
        It returns the closest label of the given sample

        Parameters
        ----------
        sample: np.ndarray
            The sample to get the closest label of

        Returns
        -------
        label: str or int
            The closest label
        """
        # compute the distance between the sample and the dataset
        distances = self.distance(sample, self.dataset.X)

        # get the k nearest neighbors
        k_nearest_indices = np.argsort(distances)[:self.k] #busca os indices
        k_nearest_values = [self.dataset.y[i] for i in k_nearest_indices]

        print(k_nearest_values,'aqui')
        return np.mean(k_nearest_values)


    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        It predicts the classes of the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the classes of

        Returns
        -------
        predictions: np.ndarray
            The predictions of the model
        """
        return np.apply_along_axis(self._get_closest_value, axis=1, arr=dataset.X)

    def score(self, dataset: Dataset) -> float:
        """
        It returns the accuracy of the model on the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate the model on

        Returns
        -------
        accuracy: float
            The accuracy of the model
        """
        predictions = self.predict(dataset)
        return rmse(dataset.y, predictions) #rmse(y_true, Y_pred)  y_true=real label values; Y_pred = predicted labels values


if __name__ == '__main__':
    #exercise 7.3
    from data.dataset import Dataset
    from model_selection.split import train_test_split
    from src.io.csv_file import read_csv

    filename= '/home/karyna/Documents/SIB/si/datasets/cpu/cpu.csv'


    dataset_ = read_csv(filename, features = True, label= True)

    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # initialize the KNN Regressor
    knn = KNNRegressor(k=3) 

    # fit the model to the train dataset
    knn.fit(dataset_train)

    # # evaluate the model on the test dataset
    score = knn.score(dataset_test)
    print(f'The accuracy of the model is: {score}') # model accuracy aumenta com o aumento de k
