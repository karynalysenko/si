#exercicio 9
from typing import Literal
from collections import Counter
import numpy as np
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy

from si.models.decision_tree_classifier import DecisionTreeClassifier

class RandomForestClassifier:
    """
    Class representing a random forest classifier.
    """

    def __init__(self, n_estimators: int=100, max_features: int=2, min_sample_split: int = 2, max_depth: int = 10,
                 mode: Literal['gini', 'entropy'] = 'gini', seed: int=None) -> None:
        """
        Creates a RandomForestClassifier object.

        Parameters
        ----------
        n_estimators: int
            number of decision trees to use
        max_features: int
            maximum number of features to use per tree
        min_sample_split: int
            minimum number of samples required to split an internal node.
        max_depth: int
            maximum depth of the tree.
        mode: Literal['gini', 'entropy']
            the mode to use for calculating the information gain.
        seed: int
            The seed to use for the random number generator
        """
        # parameters
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed

        # estimated parameters
        self.trees = []
    
    def fit(self, dataset: Dataset) -> 'RandomForestClassifier':
        """
        Fits the random forest classifier to a dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to.

        Returns
        -------
        RandomForestClassifier
            The fitted model.
        """
        n_samples, n_features = dataset.shape()

        if self.seed is not None:
            np.random.seed(self.seed)

        if self.max_features is not None:
            self.max_features = int(np.sqrt(n_features))
        
        for i in range(self.n_estimators):
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True) #array,size, replace (array can be selected again)
            bootstrap_features = np.random.choice(n_features, self.max_features, replace=False)

            bootstrap_dataset = Dataset(X=dataset.X[bootstrap_indices][:, bootstrap_features],
                                        y=dataset.y[bootstrap_indices])

            tree = DecisionTreeClassifier()
            tree.fit(bootstrap_dataset)

            self.trees.append((bootstrap_features, tree))
        return self
    
    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Makes predictions for a dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to make predictions for.

        Returns
        -------
        np.ndarray
            The predicted values.
        """
        predictions = []
        most_common = []
        for features_used, tree in self.trees:
            selected_features = [idx for idx in features_used if idx < dataset.X.shape[1]]
            tree_predictions = tree.predict(Dataset(X=dataset.X[:, selected_features], features=dataset.features[selected_features],
                                                    label=dataset.label))
            predictions.append(tree_predictions)

            most_common_class = Counter(tree_predictions).most_common(1)[0][0]
            most_common.append(most_common_class)
        return np.array(predictions)

    def score(self, dataset: Dataset) -> float:
        """
        Calculates the accuracy of the model on a dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to calculate the accuracy on.

        Returns
        -------
        float
            The accuracy of the model on the dataset.
        """
        predictions = self.predict(dataset)
        return accuracy(dataset.y, predictions)


if __name__ == '__main__':
    from io_.csv_file import read_csv
    from si.model_selection.split import train_test_split

    data = read_csv('/home/karyna/Documents/SIB/si/datasets/iris/iris.csv', sep=',', features=True, label=True)
    train, test = train_test_split(data, test_size=0.33, random_state=42)
    model = RandomForestClassifier(min_sample_split=3, max_depth=3, mode='gini', seed = 10)
    model.fit(train)
    # for i in model.fit(train).trees:
    #     print(i[1].print_tree()) 
    # print(model.predict(test))
    print(model.score(test))

    from sklearn.ensemble import RandomForestClassifier as tmp
    rf_model = tmp(n_estimators=100, max_features=2, min_samples_split= 2, max_depth=10, criterion='gini')
    rf_model.fit(train.X, train.y)
    print(rf_model.score(test.X,test.y))

    #dá muito diferente