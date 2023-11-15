#exercicio 10

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy

import numpy as np

class StackingClassifier:
    """
    Class representing a decision tree classifier.
    """
    
    def __init__(self, models: list[object], final_model:object):
        self.models = models
        self.final_model = final_model
    """
    Creates a StackingClassifier object.

    Parameters
    ----------
    models: list
        A list of models for the initial set of models.
    final_model: object
        The final model trained with predictions from the initial set of models.
    """
        
    def fit(self, dataset: Dataset) -> 'StackingClassifier':
        """
        Trains the StackingClassifier.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to.
        
        Returns
        -------
        StackingClassifier
            The fitted model.
        """
        predictions = []
        for test_model in self.models:
            test_model.fit(dataset)
            predictions.append(test_model.predict(dataset))
        stacked_y = np.column_stack(predictions) #alinha cada resultado de modelo em 1 coluna (aqui serão 3 colunas pois há 3 modelos para testar no exercicio)
        
        self.final_model.fit(Dataset(dataset.X, stacked_y))
        # print(self.final_model.fit(stacked_y))
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
        base_predictions = [model.predict(dataset) for model in self.models]
        stacked_y = np.column_stack(base_predictions)

        final_predictions = self.final_model.predict(Dataset(dataset.X, stacked_y))
        # print(final_predictions, 'final_predictions')
        
        return final_predictions

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
        return accuracy(dataset.y, predictions)

if __name__ == '__main__':
    from io_.csv_file import read_csv
    from si.model_selection.split import train_test_split
    from si.models.knn_classifier import KNNClassifier
    from si.models.logistic_regression import LogisticRegression
    from si.models.decision_tree_classifier import DecisionTreeClassifier



    data = read_csv('/home/karyna/Documents/SIB/si/datasets/breast_bin/breast-bin.csv', sep=',', features=True, label=True)
    train, test = train_test_split(data, test_size=0.20, random_state=42)
    knn = KNNClassifier(k=3)
    lg = LogisticRegression()
    dtc = DecisionTreeClassifier()

    exercicio=StackingClassifier((knn,lg,dtc),knn)
    exercicio.fit(train)
    # print(exercicio.predict(test))
    print(exercicio.score(test))