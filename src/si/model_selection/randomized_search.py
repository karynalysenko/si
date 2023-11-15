#exercicio 11
import numpy as np

from si.data.dataset import Dataset
from si.model_selection.cross_validation import k_fold_cross_validation
from typing import Callable, Tuple, Dict, Any


def randomized_search_cv(model, dataset: Dataset, hyperparameter_grid: Dict[str, Tuple], scoring: Callable = None, cv: int = 5, n_iter: int = None) -> Dict[str, Any]:
    
    for parameter in hyperparameter_grid:
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {model} does not have parameter {parameter}.")

    results = {'scores': [], 'hyperparameters': []}

    best_score = float('-inf')
    best_hyperparameters = None
    all_scores = []

    hyperparameter_combinations = {
        hyperparameter: np.random.choice(values, n_iter) #random.choice(a, size)
        for hyperparameter, values in hyperparameter_grid.items()} # dicionario por compreensao: vai ter todas as combinações possiveis de cada hiperparametro, gerados aleatóriamente
    
    for i in range(n_iter):
        current_hyperparameters = {
            hyperparameter: hyperparameter_combinations[hyperparameter][i]
            for hyperparameter in hyperparameter_grid.keys()  # para cada indice, pega nos parametros correspondentes e faz o setting no modelo em baixo
        }
        for hyperparameter, value in current_hyperparameters.items():
            setattr(model, hyperparameter, value)
    
        scores = k_fold_cross_validation(model, dataset, scoring, cv)
        mean_score = np.mean(scores)
        all_scores.append({'hyperparameters': current_hyperparameters, 'score': mean_score})

        if mean_score > best_score:
            best_score = mean_score
            best_hyperparameters = current_hyperparameters

    return {
        'all hyperparameter_combinations': hyperparameter_combinations,
        'all_scores': all_scores,
        'best_score': best_score,
        'best_hyperparameters': best_hyperparameters
    }

if __name__ == '__main__':
    from io_.csv_file import read_csv
    from src.si.models.logistic_regression import LogisticRegression

    filename= '/home/karyna/Documents/SIB/si/datasets/breast_bin/breast-bin.csv'
    dataset = read_csv(filename, features = True, label= True)
    knn = LogisticRegression()
    
    parameters = {
        'l2_penalty': np.linspace(1, 10, 10),
        'alpha': np.linspace(0.001, 0.0001, 100),
        'max_iter': np.linspace(1000, 2000, 200)
    }
    results = randomized_search_cv(knn, dataset, hyperparameter_grid = parameters, cv=3, n_iter=8)
    print(results)
    # fit the model
    # model = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)
    # model.fit(dataset_train)
