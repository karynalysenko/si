import numpy as np
import math

#exercise 7.1
def rmse(y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    summ = np.sum((y_true - Y_pred) ** 2) / len(y_true)
    return math.sqrt(summ)


