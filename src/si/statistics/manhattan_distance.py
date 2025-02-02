import numpy as np


def manhattan_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray: #calcula distancia (tipo escadinha)
    return np.abs(x - y).sum(axis=1)


if __name__ == '__main__':
    # test euclidean_distance
    x = np.array([1, 2, 3])
    y = np.array([[1, 2, 3], [4, 5, 6]])
    our_distance = manhattan_distance(x, y)


    from sklearn.metrics.pairwise import manhattan_distances
    sklearn_distance = manhattan_distances(x.reshape(1, -1), y)
    assert np.allclose(our_distance, sklearn_distance)
    print(our_distance, sklearn_distance)