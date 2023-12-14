import numpy as np

from data.dataset import Dataset
from metrics.mse import mse


class RidgeRegressionLeastSquares:

    def __init__(self, l2_penalty: float = 1, scale: bool = True):
        """

        Parameters
        ----------
        l2_penalty: float
            The L2 regularization parameter
        scale: bool
            Whether to scale the dataset or not
        """
        # parameters
        self.l2_penalty = l2_penalty
        self.scale = scale

        # attributes
        self.theta = None
        self.theta_zero = None #y intercept
        self.mean = None
        self.std = None

    def fit(self, dataset: Dataset) -> 'RidgeRegressionLeastSquares':
        """
        Fit the model to the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: RidgeRegressionLeastSquares
            The fitted model
        """
        if self.scale:
            # compute mean and std
            self.mean = np.nanmean(dataset.X, axis=0)
            self.std = np.nanstd(dataset.X, axis=0)
            # scale the dataset
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        m_samples, n_features = dataset.shape()
        X = np.c_[np.ones(m_samples), X]
        penalty_matrix = self.l2_penalty * np.eye(n_features + 1)
        penalty_matrix[0,0]=0 #garantir
        print(penalty_matrix)
        transposed_X = X.T
        A = np.linalg.inv(transposed_X.dot(X) + penalty_matrix)
        b = transposed_X.dot(y)
        thetas = A.dot(b)
        self.theta_zero=thetas[0] #indicaçõe do slide no passo 5 
        self.theta=thetas[1:]
        return self

    def predict(self, dataset: Dataset) -> np.array:
        """
        Predict the output of the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the output of

        Returns
        -------
        predictions: np.array
            The predictions of the dataset
        """
        X_scaled = (X - self.mean) / self.std if self.scale else dataset.X

        # Add an intercept term (column of ones) to X
        m_samples, n_features = dataset.shape()

        X_scaled = np.c_[np.ones(m_samples), X_scaled]

        # Compute the predicted Y (X * thetas)
        predicted_Y = X_scaled.dot(np.r_[self.theta_zero, self.theta])

        # Step 4: Return the predicted Y
        return predicted_Y

    def score(self, dataset: Dataset) -> float:
        """
        Compute the Mean Square Error of the model on the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the MSE on

        Returns
        -------
        mse: float
            The Mean Square Error of the model
        """
        y_pred = self.predict(dataset)
        return mse(dataset.y, y_pred)


# This is how you can test it against sklearn to check if everything is fine
if __name__ == '__main__':
    # make a linear dataset
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    dataset_ = Dataset(X=X, y=y)

    # fit the model
    model = RidgeRegressionLeastSquares(l2_penalty=0.1)
    model.fit(dataset_)
    # print(model.theta, 'theta')
    # print(model.theta_zero, 'theta0')

    # compute the score
    print(model.score(dataset_), 'meu score')

    # compare with sklearn
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=0.1)
    # scale data
    X = (dataset_.X - np.nanmean(dataset_.X, axis=0)) / np.nanstd(dataset_.X, axis=0)
    model.fit(X, dataset_.y)
    # print(model.coef_, 'theta') # should be the same as theta
    # print(model.intercept_, 'theta0') # should be the same as theta_zero
    print(mse(dataset_.y, model.predict(X)), 'score')
