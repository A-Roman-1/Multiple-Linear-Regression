from models.base_model import Model

import numpy as np


class MultipleLinearRegression(Model):
    """
    Multiple Linear Regression model for predicting
    a continuous target variable.

    using multiple features (predictors).
    """

    def __init__(self):
        """
        Initialize the MultipleLinearRegression model.

        has weights that is encapsulated and initialised with none
        """
        super().__init__()
        self._weights = None

    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        """
        Fit the model to the training data.

        Uses the normal equation to compute the optimal weights (parameters).
        """
        # Add a column of ones to the features array for bias term (intercept)
        augmented_features = np.c_[np.ones((features.shape[0], 1)), features]

        # Compute the weights using the equation: theta = (X^T X)^-1 X^T y
        features_transpose = augmented_features.T
        weights = (
            np.linalg.inv(features_transpose.dot(augmented_features))
            .dot(features_transpose)
            .dot(target)
        )
        self.weights = weights

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict target values for given input features.

        .
        """
        # Add a column of ones to input features for bias term (intercept)
        augmented_features = np.c_[np.ones((features.shape[0], 1)), features]

        # Compute the predictions: prediction = X * weights
        return augmented_features.dot(self.weights)

    @property
    def weights(self) -> np.ndarray:
        """
        Get the parameters of the model.

        The model's weights, including intercept as the first element.
        """
        return self._weights

    @weights.setter
    def weights(self, weights: np.ndarray) -> None:
        """
        Set the parameters of the model.

        The model's weights, including intercept as the first element.
        """
        self._weights = weights

    @property
    def coefficients(self) -> np.ndarray:
        """
        Get the model's coef without the intercept.

        The coefficients for the input features is returned.
        """
        return self.weights[1:]

    @property
    def intercept(self) -> float:
        """
        Get the bias of the model.

        The intercept term (bias) is returned.
        """
        return self.weights[0]
