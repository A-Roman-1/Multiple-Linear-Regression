from typing import Any, Dict

from models.base_model import Model

import numpy as np

from sklearn.linear_model import Lasso


class LassoWrapper(Model):
    """
    A wrapper for the Lasso regression model from sklearn.

    The class uses Lasso regression algorithm, allowing it to fit data
    and make predictions. Also stores the model parameters.
    """

    def __init__(self) -> None:
        """
        Initialize the LassoWrapper model.

        Instance of the Lasso model from sklearn, plus a dictionary
        to store model parameters.
        """
        super().__init__()
        self.model: Lasso = Lasso()
        self.parameters: Dict[str, Any] = {}  # Dictionary to store parameters

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the Lasso model to the data.

        .
        """
        self.model.fit(observations, ground_truth)  # Fit the data to the model

        # Store model parameters
        self.parameters["coefficients"] = self.model.coef_
        self.parameters["intercept"] = self.model.intercept_

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict the output for new data using the fitted model.

        .
        """
        return self.model.predict(observations)
