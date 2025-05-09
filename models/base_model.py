from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class Model(ABC):
    """
    Abstract base class for the models.

    Template for machine learning models, which shows that every
    subclass need to implement the `fit` and `predict` methods.
    Also, has defined the dictionary for parameters.
    """

    def __init__(self) -> None:
        """
        Initialize the Model class.

        Initializes an empty dictionary for weights.
        """
        self.parameters: Dict[str, Any] = {}

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the model to the provided data.

        Uses the input features and target values.
        """
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict output using the model.

        Also, it returns a list.
        """
        pass
