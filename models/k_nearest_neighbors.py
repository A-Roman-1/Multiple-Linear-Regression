from collections import Counter

from models.base_model import Model

import numpy as np


class KNearestNeighbors(Model):
    """
    K-Nearest Neighbors classifier.

    This class inherits from base_model
    and is used to predict the label of a data point
    based on the majority vote of its nearest neighbors.
    """

    def __init__(self, k=3):
        """
        Initialize the KNN classifier.

        Inherits from base_model and sets the number of neighbors to consider.
        Also initializes observations and ground truth as private attributes.
        """
        super().__init__()
        self.k = k
        self._observations = None
        self._ground_truth = None

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the model using training data.

        Stores the training data (observations) and their corresponding labels.
        """
        self._observations = observations
        self._ground_truth = ground_truth
        self.parameters = {
            "observations": observations,
            "ground_truth": ground_truth
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict the labels for a set of data points.

        Uses the k-nearest neighbors to predict the
        label for each input observation.
        """
        return np.array(
            [self._predict_single(observation) for observation in observations]
        )

    def _predict_single(self, observation: np.ndarray) -> int:
        """
        Predict the label for a single data point.

        Finds the k-nearest neighbors and returns the most common label.
        """
        distances = self._compute_distances(observation)
        nearest_neighbor_indices = self._get_nearest_neighbors(distances)
        nearest_neighbor_labels = self._ground_truth[nearest_neighbor_indices]

        return self._get_most_common_label(nearest_neighbor_labels)

    def _compute_distances(self, observation: np.ndarray) -> np.ndarray:
        # Compute the Euclidean distances between all training points.
        return np.linalg.norm(self._observations - observation, axis=1)

    def _get_nearest_neighbors(self, distances: np.ndarray) -> np.ndarray:
        # Get the indices of the k-nearest neighbors based on the distances.
        return np.argsort(distances)[: self.k]

    def _get_most_common_label(self, labels: np.ndarray) -> int:
        # Get the most common label among the nearest neighbors.
        return Counter(labels).most_common(1)[0][0]
