from models.k_nearest_neighbors import KNearestNeighbors
from models.multiple_linear_regression import MultipleLinearRegression
from models.sklearn_wrap import LassoWrapper

import pandas as pd


# Load regression data_set
regression_df = pd.read_csv("data/regression_dataset.csv")
X_regression = regression_df.drop(columns=["target"]).values
y_regression = regression_df["target"].values

# Load classification data_set
classification_df = pd.read_csv("data/classification_dataset.csv")
X_classification = classification_df.drop(columns=["target"]).values
y_classification = classification_df["target"].values

# Multiple Linear Regression
mlr = MultipleLinearRegression()
mlr.fit(X_regression, y_regression)

mlr_predictions = mlr.predict(X_regression)

print("MLR Predictions:", mlr_predictions)  # predictions of the data
print("Intercept:", mlr.intercept)  # bias term
print("Coefficients:", mlr.coefficients)  # weights


# K-Nearest Neighbors
knn = KNearestNeighbors()
knn.fit(X_classification, y_classification)
knn_predictions = knn.predict(X_classification)
print("KNN Predictions:", knn_predictions)


# Lasso Regression
lasso = LassoWrapper()
lasso.fit(X_regression, y_regression)
lasso_predictions = lasso.predict(X_regression)
print("Lasso Predictions:", lasso_predictions)
print("Lasso Parameters:", lasso.parameters)
