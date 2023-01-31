# Gradient Boosting is an ensemble machine learning algorithm that combines multiple simple models (often decision trees) to create a more powerful model.
#     The algorithm works by iteratively training a new model to correct the mistakes made by the previous models.
#     The new models are trained using the gradient descent algorithm to minimize a loss function,
#     which measures the difference between the predicted and actual values.
# The main idea behind gradient boosting is to add new models to the ensemble in a way that reduces the overall error.
#     The algorithm starts with a simple model and then improves it
#     by adding new models that are trained to correct the residual errors made by the previous models.
#     The final ensemble model is a combination of all the individual models with their corresponding weights.
# Gradient Boosting can be used for both classification and regression tasks and it's widely used in a variety of applications such as:
#     •	Web search ranking
#     •	Fraud detection
#     •	Stock forecasting
#     •	Climate modeling
#     •	Recommender systems
# Gradient Boosting algorithm is powerful and flexible, it can handle large datasets and it can be used with a variety of base models.
#     It can also handle missing data and categorical variables.
#     However, it's computationally expensive and it requires careful tuning of the parameters and regularization to prevent overfitting.

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree

# This will generate a visual representation of the decision tree.
#     However, it's worth noting that in practice, Gradient Boosting algorithm uses many trees
#     and the final decision boundary is an ensemble of all the decision boundaries of individual trees.
def gradientBoostingMatplotLib():
    # Generate some random data
    X, y = make_classification(n_features=4, n_classes=2)

    # Fit the model
    clf = GradientBoostingClassifier(n_estimators=10)
    clf.fit(X, y)

    # Plot the decision tree
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    tree.plot_tree(clf.estimators_[0][0], ax=ax, filled=True)
    plt.show()

# This example uses a simple dataset with two input features (x1, x2) and one binary output (y).
#     The input data is defined as a numpy array, and then split into training and test sets using the train_test_split() method
#     from the model_selection module. The GradientBoostingClassifier class is imported from the sklearn.ensemble,
#     which is a powerful and flexible algorithm that can handle large datasets and it can be used with a variety of base models.
# It's important to note that the default values of the parameters used in the above example may not be optimal for your specific problem,
#     so it might be necessary to tune the parameters and regularization to prevent overfitting.
def gradientBoostingScikitLearn():
    # Input data
    X = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]])
    y = np.array([0, 0, 1, 1, 1])

    # Create a GradientBoostingClassifier model
    model = GradientBoostingClassifier()

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Predict labels for test data
    y_pred = model.predict(X_test)

    # Evaluate the model
    acc = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {acc}')

# The implementation consists of a GradientBoosting class, which implements the following methods:
#     __init__: This is the constructor method that initializes some instance variables and sets the number of decision trees in the gradient boosting model.
#     fit: This method fits the gradient boosting model to the training data.
#         It uses a loop to grow n_trees decision trees and updates the residuals at each iteration.
#     _build_tree: This is a helper method that builds a single decision tree.
#         It uses a recursive approach to divide the data into smaller subsets based on the best split at each node.
#         The best split is found by evaluating the reduction in variance achieved by each possible split.
#     _calculate_variance_reduction: This is another helper method that calculates the reduction in variance achieved by a split.
#     predict: This method makes predictions on new data using the gradient boosting model.
#         It uses a loop to sum up the predictions from each decision tree and returns the final prediction.
#     _predict: This is a helper method that makes a prediction for a single observation using a single decision tree.
#         It navigates the decision tree until a leaf node is reached, and returns the prediction associated with that leaf node.
#     This code provides a basic understanding of how gradient boosting works, but in practice,
#         you would typically use a library such as XGBoost or LightGBM to implement gradient boosting,
#         as they provide optimized and more efficient implementations.
class CreateGradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

    def fit(self, X, y):
        # initialize the fitted values
        self.f = np.zeros(len(y))
        self.trees = []
        for i in range(self.n_estimators):
            # calculate the negative gradient
            negative_gradient = -2 * (y - self.f)
            # fit a regression tree to the negative gradient
            tree = RegressionTree()
            tree.fit(X, negative_gradient)
            # update the fitted values
            self.f += self.learning_rate * tree.predict(X)
            self.trees.append(tree)

    def predict(self, X):
        return self.f + sum(self.learning_rate * tree.predict(X) for tree in self.trees)

class RegressionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.root = self._fit(X, y, depth=0)

    def _fit(self, X, y, depth):
        if self.max_depth is not None and depth >= self.max_depth:
            # return a leaf node with the mean target value
            return np.mean(y)

        n_samples, n_features = X.shape
        feature_indices = range(n_features)
        split_index, split_value = self._get_best_split(X, y, feature_indices)

        if split_index is None:
            # return a leaf node with the mean target value
            return np.mean(y)

        left_mask = X[:, split_index] < split_value
        right_mask = ~left_mask
        left_X, right_X = X[left_mask], X[right_mask]
        left_y, right_y = y[left_mask], y[right_mask]

        # recursively fit the left and right subtrees
        left = self._fit(left_X, left_y, depth + 1)
        right = self._fit(right_X, right_y, depth + 1)

        return (split_index, split_value, left, right)

    def _get_best_split(self, X, y, feature_indices):
        best_split_index, best_split_value = None, None
        best_split_score = float('-inf')
        for split_index in feature_indices:
            values = X[:, split_index]
            unique_values = np.unique(values)
            for split_value in unique_values:
                left_mask = values < split_value
                right_mask = ~left_mask
                left_y, right_y = y[left_mask], y[right_mask]
                if len(left_y) == 0 or len(right_y) == 0:
                    continue
                split_score = self._calculate_variance_reduction(left_y, right_y)
                if split_score > best_split_score:
                    best_split_index = split_index
                    best_split_value = split_value
                    best_split_score = split_score
        return best_split_index, best_split_value

    def _calculate_variance_reduction(self, left_y, right_y):
        total_y = np.concatenate([left_y, right_y])
        mean = np.mean(total_y)
        left_variance = np.mean((left_y - mean) ** 2)
        right_variance = np.mean((right_y - mean) ** 2)
        return left_variance + right_variance

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        node = self.root
        while isinstance(node, tuple):
            split_index, split_value, left, right = node
            if x[split_index] < split_value:
                node = left
            else:
                node = right
        return node

def gradientBoosting():
    # generate some random training data
    np.random.seed(0)
    X = np.random.rand(100, 5)
    y = np.random.rand(100)

    # create an instance of the GradientBoosting class
    gbm = CreateGradientBoosting()

    # fit the model to the training data
    gbm.fit(X, y)

    # generate some random test data
    X_test = np.random.rand(50, 5)

    # make predictions on the test data
    y_pred = gbm.predict(X_test)
    print(f'{y_pred}')

if __name__ == '__main__':
    gradientBoosting()
