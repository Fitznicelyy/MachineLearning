# K-Nearest Neighbor (KNN) is a type of instance-based learning or non-parametric algorithm that can be used for both classification and regression tasks.
#     The idea behind KNN is to find the k-nearest data points in the feature space for a given test point,
#     and then predict the class or value for that test point based on the majority class or average value of its k-nearest neighbors.
#     The value of k is a user-defined parameter that determines the number of nearest neighbors to consider.
# The algorithm works by storing all the training data and then, when a new test point is encountered,
#     the k-nearest points to that point are found from the stored data.
#     The algorithm then uses the majority class of those k-nearest points as the prediction for the class of the test point.
# KNN is used in a variety of applications such as:
#     •	Image recognition
#     •	Handwriting recognition
#     •	Recommender systems
#     •	Anomaly detection
#     •	Video recognition
# KNN is simple to understand and implement, it can handle both numerical and categorical data and it can handle multi-class problems.
#     However, the algorithm can be computationally expensive, it requires a large amount of memory to store the entire dataset
#     and it's sensitive to irrelevant features and the scale of the data.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets

# A visual representation of a k-nearest neighbor (k-NN) algorithm is typically a scatter plot of the data points,
#     with a new point (the point for which the classification is to be determined) highlighted.
#     The k-NN algorithm classifies the new point based on the class labels of the k-nearest data points.
#     The number of nearest data points considered is represented by the value of k.
def knnMatplotLib():
    # Generate some random data
    X, y = make_classification(n_classes=2, n_features=2, n_informative=2, n_redundant=0)

    # Fit the model
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)

    # Plot the data and the new point
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.scatter(0, 0, c='r', marker='x', s=100)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

# A visual representation of a k-nearest neighbor (KNN) algorithm is typically a scatter plot of the data points,
#     with a decision boundary that separates the different classes.
#     The decision boundary is determined by the k nearest points to a given test point,
#     and the class of the majority of these k points is assigned to the test point.
#     The k nearest points are typically determined by calculating the Euclidean distance between the test point and all other points in the dataset.
def knnMatplotLib2():
    # Generate some random data
    X, y = datasets.make_classification(n_features=2, n_redundant=0, n_informative=2,
                            n_clusters_per_class=1, random_state=4)

    # Fit the model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)

    # Plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                        np.arange(y_min, y_max, 0.01))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

# In this example, the decision boundary is more like a gradient and not well defined.
#     As you can see in the plot, the decision boundary is smooth because it assigns a probability of each point being in a certain class.
def knnMatplotLib3():
    # Generate some random data
    X, y = datasets.make_classification(n_features=2, n_redundant=0, n_informative=2,
                            n_clusters_per_class=1, random_state=4)

    # Fit the model
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)

    # Plot the decision boundary
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
    Z = knn.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='Paired')
    plt.show()

# This implementation assumes that the input data, X, is a two-dimensional array,
#     where each row represents a sample and each column represents a feature. The target variable, y, is a one-dimensional array,
#     where each element corresponds to the class of the corresponding sample in X.
# You can use this implementation by creating an instance of the KNN class,
#     fitting it to your data using the fit method, and then making predictions for new data using the predict method.
class CreateKNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = []
        for x in X:
            distances = []
            for i in range(self.X_train.shape[0]):
                distance = np.linalg.norm(x - self.X_train[i])
                distances.append((distance, self.y_train[i]))
            k_nearest = sorted(distances)[:self.k]
            k_nearest_labels = [label for distance, label in k_nearest]
            y_pred.append(max(set(k_nearest_labels), key=k_nearest_labels.count))
        return y_pred

# In this example the number of neighbors considered for classification is set to 3,
#     but you can change it by passing the number of neighbors you want as an argument in the constructor of the class.
def knn():
    knn = CreateKNN(k=3)

    X = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]])
    y = np.array([0, 0, 1, 1, 1])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train the model on the training data
    knn.fit(X_train, y_train)

    # Predict labels for test data
    y_pred = knn.predict(X_test)

    # Evaluate the model
    acc = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {acc}')

# Here, data is a list of input data points, where each point is a list of feature values followed by a label.
#     The query is a single data point for which we want to find the label and k is the number of nearest neighbors to consider.
#     The euclidean_distance function computes the Euclidean distance between two points,
#     and the knn function sorts the distances to all data points and finds the label of the majority of the k nearest points.
def knn2():
    def euclidean_distance(x1, x2):
        return sum((x1[i] - x2[i]) ** 2 for i in range(len(x1))) ** 0.5

    def createKNN(data, query, k):
        distances = [(euclidean_distance(data[i][:-1], query), data[i][-1]) for i in range(len(data))]
        distances = sorted(distances, key=lambda x: x[0])
        return max(distances[:k], key=lambda x: x[1])[1]

    data = [[1,2,3,'A'],[4,5,6,'B'],[7,8,9,'A']]
    query = [5,5,5]
    k = 2
    print(createKNN(data,query,k))
    # This would output 'A'

if __name__ == '__main__':
    knnMatplotLib()