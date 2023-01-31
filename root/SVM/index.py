# Support Vector Machine (SVM) is a supervised learning algorithm that can be used for both classification and regression tasks.
#     The main idea behind the algorithm is to find the best boundary (or hyperplane) that separates
#     the different classes in the data, while maximizing the margin, which is the distance between the boundary and the closest data points from each class.
#     The data points that are closest to the boundary and used to define the margin are called support vectors.
# SVMs are particularly well suited for problems where the number of features is much greater than the number of samples,
#     and for problems where the data is not linearly separable, by projecting the data into a higher-dimensional space,
#     where it can be separated by a linear boundary.
# SVMs are used in a variety of applications such as:
#     •	Text classification
#     •	Image classification
#     •	Hand-written character recognition
#     •	Bioinformatics (classifying proteins)
#     •	Face recognition
# SVMs are powerful, versatile and accurate, they perform well in high-dimensional spaces and they can be used
#     with a variety of kernel functions, which allows them to capture complex relationships between the data.
#     However, they can be sensitive to the choice of kernel and the regularization parameter,
#     and they can be computationally expensive when the number of samples is large.

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# A visual representation of a Support Vector Machine (SVM) algorithm can be illustrated as a plot of the data points,
#     with different colors representing different classes. The SVM algorithm creates a decision boundary, called a hyperplane,
#     that maximally separates the different classes.
#     The data points that are closest to the hyperplane are called support vectors,  and the distance between the hyperplane and the closest data points is called the margin.
#     The goal of the SVM algorithm is to find the hyperplane with the largest margin.
def svmScikitLearn():
    # Generate some random data
    X, y = datasets.make_classification(n_features=2, n_redundant=0, n_informative=2,
                            n_clusters_per_class=1, random_state=4)

    # Fit the model
    svm = SVC(kernel='linear', C=1)
    svm.fit(X, y)

    # Plot the decision boundary
    w = svm.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - svm.intercept_[0] / w[1]

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='Paired')
    plt.plot(xx, yy, 'k-', label="non weighted div")
    plt.legend()
    plt.show()

# This example uses a simple dataset with two input features (x1, x2) and one binary output (y).
#     The input data is defined as a numpy array, and then split into training and test sets using the train_test_split() method
#     from the model_selection module. The SVC class is imported from the sklearn.svm,
#     and the kernel parameter is set to 'linear' to use a linear kernel, which means that the boundary is a straight line.
def svmScikitLearn2():
    # Input data
    X = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]])
    y = np.array([0, 0, 1, 1, 1])

    # Create a SVC model
    model = SVC(kernel='linear')

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Predict labels for test data
    y_pred = model.predict(X_test)

    # Evaluate the model
    acc = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {acc}')

# In this example, the decision boundary is a straight line that separates the data points into two classes.
#     The points closest to the boundary are called support vectors and have a higher influence in determining the boundary.
# It's important to note that SVM can also be used with non-linear decision boundaries by using different kernel functions.
def svmMatplotLib():
    # Generate some random data
    X, y = datasets.make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)

    # Fit the model
    clf = SVC(kernel='linear', C=1)
    clf.fit(X, y)

    # Plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='Paired')
    plt.show()

# This is a simple linear SVM. You can use this class by creating an instance of it and calling the fit method to train the model,
#     and the predict method to make predictions on new data.
# The fit function takes two arguments:
#     •	X: a 2D array of size (n_samples, n_features) representing the training data
#     •	y: a 1D array of size (n_samples) representing the labels for the training data
# The predict function takes one argument:
#     •	X: a 2D array of size (n_samples, n_features) representing the data for which you want to make predictions.
# This is just a simple example that is probably not optimal for most use cases.
#     One of the main limitation of this implementation is that it is only for linear SVM.
class CreateSVM:
    def __init__(self, learning_rate=0.001, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.max_iterations):
            for i in range(n_samples):
                if (y[i] * (np.dot(X[i], self.weights) + self.bias)) < 1:
                    self.weights = self.weights + self.learning_rate * (X[i] * y[i] - (2 * (1/self.max_iterations) * self.weights))
                    self.bias = self.bias + self.learning_rate * y[i]
                else:
                    self.weights = self.weights + self.learning_rate * (-2 * (1/self.max_iterations) * self.weights)

    def predict(self, X):
        return np.sign(np.dot(X, self.weights) + self.bias)

def svm():
    # Input data
    X = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]])
    y = np.array([0, 0, 1, 1, 1])

    # Create a SVC model
    svm = CreateSVM()

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train the model on the training data
    svm.fit(X_train, y_train)

    # Predict labels for test data
    y_pred = svm.predict(X_test)

    # Evaluate the model
    acc = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {acc}')

if __name__ == '__main__':
    svm()
