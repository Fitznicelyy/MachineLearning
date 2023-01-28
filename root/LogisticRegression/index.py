# Logistic Regression is a type of generalized linear model (GLM) that is used for classification tasks.
#     Its used to predict a binary outcome (1 or 0, true or false, yes or no) based on one or more predictor variables (also known as independent variables or features).
#     Its also known as logit regression or maximum-entropy classification.
# The logistic regression model uses a logistic function (also known as the sigmoid function) to predict the probability that an instance belongs to a certain class.
#     The logistic function is defined as follows:
#         p = 1 / (1 + e^(-z))
#         where z is the linear combination of predictor variables and parameters (z = b0 + b1x1 + b2x2 + ...),
#         and p is the probability of a positive outcome (class 1).
        
#         The goal of logistic regression is to find the best values for the parameters that maximize the likelihood of the observed data.
# Logistic regression is used in a variety of applications such as:
#     •	Credit scoring
#     •	Medical diagnosis
#     •	Fraud detection
#     •	Customer churn prediction
#     •	Image classification
# Logistic regression is simple to implement and interpret, its efficient to train and it works well with small datasets, its also useful for binary classification problems,
#     but its not suitable for multi-class classification problems and its sensitive to correlated independent variables.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Keep in mind that logistic regression is a classification algorithm and it is used when the dependent variable is binary.
def logisticRegressionMatplotLib():
    # Generate some random data
    np.random.seed(0)
    x = np.random.randn(100, 1)
    y = np.random.randint(0, 2, size=(100, 1))

    # Fit the model
    w = np.linalg.lstsq(np.c_[np.ones(x.shape[0]), x], y, rcond=None)[0]

    # Plot the data and the model
    plt.scatter(x, y)
    plt.plot(x, 1 / (1 + np.exp(-w[0] - w[1] * x)), 'r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# This example uses a simple dataset with two input features (x1, x2) and one binary output (y).
#     The input data is defined as a numpy array, and then split into training and test sets using the train_test_split() method from the model_selection module.
#     The LogisticRegression class is imported from the scikit-learn library, and a new instance of the class is created with 'lbfgs' as the solver.
#     The fit() method is then used to train the model on the input data. The model can then be used to make predictions on new data using the predict() method.
#     Finally, the accuracy of the model is calculated by comparing the predicted labels to the true labels of the test data set.
def logisticRegressionScikitLearn():
    # Input data
    X = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]])
    y = np.array([0, 0, 1, 1, 1])

    # Create a logistic regression model
    model = LogisticRegression(solver='lbfgs')

    # Split data into training and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Predict labels for test data
    y_pred = model.predict(X_test)

    # Evaluate the model
    acc = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {acc}')

# In this example, we first generate some example data by creating a list of x values and a list of y values that are binary.
#     We then initialize the weights (w) with random values.
#     We then define the sigmoid function, which is used to convert the output of the linear model into a probability between 0 and 1.
#     We also define the log-likelihood function, which is used to measure the likelihood of the data given the model, and the gradient of the log-likelihood function, which is used to update the weights.
#     After that we define the step size and number of iterations for the gradient ascent and perform the gradient ascent to update the weights.
#     Finally, we use the final weights (w) to predict the output (y_pred) for the input data, and we print the predictions.
#     Its important to note that this is a simplified example and in practice, its more efficient and common to use optimization libraries such as scikit-learn to perform logistic regression.
def logisticRegression():
    # Generate some example data
    x = [[1, 2], [2, 3], [3, 4], [4, 5]]
    y = [0, 0, 1, 1]

    # Initialize the weights with random values
    w = [np.random.random(), np.random.random()]

    # Define the sigmoid function
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    # Define the log-likelihood function
    def log_likelihood(x, y, w):
        return sum([y[i] * np.log(sigmoid(np.dot(w, x[i]))) + (1 - y[i]) * np.log(1 - sigmoid(np.dot(w, x[i]))) for i in range(len(x))])

    # Define the gradient of the log-likelihood function
    def gradient(x, y, w):
        return [sum([(y[i] - sigmoid(np.dot(w, x[i]))) * x[i][j] for i in range(len(x))]) for j in range(len(w))]

    # Define the step size and number of iterations
    step_size = 0.01
    num_iterations = 1000

    # Perform the gradient ascent
    for i in range(num_iterations):
        w = [w[j] + step_size * gradient(x, y, w)[j] for j in range(len(w))]

    # Predict the output for the input data
    y_pred = [sigmoid(np.dot(w, x[i])) for i in range(len(x))]

    # Print the predictions
    print(y_pred)

if __name__ == '__main__':
    logisticRegressionMatplotLib()