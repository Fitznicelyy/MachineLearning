import math
import numpy as np

# define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
# def sigmoid(x):
#     return 1 / (1 + math.exp(-x))


# define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Define the log-likelihood function
def log_likelihood(x, y, w):
    return sum([y[i] * np.log(sigmoid(np.dot(w, x[i]))) + (1 - y[i]) * np.log(1 - sigmoid(np.dot(w, x[i]))) for i in range(len(x))])

# Define the gradient of the log-likelihood function
def gradient(x, y, w):
    return [sum([(y[i] - sigmoid(np.dot(w, x[i]))) * x[i][j] for i in range(len(x))]) for j in range(len(w))]

# Define a function to calculate the entropy of a dataset
def entropy(dataset):
    label_counts = {}
    for row in dataset:
        label = row[-1]
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1
    entropy = 0
    for label in label_counts:
        probability = label_counts[label] / len(dataset)
        entropy -= probability * math.log2(probability)
    return entropy

# Define a function to calculate the probability of a given value for a Gaussian distribution
def gaussian_probability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

# Define a function to calculate the euclidean distance between two points
def euclidean_distance(x1, x2):
    return sum((x1[i] - x2[i]) ** 2 for i in range(len(x1))) ** 0.5
