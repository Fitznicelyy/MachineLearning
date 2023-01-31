# Naive Bayes is a probabilistic machine learning algorithm based on Bayes' theorem,
#     which is used for classification tasks. The "naive" in the name comes from the assumption that
#     all the features in the dataset are independent from each other, which is rarely the case in real-world data.
#     Despite this assumption, Naive Bayes can still perform well in practice,
#     especially when the relationship between the features and the target variable is complex and hard to model.
# There are different types of Naive Bayes algorithms, such as
#     Gaussian Naive Bayes, Multinomial Naive Bayes, and Bernoulli Naive Bayes.
#     The choice of algorithm depends on the type of data and the problem you are trying to solve.
# Gaussian Naive Bayes is used when the features are continuous and normally distributed.
#     It's used to predict a continuous target variable based on a Gaussian probability distribution.
# Multinomial Naive Bayes is used when the features are discrete and count-based, such as text classification problems.
#     It's used to predict a categorical target variable based on a multinomial probability distribution.
# Bernoulli Naive Bayes is used when the features are binary, such as spam detection.
#     It's used to predict a binary target variable based on a Bernoulli probability distribution.
# Naive Bayes is used in a variety of applications such as:
#     •	Text classification
#     •	Spam detection
#     •	Sentiment analysis
#     •	Medical diagnosis
#     •	Fraud detection
# Naive Bayes is simple to implement and scale, it's efficient to train and it can handle high-dimensional datasets,
#     it's also useful when the data is limited and it's a good algorithm for text classification.
#     However, it's sensitive to irrelevant features and it's not suitable for regression tasks.


import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from collections import defaultdict
from math import log

# Keep in mind that Naive Bayes algorithm is a probabilistic algorithm which is based on Bayes Theorem,
#     it makes an assumption that all the features are independent of each other which is not always true in real-world scenarios.
#     It's used for classification problems and it's particularly suited for high dimensional data sets.
def naiveBayesPgmpy():
    # Generate some random data
    model = BayesianModel([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D')])
    cpd_a = TabularCPD('A', 2, [[0.2, 0.8]])
    cpd_b = TabularCPD('B', 2, [[0.4, 0.6], [0.5, 0.5]], evidence=['A'], evidence_card=[2])
    cpd_c = TabularCPD('C', 2, [[0.7, 0.3], [0.3, 0.7]], evidence=['A'], evidence_card=[2])
    cpd_d = TabularCPD('D', 2, [[0.8, 0.2, 0.6, 0.4], [0.2, 0.8, 0.4, 0.6]], evidence=['B', 'C'], evidence_card=[2, 2])
    model.add_cpds(cpd_a, cpd_b, cpd_c, cpd_d)

    # Plot the graph
    infer = VariableElimination(model)
    infer.query(variables=['D'])

# This example uses a simple dataset with two input features (x1, x2) and one binary output (y).
#     The input data is defined as a numpy array, and then split into training and test sets using the train_test_split() method from the model_selection module.
#     The GaussianNB class is imported from the sklearn.naive_bayes module and a new instance of the class is created.
#     The fit() method is then used to train the model on the input data.
#     The model can then be used to make predictions on new data using the predict() method.
#     Finally, the accuracy of the model is calculated by comparing the predicted labels to the true labels of the test data set.
def naiveBayesScikitLearn():
    # Input data
    X = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]])
    y = np.array([0, 0, 1, 1, 1])

    # Create a GaussianNB model
    model = GaussianNB()

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
    
class CreateNaiveBayes:
    def fit(self, X, y):
        self.X, self.y = X, y
        self.classes = np.unique(y)
        self.parameters = []
        for i, c in enumerate(self.classes):
            X_where_c = X[np.where(y == c)]
            self.parameters.append([])
            for j in range(X.shape[1]):
                col = X_where_c[:, j]
                parameters = {"mean": col.mean(), "var": col.var()}
                self.parameters[i].append(parameters)

    def _calculate_likelihood(self, mean, var, x):
        eps = 1e-4
        coef = 1.0 / np.sqrt(2 * np.pi * var + eps)
        exp = np.exp(- (x - mean) ** 2 / (2 * var + eps))
        return coef * exp

    def _calculate_prior(self, c):
        return np.mean(self.y == c)

    def predict(self, X):
        y_pred = []
        for x in X:
            posteriors = []
            for i, c in enumerate(self.classes):
                likelihood = 1.0
                for j in range(len(x)):
                    mean = self.parameters[i][j]["mean"]
                    var = self.parameters[i][j]["var"]
                    likelihood *= self._calculate_likelihood(mean, var, x[j])
                prior = self._calculate_prior(c)
                posterior = likelihood * prior
                posteriors.append(posterior)
            y_pred.append(self.classes[np.argmax(posteriors)])
        return y_pred

# This implementation assumes that the input data, X, is a two-dimensional array,
#     where each row represents a sample and each column represents a feature.
#     The target variable, y, is a one-dimensional array, where each element corresponds to the class of the corresponding samplein X.
# You can use this implementation by creating an instance of the NaiveBayes class,
#     fitting it to your data using the fit method, and then making predictions for new data using the predict method.
def naiveBayes():
    nb = CreateNaiveBayes()

    X = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]])
    y = np.array([0, 0, 1, 1, 1])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train the model on the training data
    nb.fit(X_train, y_train)

    # Predict labels for test data
    y_pred = nb.predict(X_test)

    # Evaluate the model
    acc = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {acc}')

# This is a simple implementation of Naive Bayes for binary classification.
#     The train function takes in training data and labels and returns the class probabilities and feature probabilities
#     for each class.
#     The predict function takes in class probabilities, feature probabilities, and a test data point, and returns the predicted class.
#     Note that this implementation uses a defaultdict to handle cases where a feature may not have been seen in the training data.
def naiveBayes2():
    def train(data, labels):
        classes = set(labels)
        class_probabilities = defaultdict(float)
        feature_probabilities = defaultdict(lambda: defaultdict(float))
        
        # calculate class probabilities
        for c in classes:
            class_probabilities[c] = labels.count(c) / float(len(labels))
        
        # calculate feature probabilities for each class
        for c in classes:
            class_data = [x for i, x in enumerate(data) if labels[i] == c]
            class_length = len(class_data)
            
            for feature_vector in class_data:
                for feature in feature_vector:
                    feature_probabilities[feature][c] += 1 / float(class_length)
        
        return class_probabilities, feature_probabilities

    def predict(class_probabilities, feature_probabilities, x):
        classes = class_probabilities.keys()
        best_p = None
        best_class = None
        
        for c in classes:
            p = log(class_probabilities[c])
            for feature in x:
                p += log(feature_probabilities[feature][c])
            
            if best_p is None or p > best_p:
                best_p = p
                best_class = c
        
        return best_class

    # example usage
    data = [[1, 0, 1], [0, 1, 0], [1, 1, 1], [0, 0, 0]]
    labels = [1, 0, 1, 0]
    class_probabilities, feature_probabilities = train(data, labels)

    x = [1, 1, 0]
    print(predict(class_probabilities, feature_probabilities, x))

if __name__ == '__main__':
    naiveBayes2()