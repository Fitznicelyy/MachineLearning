# A decision tree is a type of supervised machine learning algorithm that is used for both classification and regression tasks.
#     Its a tree-based model where each internal node represents a feature, each branch represents a decision based on that feature,
#     and each leaf node represents an outcome or prediction.
#     The goal of decision tree is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.
# A decision tree is built recursively by selecting the feature that best splits the data into subsets with similar values for the target variable.
#     This process is repeated on each subset until the leaf nodes only contain samples with the same target variable value
#     or reaching a pre-defined stopping criteria (e.g. maximum depth, minimum number of samples per leaf).
# Decision trees are used in a variety of applications such as:
#     •	Fraud detection
#     •	Medical diagnosis
#     •	Credit scoring
#     •	Customer churn prediction
#     •	Image classification
# Decision trees are simple to understand and interpret, they can handle both numerical and categorical data,
#     they can handle missing data, and they can be used for both classification and regression tasks.
#     However, decision trees tend to overfit with large datasets, and they are sensitive to small variations in the data.

import math
import graphviz
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree

# Keep in mind that decision trees are a popular method for classification and regression, they are easy to understand and interpret, and they can handle both categorical and numerical data.
def decisionTreeGraphviz():
    # Generate some random data
    X = [[0, 0], [1, 1]]
    Y = [0, 1]

    # Fit the model
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)

    # Plot the decision tree
    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("decision_tree")

def decisionTree():
    # Generate some example data
    data = [['Sunny', 'Hot', 'High', 'Weak', 'No'],
            ['Sunny', 'Hot', 'High', 'Strong', 'No'],
            ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
            ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
            ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
            ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
            ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
            ['Sunny', 'Mild', 'High', 'Weak', 'No'],
            ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
            ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
            ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
            ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
            ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
            ['Rain', 'Mild', 'High', 'Strong', 'No']]

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

    # Define a function to split a dataset on a given feature
    def split_dataset(dataset, feature_index, feature_value):
        left_dataset = []
        right_dataset = []
        for row in dataset:
            if row[feature_index] == feature_value:
                left_dataset.append(row)
            else:
                right_dataset.append(row)
        return left_dataset, right_dataset

    # Define a function to find the best feature to split a dataset on
    def find_best_feature(dataset):
        best_feature_index = None
        best_information_gain = 0
        current_entropy = entropy(dataset)
        num_features = len(dataset[0]) - 1
        for feature_index in range(num_features):
            feature_values = set([row[feature_index] for row in dataset])
            for feature_value in feature_values:
                left_dataset, right_dataset = split_dataset(dataset, feature_index, feature_value)
                probability = len(left_dataset) / len(dataset)
                information_gain = current_entropy - probability * entropy(left_dataset) - (1 - probability) * entropy(right_dataset)
                if information_gain > best_information_gain:
                    best_feature_index = feature_index
                    best_information_gain = information_gain
        return best_feature_index

    # Define a function to create a leaf node
    def create_leaf(dataset):
        label_counts = {}
        for row in dataset:
            label = row[-1]
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
        return max(label_counts, key=label_counts.get)

    # Define a function to create a decision tree
    def create_tree(dataset, feature_names):
        class_labels = [row[-1] for row in dataset]
        if class_labels.count(class_labels[0]) == len(class_labels):
            return class_labels[0]
        if len(dataset[0]) == 1:
            return create_leaf(dataset)
        best_feature_index = find_best_feature(dataset)
        best_feature_name = feature_names[best_feature_index]
        tree = {best_feature_name: {}}
        # del feature_names[best_feature_index]
        feature_values = set([row[best_feature_index] for row in dataset])
        for feature_value in feature_values:
            sub_feature_names = feature_names[:]
            tree[best_feature_name][feature_value] = create_tree(split_dataset(dataset, best_feature_index, feature_value)[0], sub_feature_names)
        return tree

    # Define the feature names
    feature_names = ['Outlook', 'Temperature', 'Humidity', 'Wind']

    # Create the decision tree
    tree = create_tree(data, feature_names)
    print(tree)
   
def DecisionTree2():
    # Define a class for the Decision Tree node
    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, prediction=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.prediction = prediction

    # Define the gini impurity calculation function
    def gini_impurity(labels):
        total = len(labels)
        count = {}
        for label in labels:
            if label in count:
                count[label] += 1
            else:
                count[label] = 1
        impurity = 1
        for key in count:
            prob = count[key] / total
            impurity -= prob ** 2
        return impurity

    # Define the split calculation function
    def split(X, y, feature, threshold):
        left_indices = [i for i in range(len(X)) if X[i][feature] < threshold]
        right_indices = [i for i in range(len(X)) if X[i][feature] >= threshold]
        return (X[left_indices], y[left_indices]), (X[right_indices], y[right_indices])

    # Define the best split calculation function
    def best_split(X, y):
        best_feature = None
        best_threshold = None
        best_impurity = float('inf')
        for feature in range(len(X[0])):
            values = set([X[i][feature] for i in range(len(X))])
            for threshold in values:
                left, right = split(X, y, feature, threshold)
                left_impurity = gini_impurity(left[1])
                right_impurity = gini_impurity(right[1])
                impurity = len(left[0]) / len(X) * left_impurity + len(right[0]) / len(X) * right_impurity
                if impurity < best_impurity:
                    best_impurity = impurity
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    # Define the build tree function
    def build_tree(X, y, depth=0, max_depth=None):
        label_counts = {}
        for label in y:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
        if len(label_counts) == 1 or (max_depth is not None and depth >= max_depth):
            return Node(prediction=list(label_counts.keys())[0])
        feature, threshold = best_split(X, y)
        left, right = split(X, y, feature, threshold)
        left_node = build_tree(left[0], left[1], depth + 1, max_depth)
        right_node = build_tree(right[0], right[1], depth + 1, max_depth)
        return Node(feature=feature, threshold=threshold, left=left_node, right=right_node)

    # Define a function to make predictions using the decision tree
    def predict(node, x):
        if node.prediction is not None:
            return node.prediction
        if x[node.feature] < node.threshold:
            return predict(node.left, x)
        else:
            return predict(node.right, x)

    # Define the fit function
    def fit(X_train, y_train, max_depth=None):
        return build_tree(X_train, y_train, max_depth=max_depth)

    # Generate some sample data
    np.random.seed(0)
    X = np.random.rand(100, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)

    # Split the data into training and testing sets
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]

    # Train the decision tree
    root = fit(X_train, y_train)

    # Use the decision tree to make predictions on the test set
    y_pred = np.array([predict(root, x) for x in X_test])

    # Evaluate the accuracy of the predictions
    accuracy = np.mean(y_pred == y_test)
    print("Accuracy:", accuracy)

def DecisionTree3():
    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, prediction=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.prediction = prediction

    def predict(node, x):
        if node.prediction is not None:
            return node.prediction
        
        if x[node.feature] < node.threshold:
            return predict(node.left, x)
        else:
            return predict(node.right, x)

    def get_entropy(y):
        _, counts = np.unique(y, return_counts=True)
        probas = counts / counts.sum()
        return -np.sum(probas * np.log2(probas))

    def find_best_split(X, y):
        n_samples, n_features = X.shape
        best_feature, best_threshold = None, None
        min_entropy = float("inf")
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = X[:, feature] < threshold
                right_idx = ~left_idx
                y_left, y_right = y[left_idx], y[right_idx]
                entropy = get_entropy(y_left) + get_entropy(y_right)
                
                if entropy < min_entropy:
                    best_feature = feature
                    best_threshold = threshold
                    min_entropy = entropy
        
        return best_feature, best_threshold

    def fit(X, y, depth=0, max_depth=None):
        n_samples, n_features = X.shape
        unique_classes, counts = np.unique(y, return_counts=True)
        
        # If all the samples belong to one class or if we have reached the maximum depth, return a leaf node
        if len(unique_classes) == 1 or (max_depth is not None and depth == max_depth):
            return Node(prediction=unique_classes[np.argmax(counts)])
        
        feature, threshold = find_best_split(X, y)
        left_idx = X[:, feature] < threshold
        right_idx = ~left_idx
        left_node = fit(X[left_idx], y[left_idx], depth + 1, max_depth)
        right_node = fit(X[right_idx], y[right_idx], depth + 1, max_depth)
        
        return Node(feature=feature, threshold=threshold, left=left_node, right=right_node)

    # Load and split the data
    # ...
    # Load the data into a pandas dataframe
    data = pd.read_csv('data.csv')

    # Split the data into features (X) and target (y)
    X = data.drop('target_column', axis=1)
    y = data['target_column']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the decision tree
    root = fit(X_train, y_train, max_depth=5)

    # Use the decision tree to make predictions on the test set
    y_pred = np.array([predict(root, x) for x in X_test])

    # Evaluate the accuracy of the predictions
    accuracy = np.mean(y_pred == y_test)
    print("Accuracy:", accuracy)

if __name__ == '__main__':
    decisionTree()

# 'Outlook':
#     'Rain':
#         'Wind':
#             'Weak': 'Yes'
#             'Strong': 'No'
#     'Overcast': 'Yes'
#     'Sunny':
#         'Humidity':
#             'Normal': 'Yes'
#             'High': 'No'
