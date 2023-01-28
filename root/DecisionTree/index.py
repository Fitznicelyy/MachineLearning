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
        del feature_names[best_feature_index]
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

if __name__ == '__main__':
    decisionTree()