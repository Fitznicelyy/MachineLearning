# Random Forest is an ensemble learning algorithm that can be used for both classification and regression tasks.
#     The algorithm creates a collection of decision trees, called a "forest,"
#     and then makes predictions by averaging the predictions of the individual trees.
#     The process of creating the individual trees involves randomly selecting a subset of the data,
#     as well as a subset of the features, to use at each decision node of the tree.
# The randomness in the data and feature subsets helps to decorrelate the trees,
#     so that the predictions of the ensemble are less prone to overfitting than a single decision tree.
#     The idea behind this approach is that by aggregating the predictions of multiple models,
#     the final predictions will be more robust and accurate.
# Random Forest algorithm is used in a variety of applications such as:
#     •	Fraud detection
#     •	Image classification
#     •	Face recognition
#     •	Object detection
#     •	Medical diagnosis
#     •	Climate modeling
# Random Forest is easy to use and interpret, it can handle both numerical and categorical data and it can handle high-dimensional data.
#     However, the algorithm can be computationally expensive and it's sensitive to irrelevant features.
#     Also, the model can be too complex and difficult to interpret which can be a disadvantage.

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree
import random
from collections import Counter

# A visual representation of a random forest algorithm is often a decision tree, which is a flowchart-like tree structure,
#     where each internal node represents a feature(or attribute), the branch represents a decision rule, and each leaf node represents the outcome.
#     In Random Forest, multiple decision trees are created and combined to make a final prediction.

# This will generate a visual representation of the decision tree. As Random Forest uses multiple decision tree,
#     the final decision boundary is an ensemble of all the decision boundaries of individual trees.
def randomForestMatplotLib():
    # Generate some random data
    X, y = make_classification(n_features=4, n_classes=2)

    # Fit the model
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(X, y)

    # Plot the decision tree
    tree.plot_tree(clf.estimators_[0], filled=True)
    plt.show()

# This example uses a simple dataset with two input features (x1, x2) and one binary output (y).
#     The input data is defined as a numpy array, and then split into training and test sets using the train_test_split() method from the model_selection module.
def randomForestScikitLearn():
    # Input data
    X = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]])
    y = np.array([0, 0, 1, 1, 1])

    # Create a RandomForestClassifier model
    model = RandomForestClassifier(n_estimators=100)

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


# Note that this is just a simple implementation of the Random Forest algorithm and can be optimized further.
#     The main idea behind the implementation is to build multiple decision trees,
#     each trained on a random subset of the training data, and then combine their predictions to get the final prediction.
def randomForest():
    def train(data, labels, n_trees=10, sample_size=None, max_depth=None, min_samples_split=2):
        # Train a random forest classifier using the given data and labels.
        
        # Parameters:
        # data (List[List[int]]): The training data, represented as a list of feature vectors.
        # labels (List[int]): The corresponding class labels for each feature vector.
        # n_trees (int): The number of trees to use in the forest.
        # sample_size (int): The number of samples to use for each tree. If None, use all samples.
        # max_depth (int): The maximum depth of each tree. If None, grow the tree until all leaves are pure.
        # min_samples_split (int): The minimum number of samples required to split an internal node.
        
        # Returns:
        # List[int]: The tree roots for each of the `n_trees` trees in the forest.
        sample_size = len(data) if sample_size is None else sample_size
        max_depth = float('inf') if max_depth is None else max_depth
        
        def build_tree(data, labels, depth):
            # Build a single decision tree using the given data and labels.
            
            # Parameters:
            # data (List[List[int]]): The training data, represented as a list of feature vectors.
            # labels (List[int]): The corresponding class labels for each feature vector.
            # depth (int): The current depth of the tree.
            
            # Returns:
            # int: The root node of the tree.
            if depth >= max_depth or len(data) <= min_samples_split:
                return Counter(labels).most_common(1)[0][0]
            
            feature = random.randint(0, len(data[0]) - 1)
            feature_values = set(row[feature] for row in data)
            node = (feature, {})
            for value in feature_values:
                sub_data = [row for row in data if row[feature] == value]
                sub_labels = [label for i, label in enumerate(labels) if data[i][feature] == value]
                child = build_tree(sub_data, sub_labels, depth + 1)
                node[1][value] = child
            
            return node
        
        trees = []
        for i in range(n_trees):
            indices = random.sample(range(len(data)), sample_size)
            sample_data = [data[i] for i in indices]
            sample_labels = [labels[i] for i in indices]
            trees.append(build_tree(sample_data, sample_labels, 0))
        
        return trees

    def predict(trees, x):
        # Predict the class of the given feature vector using the given random forest.
        
        # Parameters:
        # trees (List[int]): The tree roots for each of the trees in the forest.
        # x (List[int]): The feature vector to classify.
        #     Returns:
        # int: The predicted class label.
        predictions = [predict_tree(tree, x) for tree in trees]
        return Counter(predictions).most_common(1)[0][0]

    def predict_tree(tree, x):
        # Predict the class of the given feature vector using the given decision tree.
        
        # Parameters:
        # tree (int): The root node of the decision tree.
        # x (List[int]): The feature vector to classify.
        
        # Returns:
        # int: The predicted class label.
        if type(tree) is int:
            return tree
        
        feature, children = tree
        return predict_tree(children[x[feature]], x)

    # This code generates 100 random feature vectors and corresponding class labels,
    #     then trains a random forest with 10 trees on this data.
    #     Finally, it generates a random feature vector and makes a prediction for its class using the trained random forest.

    # generate some toy data and labels
    data = [[random.randint(0, 2) for _ in range(5)] for _ in range(100)]
    labels = [random.randint(0, 2) for _ in range(100)]

    # train the random forest on the data
    trees = train(data, labels, n_trees=10, sample_size=None, max_depth=None, min_samples_split=2)

    # make a prediction for a new feature vector
    x = [random.randint(0, 2) for _ in range(5)]
    print(predict(trees, x))

if __name__ == '__main__':
    randomForest()