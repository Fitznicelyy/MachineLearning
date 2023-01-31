import random
from collections import Counter

# This code defines a loss function, a gradient function, and a decision stump training function.
#     It also defines a train function that trains a gradient boosting model by training many decision stumps
#     and updating the predicted labels after each one.
#     Finally, it defines a predict function that uses a gradient boosting model to make predictions for new feature vectors.
def gradientBoosting():
    def loss(y_true, y_pred):
        # Compute the mean squared error between the true labels and the predicted labels.
        
        # Parameters:
        # y_true (List[int]): The true labels.
        # y_pred (List[int]): The predicted labels.
        
        # Returns:
        # float: The mean squared error.
        return sum((y1 - y2) ** 2 for y1, y2 in zip(y_true, y_pred)) / len(y_true)

    def gradient(y_true, y_pred):
        # Compute the gradient of the loss function with respect to the predicted labels.
        
        # Parameters:
        # y_true (List[int]): The true labels.
        # y_pred (List[int]): The predicted labels.
        
        # Returns:
        # List[float]: The gradient of the loss with respect to the predicted labels.
        return [2 * (y1 - y2) for y1, y2 in zip(y_true, y_pred)]

    def decision_stump(x, y_true, y_pred, feature):
        # Train a decision stump (a tree with maximum depth 1) to predict the gradient of the loss
        # with respect to the predicted labels.
        
        # Parameters:
        # x (List[List[int]]): The feature vectors.
        # y_true (List[int]): The true labels.
        # y_pred (List[int]): The predicted labels.
        # feature (int): The index of the feature to use for the decision.
        
        # Returns:
        # int: The threshold for the decision stump.
        gradients = gradient(y_true, y_pred)
        feature_values = [row[feature] for row in x]
        thresholds = sorted(set(feature_values))
        
        best_threshold = None
        best_gain = float('-inf')
        for threshold in thresholds:
            gains = [gradients[i] if feature_values[i] < threshold else -gradients[i] for i in range(len(x))]
            gain = sum(gains)
            if gain > best_gain:
                best_threshold = threshold
                best_gain = gain
        
        return best_threshold

    def train(x, y_true, n_estimators=100, learning_rate=0.1):
        # Train a gradient boosting classifier on the given data.
        
        # Parameters:
        # x (List[List[int]]): The feature vectors.
        # y_true (List[int]): The true labels.
        # n_estimators (int): The number of trees to use in the gradient boosting model.
        # learning_rate (float): The learning rate for the gradient boosting model.
        
        # Returns:
        # List[Tuple[int, Tuple[int, int]]]: The trained gradient boosting model.
        trees = []
        y_pred = [0.0 for _ in range(len(y_true))]
        for i in range(n_estimators):
            gradients = gradient(y_pred) # fix
            return gradients
            # Train a decision stump to predict the gradient of the loss
        decision_feature = random.randint(0, len(x[0]) - 1)
        decision_threshold = decision_stump(x, y_true, y_pred, decision_feature)
        
        # Update the predicted labels
        y_pred = [y_pred[i] + learning_rate * (1 if x[i][decision_feature] < decision_threshold else -1) for i in range(len(x))]
        
        # Store the decision stump in the list of trees
        trees.append((decision_feature, decision_threshold))
    
        return trees

    def predict(trees, x):
        # Predict the label for a new feature vector using a gradient boosting model.
        
        # Parameters:
        # trees (List[Tuple[int, Tuple[int, int]]]): The trained gradient boosting model.
        # x (List[int]): The feature vector to predict the label for.
        
        # Returns:
        # int: The predicted label.
        y_pred = 0.0
        for decision_feature, decision_threshold in trees:
            y_pred += 1 if x[decision_feature] < decision_threshold else -1
        return 1 if y_pred >= 0 else 0
    

    x = [[1, 2], [2, 1], [2, 3], [3, 2]]
    y = [0, 0, 1, 1]

    # Train the model by calling the train function and passing in the data:
    trees = train(x, y, learning_rate=0.1, num_trees=10)

    #Make predictions with the trained model by calling the predict function and passing in the feature vectors you want to predict:
    new_x = [1, 2]
    prediction = predict(trees, new_x)
    print(prediction)

def gradientBoosting2():
    def train(x, y, learning_rate=0.1, num_trees=10):
        # Train a gradient boosting model.
        
        # Parameters:
        # x (List[List[int]]): The feature vectors.
        # y (List[int]): The target labels.
        # learning_rate (float, optional): The learning rate. Defaults to 0.1.
        # num_trees (int, optional): The number of decision trees in the model. Defaults to 10.
        
        # Returns:
        # List[Tuple[int, Tuple[int, int]]]: The trained gradient boosting model.
        def loss(y_true, y_pred):
            # Compute the loss for a given set of true labels and predicted labels.
            
            # Parameters:
            # y_true (List[int]): The true labels.
            # y_pred (List[int]): The predicted labels.
            
            # Returns:
            # float: The loss.
            return sum([(y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true))]) / len(y_true)
        
        def gradient(y_true, y_pred):
            # Compute the gradient of the loss for a given set of true labels and predicted labels.
            
            # Parameters:
            # y_true (List[int]): The true labels.
            # y_pred (List[int]): The predicted labels.
            
            # Returns:
            # List[int]: The gradient of the loss with respect to the predicted labels.
            return [2 * (y_true[i] - y_pred[i]) for i in range(len(y_true))]
        
        def decision_stump(x, y_true, y_pred, decision_feature):
            # Train a decision stump to predict the gradient of the loss.
            
            # Parameters:
            # x (List[List[int]]): The feature vectors.
            # y_true (List[int]): The true labels.
            # y_pred (List[int]): The predicted labels.
            # decision_feature (int): The feature to make the decision based on.
            
            # Returns:
            # int: The threshold for the decision stump.
            sorted_x = sorted(enumerate(x), key=lambda x_i: x_i[1][decision_feature])
            sorted_y_true = [y_true[i[0]] for i in sorted_x]
            sorted_y_pred = [y_pred[i[0]] for i in sorted_x]
            sorted_gradient = gradient(sorted_y_true, sorted_y_pred)
            
            min_loss = float('inf')
            best_threshold = None
            for i in range(1, len(sorted_x)):
                loss_left = loss(sorted_y_true[:i], sorted_y_pred[:i] + [sorted_gradient[i - 1]] * i)
                loss_right = loss(sorted_y_true[i:], [sorted_gradient[i - 1]] * (len(sorted()))) # fix
                if loss_left + loss_right < min_loss:
                    min_loss = loss_left + loss_right
                    best_threshold = sorted_x[i][1][decision_feature]
        
            return best_threshold
        
        def predict(x, tree):
            # Predict the target label for a given feature vector using a gradient boosting model.
            
            # Parameters:
            # x (List[int]): The feature vector.
            # tree (List[Tuple[int, Tuple[int, int]]]): The gradient boosting model.
            
            # Returns:
            # int: The predicted target label.
            y_pred = 0
            for i in range(len(tree)):
                feature, (threshold, direction) = tree[i]
                if x[feature] < threshold:
                    y_pred += direction * learning_rate
            return y_pred
        
        trees = []
        y_pred = [0 for i in range(len(y))]
        for t in range(num_trees):
            gradient_t = gradient(y, y_pred)
            tree_t = []
            for j in range(len(x[0])):
                threshold = decision_stump(x, y, y_pred, j)
                direction = sum([gradient_t[i] for i in range(len(x)) if x[i][j] < threshold]) / sum([1 for i in range(len(x)) if x[i][j] < threshold])
                tree_t.append((j, (threshold, direction)))
            trees.append(tree_t)
            y_pred = [y_pred[i] + predict(x[i], tree_t) for i in range(len(x))]
        
        return trees



if __name__ == '__main__':
    gradientBoosting()
