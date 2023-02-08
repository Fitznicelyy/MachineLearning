# A neural network is a set of algorithms modeled loosely after the human brain that are designed to recognize patterns.
#     They are used to interpret sensory data through a kind of machine perception, such as labeling or clustering raw input.
#     They are particularly useful in tasks that involve image, video, audio, and text data,
#     and are used in a wide range of applications such as image recognition, speech recognition, natural language processing, and decision-making systems.
#     They are also used in a variety of industries, including finance, healthcare, and manufacturing.
#     Neural networks can be trained using a variety of techniques, including supervised learning, unsupervised learning, and reinforcement learning.
# A neural network is a type of machine learning algorithm that is modeled after the structure and function of the human brain.
#     It is composed of layers of interconnected nodes, called artificial neurons, which are organized into input, hidden, and output layers.
# Neural networks are used for a wide range of tasks, including image recognition, natural language processing, speech recognition, and time series forecasting.
#     They are particularly useful for problems where the relationship between the input features and the output is complex and not easily described by a simple mathematical function.
# In image recognition for example, a neural network is trained on a large dataset of labeled images, such as a dataset of labeled images of handwritten digits.
#     The network learns to recognize patterns in the images, such as the shapes of the digits and the variations in their writing style.
#     Once the network is trained, it can be used to classify new images of handwritten digits with high accuracy.
#     -   In natural language processing, they can be used to understand the meaning of words and phrases, translate text from one language to another, and to generate text.
#     -   In speech recognition, they can be used to transcribe speech to text, and to recognize spoken commands.
#     -   In time series forecasting, they can be used to predict future values of a time series based on its past values, such as stock market prices, weather conditions, and energy consumption.

import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def NeuralNetwork():
    # define the sigmoid function
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    # define the derivative of the sigmoid function
    def sigmoid_derivative(x):
        return x * (1 - x)

    # define the neural network class
    class CreateNeuralNetwork:
        def __init__(self, x, y):
            self.input = x
            self.weights1 = np.random.rand(self.input.shape[1],4) 
            self.weights2 = np.random.rand(4,1)                 
            self.y = y
            self.output = np.zeros(self.y.shape)

        def feedforward(self):
            self.layer1 = sigmoid(np.dot(self.input, self.weights1))
            self.output = sigmoid(np.dot(self.layer1, self.weights2))
            return self.output

        def backprop(self):
            # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
            d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
            d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

            # update the weights with the derivative (slope) of the loss function
            self.weights1 += d_weights1
            self.weights2 += d_weights2

    X = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
    y = np.array([[0],[1],[1],[0]])
    nn = CreateNeuralNetwork(X,y)

    for i in range(1500):
        print(nn.feedforward())
        nn.backprop()
        
def NN():
    def init_params():
        W1 = np.random.rand(10, 784) - 0.5
        b1 = np.random.rand(10, 1) - 0.5
        W2 = np.random.rand(10, 10) - 0.5
        b2 = np.random.rand(10, 1) - 0.5
        return W1, b1, W2, b2

    def ReLU(Z):
        return np.maximum(Z, 0)

    def softmax(Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A
        
    def forward_prop(W1, b1, W2, b2, X):
        Z1 = W1.dot(X) + b1
        A1 = ReLU(Z1)
        Z2 = W2.dot(A1) + b2
        A2 = softmax(Z2)
        return Z1, A1, Z2, A2

    def ReLU_deriv(Z):
        return Z > 0

    def one_hot(Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
        one_hot_Y = one_hot(Y)
        dZ2 = A2 - one_hot_Y
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2)
        dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1)
        return dW1, db1, dW2, db2

    def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1    
        W2 = W2 - alpha * dW2  
        b2 = b2 - alpha * db2    
        return W1, b1, W2, b2

    def get_predictions(A2):
        return np.argmax(A2, 0)

    def get_accuracy(predictions, Y):
        print(predictions, Y)
        return np.sum(predictions == Y) / Y.size

    def gradient_descent(X, Y, alpha, iterations):
        W1, b1, W2, b2 = init_params()
        for i in range(iterations):
            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
            dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
            W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
            if i % 10 == 0:
                print("Iteration: ", i)
                predictions = get_predictions(A2)
                print(get_accuracy(predictions, Y))
        return W1, b1, W2, b2

    def make_predictions(X, W1, b1, W2, b2):
        _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
        predictions = get_predictions(A2)
        return predictions

    def test_prediction(index, W1, b1, W2, b2):
        current_image = X_train[:, index, None]
        prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
        label = Y_train[index]
        print("Prediction: ", prediction)
        print("Label: ", label)
        
        current_image = current_image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.show()

    data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data)

    data_dev = data[0:1000].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n]
    X_dev = X_dev / 255.

    data_train = data[1000:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    X_train = X_train / 255.
    _,m_train = X_train.shape

    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)
    test_prediction(0, W1, b1, W2, b2)
    test_prediction(1, W1, b1, W2, b2)
    test_prediction(2, W1, b1, W2, b2)
    test_prediction(3, W1, b1, W2, b2)

    dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
    get_accuracy(dev_predictions, Y_dev)

if __name__ == "__main__":
    NeuralNetwork()
