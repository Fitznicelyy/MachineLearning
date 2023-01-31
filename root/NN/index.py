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

if __name__ == "__main__":
    NeuralNetwork()
