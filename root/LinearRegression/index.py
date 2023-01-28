# Linear regression is a type of supervised machine learning algorithm that is used to predict a continuous 
#     outcome variable (also known as the dependent variable) based on one or more 
#     predictor variables (also known as independent variables or features).
#     The goal of linear regression is to find the best linear relationship between the predictor variables and the outcome variable.
#     It does this by finding the line (or hyperplane in the case of multiple predictor variables)
#     that minimizes the difference between the predicted values and the true values.
# The equation for a simple linear regression with one predictor variable (x) and one outcome variable (y) is:
#     y = mx + b
#     where m is the slope of the line (also known as the coefficient of x) and b is the y-intercept.
#     The goal of linear regression is to find the best values for m and b that minimize the difference between
#     the predicted values and the true values.
# Linear regression is used in a variety of applications such as:
#   •	Predictive modeling
#   •	Forecasting
#   •	Exploring the relationship between variables
#   •	Modeling the impact of changes in one variable on the other.
# Linear regression is a simple and interpretable model, its easy to implement and it can be used with a small dataset,
#     and its good for a linear relationship between the dependent and independent variables.
#     However, its not suitable for non-linear relationship, and its sensitive to outliers.

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Keep in mind that linear regression is a simple technique and it can be used only when the relationship between independent and dependent variable is linear.
def linearRegressionMathplotLib():
    # Generate some random data
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = 2 + 3 * x + np.random.rand(100, 1)

    # Fit the model
    b, a = np.linalg.lstsq(np.c_[x, np.ones(x.shape[0])], y, rcond=None)[0]

    # Plot the data and the model
    plt.scatter(x, y)
    plt.plot(x, a + b * x, 'r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


# This example uses a simple dataset with one input feature (x) and one output (y).
#     The input data is defined as a numpy array, and then reshaped using the reshape() function to ensure that it has the correct shape for the model.
#     The LinearRegression class is imported from the scikit-learn library, and a new instance of the class is created.
#     The fit() method is then used to train the model on the input data.
#     The model can then be used to make predictions on new data using the predict() method.
#     Finally, the mean squared error is calculated to evaluate the models performance on the training data.
def linearRegressionScikitLearn(predictor):
    # Input data
    x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    y = np.array([2, 4, 6, 8, 10])

    # Create a linear regression model
    model = LinearRegression()

    # Train the model on the input data
    model.fit(x, y)

    # Predict values for new data
    y_pred = model.predict([[predictor]])
    print(f'Predicted value for x={predictor}: {y_pred[0]}')

    # Evaluate the model
    mse = mean_squared_error(y, y_pred)
    print(f'Mean squared error: {mse}')

# In this example, we first generate some example data by creating a list of x values and a list of y values that are related to x by a linear relationship.
# We then calculate the mean of x and y, and use that to calculate the variance and covariance of x and y.
# Next, we use the variance and covariance to calculate the weights (w0 and w1) for the model.
# After that we use the calculated weights (w0 and w1) to predict the output (y_pred) for the input data, and finally we print the predictions.
# Its important to note that this example is for the simple linear regression case (one independent variable and one dependent variable)
#     and for larger datasets and more complex models, the implementation can be more complex and an optimization algorithm such as gradient descent should be implemented.
def linearRegression():
    # Generate some example data
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]

    # Calculate the mean of x and y
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)

    # Calculate the variance and covariance of x and y
    var_x = sum([(i - mean_x) ** 2 for i in x]) / len(x)
    cov_xy = sum([(x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))]) / len(x)

    # Calculate the weights
    w1 = cov_xy / var_x
    w0 = mean_y - w1 * mean_x

    # Predict the output for the input data
    y_pred = [w0 + w1 * i for i in x]

    # Print the predictions
    print(y_pred)


if __name__ == '__main__':
    linearRegressionMathplotLib()