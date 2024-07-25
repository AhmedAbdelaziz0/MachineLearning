import numpy as np
from utils import generate_data, plot_animation, plot_data
from LA_LinearRegression import sklearn_Linear_Regression


def cost_function(x, y, theta):
    return np.linalg.norm(x @ theta - y) ** 2 / y.shape[0]


def gradient_cost_function(x, y, theta):
    """
    x first column must be a column of ones
    """
    return 2 * x.T @ ((x @ theta) - y) / y.shape[0]


def myLinearRegression(x, y):
    x = np.hstack((np.ones((x.shape[0], 1)), x))
    print(x.shape)
    theta = np.zeros((x.shape[1], 1))
    alpha = 0.4
    theta_values = []
    cost_values = []
    for _ in range(100):
        theta = theta - alpha * gradient_cost_function(x, y, theta)
        theta_values.append(theta.copy())
        cost_values.append(cost_function(x, y, theta))
    print(f"Line Equation: {theta[0][0]} + {theta[1][0]}x")
    # plot cost
    plot_data(np.arange(len(cost_values)), cost_values, 'cost function')
    return theta, np.array(theta_values)

if __name__ == "__main__":
    x, y = generate_data()
    sklearn_Linear_Regression(x, y)
    theta, theta_values = myLinearRegression(x, y)
    print(theta)
    plot_animation(x, y, theta_values[:, 1], theta_values[:, 0])
