import numpy as np
from sklearn import sklearn
from sklearn.linear_model import LinearRegression

from utils import load_dataset, generate_data, plot_animation, plot_data, plot_line, prepare_dataset
from utils import train_test_split


class LinearRegressionImpl:
    def __init__(self, x, y):
        self.coefficients = np.array([])
        self.intercept = np.array([])
        self.x = x
        self.y = y
        self.theta = np.random.rand(x.shape[1] + 1, 1) * 1e-4

    def _cost_function(self, x, y, theta):
        return np.linalg.norm(x @ theta - y) ** 2 / y.shape[0]

    def _gradient_cost_function(self, x, y, theta):
        """
        x first column must be a column of ones
        """
        return (x.T @ (x @ theta - y)) / y.shape[0] / 2

    # TODO stopping condition
    def _gradient(self, alpha=0.4, iter=10000, verbose=False):
        modified_x = np.hstack((np.ones((self.x.shape[0], 1)), self.x))
        theta_values = []
        cost_values = []
        for _ in range(iter):
            self.theta = self.theta - alpha * self._gradient_cost_function(
                modified_x, self.y, self.theta
            )
            theta_values.append(self.theta.copy())
            cost_values.append(self._cost_function(modified_x, self.y, self.theta))

        if verbose:
            if self.x.shape[1] == 1:
                print(f"Line Equation: {self.theta[0][0]} + {self.theta[1][0]}x")
            # plot cost
            plot_data(np.arange(len(cost_values)), cost_values, "cost function")
            print(self.theta)

        self.coefficients = self.theta[1:, :].T
        self.intercept = self.theta[0, :]
        return np.array(theta_values)

    def _linear_algebra(self, verbose=False):
        A = np.hstack([self.x, np.ones((self.x.shape[0], 1))])
        b = self.y
        res = np.linalg.lstsq(A, b, rcond=None)[0]

        if verbose:
            print(f"Coefficients: {res.T[:,:-1]}, Intercept: {res.T[:, -1]}")
            if self.x.shape[1] == 1:
                plot_line(
                    self.x,
                    self.y,
                    res.T[:, 0],
                    res.T[:, 1],
                    line_label="My Fitted line",
                    color="b",
                )

        self.coefficients = res.T[:, :-1]
        self.intercept = res.T[:, -1]

    def fit(self, method="gradient", verbose=False, **kwargs):
        """
        method can be "gradient" or "linear_algebra"

        args is a dictionary
        {alpha: 0.4, iter: 1000}
        """
        if method == "gradient":
            return self._gradient(verbose=verbose, **kwargs)
        if method == "linear_algebra":
            return self._linear_algebra(verbose=verbose)
        else:
            raise ValueError("method must be 'gradient' or 'linear_algebra'")

    def predict(self, x):
        return np.dot(x, self.coefficients.T) + self.intercept


def sklearn_Linear_Regression(x, y, verbose=False):
    reg = LinearRegression()
    reg.fit(x, y)
    if verbose:
        print(f"Coefficients: {reg.coef_}, Intercept: {reg.intercept_}")
        print("Variance score: ", reg.score(x, y))
        if x.shape[1] == 1:
            plot_line(x, y, reg.coef_, reg.intercept_, line_label="sklearn Fitted line")
    print(f"sklearn: {np.sum(reg.predict(x) - y)}")
    return {"coefficients": reg.coef_, "intercept": reg.intercept_}


def test_random_number(dims=3, number_of_points=10):
    x, y = generate_data(dims, number_of_points)

    # SKLearn
    sk = sklearn_Linear_Regression(x, y)

    # Creating object
    lr = LinearRegressionImpl(x, y)

    # Gradient
    theta_values = lr.fit(method="gradient", alpha=0.3, iter=100000)

    print(
        "Gradient: ",
        np.allclose(lr.coefficients, sk["coefficients"], atol=1e-5)
        and np.allclose(lr.intercept, sk["intercept"], atol=1e-5),
    )
    print("prediction: ", np.sum(lr.predict(x) - y))

    if dims == 1:
        if theta_values is not None:
            plot_animation(x, y, theta_values[:, 1], theta_values[:, 0])

    print("Linear Algebra Method Linear Regression: ")
    lr.fit(method="linear_algebra")
    print(
        np.allclose(sk["coefficients"], lr.coefficients)
        and np.allclose(sk["intercept"], lr.intercept)
    )


if __name__ == "__main__":
    # test_random_number(dims=9, number_of_points=100)
    X, y = load_dataset()

    X = prepare_dataset(X)

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
    
    lr = LinearRegressionImpl(train_X.to_numpy(), train_y.to_numpy()) #.fit(train_X, train_y)
    lr.fit('linear_algebra')

    predicted_data = lr.predict(test_X.to_numpy())

    print(np.sqrt(np.sum((predicted_data - test_y.to_numpy()) ** 2)))
