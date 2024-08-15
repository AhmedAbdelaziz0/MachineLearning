import numpy as np
from sklearn.linear_model import LogisticRegression as SKLearnLogisticRegression
from sklearn.metrics import accuracy_score


class MultiClassLogisticRegression:
    def __init__(
        self,
        learning_rate=[0.01],
        num_iterations=1000,
        regularization_param=0.0,
        num_classes=3,
        tolerance=1e-4,
    ):
        self.learning_rate = (
            learning_rate if isinstance(learning_rate, list) else [learning_rate]
        )
        self.num_iterations = num_iterations
        self.regularization_param = regularization_param
        self.weights = np.zeros((1, num_classes))  # dummy weight to silence the linter
        self.bias = None
        self.num_classes = num_classes
        self.tol = tolerance

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y, verbose=False):
        num_samples, num_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros((num_features, self.num_classes))
        self.bias = np.zeros((1, self.num_classes))

        # Convert y to one-hot encoding
        y_one_hot = np.eye(self.num_classes)[y]

        y_pred = self.feedforward(X)

        score = np.zeros((len(self.learning_rate), 1))
        prev_cost = np.inf
        # Gradient descent
        for i in range(self.num_iterations):
            # Compute gradients
            dW = np.dot(X.T, (y_pred - y_one_hot)) / num_samples + self.regularization_param * self.weights
            db = np.sum(y_pred - y_one_hot, axis=0, keepdims=True) / num_samples
    
            costs = []
            for alpha in self.learning_rate:
                weights = self.weights - alpha * dW
                bias = self.bias - alpha * db
                z = np.dot(X, weights) + bias
                y_pred = self.softmax(z)
                cost = -np.mean(np.sum(y_one_hot * np.log(y_pred), axis=1))
                costs.append(cost)
            idx = np.argmin(costs)
            alpha = self.learning_rate[idx]
            score[idx] += 1

            # Update weights and bias
            self.weights -= alpha * dW
            self.bias -= alpha * db

            # Feedforward
            y_pred = self.feedforward(X)

            # Compute cost
            cost = -np.mean(np.sum(y_one_hot * np.log(y_pred), axis=1))
            if abs(prev_cost - cost) < self.tol:
                print(f"Converged at iteration {i}")
                break
            prev_cost = cost

            # Print cost every 100 iterations
            if verbose and i % 100 == 0:
                cost = -np.mean(np.sum(y_one_hot * np.log(y_pred), axis=1))
                print(cost)
        print(score)

    def feedforward(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.softmax(z)
        return y_pred

    def predict(self, X):
        y_pred = self.feedforward(X)
        return np.argmax(y_pred, axis=1)


def get_test_dataset():
    from utils import load_dataset, prepare_dataset, train_test_split

    np.random.seed(10)
    X, y = load_dataset(id=53, name="iris")
    y = y.replace("Iris-setosa", 0)
    y = y.replace("Iris-versicolor", 1)
    y = y.replace("Iris-virginica", 2)
    y = y.infer_objects()

    # make y start from 0 to 27
    # y = np.unique(y, return_inverse=True)[1]
    # from pandas import DataFrame

    # y = DataFrame(y)

    X = prepare_dataset(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    return X_train, y_train, X_test, y_test


def test(X_train, y_train, X_test, y_test):
    # Train custom model
    custom_model = MultiClassLogisticRegression(
        learning_rate=[0.1, 1.0, 10.0],
        num_iterations=1000,
        regularization_param=0.0001,
        num_classes=28,
        tolerance=1e-4,
    )
    # calculate time taken
    import time

    start = time.time()

    custom_model.fit(X_train, y_train.reshape(-1))

    end = time.time()
    print(f"Time taken: {end - start} seconds")

    custom_predictions = custom_model.predict(X_test)
    custom_train_predictions = custom_model.predict(X_train)

    # Train sklearn model
    sklearn_model = SKLearnLogisticRegression(max_iter=1000)
    sklearn_model.fit(X_train, y_train.reshape(-1))
    sklearn_predictions = sklearn_model.predict(X_test)
    sklearn_train_predictions = sklearn_model.predict(X_train)

    # Compare predictions
    custom_accuracy = accuracy_score(y_test, custom_predictions)
    sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)

    print(
        f"Custom Model Train Accuracy: {accuracy_score(y_train, custom_train_predictions)}"
    )
    print(
        f"SKLearn Model Train Accuracy: {accuracy_score(y_train, sklearn_train_predictions)}"
    )

    print(f"Custom Model Accuracy: {custom_accuracy}")
    print(f"SKLearn Model Accuracy: {sklearn_accuracy}")


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_test_dataset()
    test(X_train, y_train, X_test, y_test)
