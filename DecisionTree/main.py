from sklearn import tree
from sklearn.model_selection import train_test_split
from utils import load_dataset, prepare_dataset
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


# # load dataset
# X, y = load_dataset(id=53, name="iris")
# y = y.replace("Iris-setosa", 0)
# y = y.replace("Iris-versicolor", 1)
# y = y.replace("Iris-virginica", 2)
# y = y.infer_objects()
#
# # one hot encode
# X = prepare_dataset(X)
# # print numpber of features
#
# # split dataset
# train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)
#
# # train decision tree
# clf = tree.DecisionTreeClassifier(max_depth=50, min_samples_leaf=10, random_state=42)
# clf.fit(train_X, train_y)
#
# # evaluate
# print(f"Accuracy test: {clf.score(test_X, test_y)}")
# print(f"Accuracy train: {clf.score(train_X, train_y)}")


import numpy as np


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def entropy(self, y):
        unique_labels, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum([p * np.log2(p) for p in probabilities])
        return entropy

    def information_gain(self, X_column, y, threshold):
        parent_entropy = self.entropy(y)
        left_idxs, right_idxs = self.split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        e_left, e_right = self.entropy(y[left_idxs]), self.entropy(y[right_idxs])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

        ig = parent_entropy - child_entropy
        return ig

    def best_split(self, X, y):
        best_gain = -1
        split_idx, split_thresh = None, None

        for i in range(X.shape[1]):
            X_column = X[:, i]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self.information_gain(X_column, y, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = i
                    split_thresh = threshold

        return split_idx, split_thresh

    def split(self, X_column, threshold):
        left_idxs = np.where(X_column <= threshold)[0]
        right_idxs = np.where(X_column > threshold)[0]
        return left_idxs, right_idxs

    def build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        unique_labels = np.unique(y)

        if len(unique_labels) == 1:
            return unique_labels[0]

        if self.max_depth and depth >= self.max_depth:
            return np.bincount(y).argmax()

        split_idx, split_thresh = self.best_split(X, y)

        if split_idx is None:
            return np.bincount(y).argmax()

        left_idxs, right_idxs = self.split(X[:, split_idx], split_thresh)
        left = self.build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self.build_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return {
            "feature_index": split_idx,
            "threshold": split_thresh,
            "left": left,
            "right": right,
        }

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict_sample(self, x, tree):
        if not isinstance(tree, dict):
            return tree

        feature_index = tree["feature_index"]
        threshold = tree["threshold"]

        if x[feature_index] <= threshold:
            return self.predict_sample(x, tree["left"])
        else:
            return self.predict_sample(x, tree["right"])

    def predict(self, X):
        return np.array([self.predict_sample(x, self.tree) for x in X])


# Example usage
if __name__ == "__main__":
    # Convert the data dictionary to a NumPy array (X)
    X, y = load_dataset(id=53, name="iris")
    X = X.infer_objects()
    X = X.to_numpy()
    pd.set_option("future.no_silent_downcasting", True)

    # Convert the target dictionary to a NumPy array (y)
    y = y.replace("Iris-setosa", 0)
    y = y.replace("Iris-versicolor", 1)
    y = y.replace("Iris-virginica", 2)
    y = y.infer_objects().to_numpy().reshape(-1)

    # Example usage with the DecisionTree class
    clf = DecisionTree(max_depth=3)
    clf.fit(X, y)

    # Make predictions
    predictions = clf.predict(X)
    print("Predictions:", predictions)
    print("Accuracy:", accuracy_score(y, predictions))
