import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from utils import load_dataset, prepare_dataset, train_test_split


def knn(X_test, X_train, y_train, p=2, k=3):
    def minkowski_distance(points, array, p):
        return np.sum(np.abs(points[:, np.newaxis, :] - array) ** p, axis=2) ** (1 / p)

    distances = minkowski_distance(X_test, X_train, p)

    # Get the indices of the k smallest distances
    k_nearest_indices = np.argsort(distances, axis=1)[:, :k]
    k_nearest_labels = y_train[k_nearest_indices]

    # Determine the most common label (mode) among the k nearest neighbors for each testing point
    predictions = np.array([np.bincount(row).argmax() for row in k_nearest_labels])

    return predictions


def sklearnKNN(point, points, labels, k=3):
    knn = KNeighborsClassifier(n_neighbors=k, algorithm="auto")
    knn.fit(points, labels)
    return knn.predict(point)


def generate_data(num_points, dim, features):
    points = np.random.random((num_points, dim))
    labels = np.random.randint(0, features, num_points)
    return points, labels


def test_random(dim, features, k, data_size, test_size):
    training_points, labels = generate_data(data_size, dim, features)
    testing_point = np.random.random((test_size, dim))
    myKnn = knn(testing_point, training_points, labels, k=k, p=2)
    skKnn = sklearnKNN(testing_point, training_points, labels, k=k)
    # print(myKnn)
    # print(skKnn)
    # print(np.sum(myKnn == skKnn) / len(myKnn))
    return np.all(myKnn == skKnn)

# micro f1
def score(prediction, labels):
    return np.sum(prediction == labels) / len(labels)

if __name__ == "__main__":
    X, y = load_dataset()
    X = prepare_dataset(X)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
    train_X = train_X.to_numpy()
    test_X = test_X.to_numpy()
    train_y = train_y.to_numpy().flatten()
    test_y = test_y.to_numpy().flatten()
    

    prediction = knn(test_X, train_X, train_y, p=2, k=3)

    print(score(prediction, test_y))
