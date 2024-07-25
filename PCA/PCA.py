import numpy as np
from sklearn.decomposition import PCA


def myPCA(data, number_of_components=2):
    X = np.array(data)

    # standardized
    X_zero_mean = X - np.mean(X, axis=0)

    # Covariance matrix
    covariance = (X_zero_mean.T @ X_zero_mean) / (X.shape[0] - 1)
    # covariance = np.cov(X_zero_mean)

    # eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(covariance)
    eigenvectors = np.real(eigenvectors)

    # explained_variance = eigenvalues / np.sum(eigenvalues)
    # print("Explained Variance:", explained_variance)

    idx = np.argsort(eigenvalues)[::-1]
    idx = idx[:number_of_components]
    sorted_eigenvectors_subset = eigenvectors[:, idx]

    # PCA
    X_reduced = np.dot(X_zero_mean, sorted_eigenvectors_subset)
    return X_reduced


def sklearn_PCA(data, number_of_components):
    pca = PCA(n_components=number_of_components)
    pca.fit(data)
    return pca.transform(data)


def cosine_distance(A, B):
    """
    Cosine distance between two matrices column vector by column vector.
    """
    A = A.T
    B = B.T
    dot_product = np.sum(A * B, axis=1)
    norm_A = np.linalg.norm(A, axis=1)
    norm_B = np.linalg.norm(B, axis=1)
    cosine_sim = dot_product / (norm_A * norm_B)
    cosine_dist = np.average(np.abs(cosine_sim))
    return cosine_dist


if __name__ == "__main__":
    data = np.random.rand(3, 4)
    T = []
    for components in range(1, data.shape[0]):
        sk = np.array(sklearn_PCA(data, components))
        me = myPCA(data, components)
        # print(f"Cosine distance {components}: {cosine_distance(sk, me)}")
        print(np.all(np.logical_or(np.isclose(sk, me), np.isclose(sk, -me))))
