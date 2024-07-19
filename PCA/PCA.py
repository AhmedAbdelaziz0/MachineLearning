import numpy as np
from sklearn.decomposition import PCA


def myPCA(data, number_of_components=2):
    X = np.array(data)

    # standardized
    X_demean = X - np.mean(X, axis=0)

    # eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(X_demean.T @ X_demean)
    eigenvectors = np.real(eigenvectors)

    idx = np.argsort(eigenvalues)[::-1]
    idx = idx[:number_of_components]
    sorted_eigenvectors_subset = eigenvectors[:, idx]

    # PCA
    X_reduced = np.dot(X_demean, sorted_eigenvectors_subset)
    return X_reduced


def sklearn_PCA(data, number_of_components):
    pca = PCA(n_components=number_of_components)
    pca.fit(data)
    return pca.transform(data)


def cosine_distance(A, B):
    A = A.T
    B = B.T
    dot_product = np.sum(A * B, axis=1)
    norm_A = np.linalg.norm(A, axis=1)
    norm_B = np.linalg.norm(B, axis=1)
    cosine_sim = dot_product / (norm_A * norm_B)
    cosine_dist = np.average(np.abs(cosine_sim))
    return cosine_dist


if __name__ == "__main__":
    data = np.random.rand(20, 100)
    T = []
    for components in range(1, data.shape[0]):
        sk = np.array(sklearn_PCA(data, components))
        me = np.array(myPCA(data, components))
        print(f"Cosine distance {components}: {cosine_distance(sk, me)}")
