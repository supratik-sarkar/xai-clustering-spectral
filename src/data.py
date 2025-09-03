import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

def load_mnist(sample_size=None, random_state=42):
    """
    Returns:
      X: (N, 784) float32 normalized [0,1]
      y: (N,) int labels
      images: (N, 28, 28) uint8 images (0-255)
    """
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist["data"].astype("float32") / 255.0
    y = mnist["target"].astype(int)
    if sample_size is not None and sample_size < X.shape[0]:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(X.shape[0], size=sample_size, replace=False)
        X = X[idx]
        y = y[idx]
    images = (X.reshape(-1, 28, 28) * 255).astype("uint8")
    return X, y, images

def pca_reduce(X, n_components=50, random_state=0):
    pca = PCA(n_components=n_components, random_state=random_state)
    Z = pca.fit_transform(X)
    return Z, pca
