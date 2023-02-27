from sklearn.datasets import make_blobs, make_moons
import matplotlib.pyplot as plt

from algorithms.leaders import Leaders
from algorithms.SL import SL

if __name__ == '__main__':

    # Generate synthetic data using the make_blobs function
    X, y = make_blobs(n_samples=1000, centers = 3)

    # Clusterize with our methods
    SL = SL(1)
    y_pred = SL.predict(X)

    # Plot results
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.title("Synthetic Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
