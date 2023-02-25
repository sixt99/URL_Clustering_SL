from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

from algorithms.clustering import Leaders

if __name__ == '__main__':
    # Set the number of clusters and the number of features for the synthetic data
    n_samples = 1000
    n_features = 2
    n_clusters = 3

    # Generate synthetic data using the make_blobs function
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)

    # Clusterize with our methods
    leaders = Leaders(5)
    y_pred = leaders.predict(X)

    # Plot results
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.title("Synthetic Data with {} Clusters".format(n_clusters))
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
