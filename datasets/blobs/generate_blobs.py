from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

def generate_blobs(N):
    # Create 1000 samples with 2 features (i.e. 2D data)
    X, y = make_blobs(n_samples=N, n_features=2, centers=4, cluster_std=0.7, random_state=42)
    res = np.hstack((X, y.reshape(-1, 1)))
    np.savetxt("blobs.csv", res, delimiter=",", header="x,y,label", comments="", fmt='%.5f')


if __name__ == '__main__':
    generate_blobs(50000)

