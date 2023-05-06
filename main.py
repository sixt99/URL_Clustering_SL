from sklearn.datasets import make_blobs, make_moons
import matplotlib.pyplot as plt
import numpy as np

from algorithms.ALSL import ALSL
from algorithms.Leaders import Leaders
from algorithms.SL import SL
from algorithms.LSL import LSL
import math
import pandas as pd


def spiral():
    N = 400
    pi = math.pi
    theta = np.sqrt(np.random.rand(N)) * 2 * pi

    r_a = 2 * theta + pi
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
    x_a = data_a + np.random.randn(N, 2)

    r_b = -2 * theta - pi
    data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
    x_b = data_b + np.random.randn(N, 2)

    X = np.append(x_a, x_b, axis=0)
    y = np.append(np.zeros((N, 1)), np.ones((N, 1)))

    np.savetxt("data.csv", X, delimiter=",", header="x,y", comments="", fmt='%.5f')
    np.savetxt("labels.csv", y, delimiter=",", header="label", comments="", fmt='%.5f')

    return X, y


if __name__ == '__main__':
    X = np.array(pd.read_csv('data.csv'))
    y = np.array(pd.read_csv('labels.csv'))

    # Clusterize with our methods
    sixte = LSL(1.6)
    sixte.fit(X)
    y_pred = sixte.predict()

    # Plot results
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.title("Synthetic Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
