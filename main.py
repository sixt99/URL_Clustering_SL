from sklearn.datasets import make_blobs, make_moons
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

from algorithms.ALSL import ALSL
from algorithms.Leaders import Leaders
from algorithms.Rand import rand
from algorithms.SL import SL
from algorithms.LSL import LSL
import math
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

from sklearn.cluster import KMeans
# import sklearn.decomposition.PCA

if __name__ == '__main__':
    # Load dataset
    dataset = ['spiral', 'banana', 'dna', 'shuttle', 'blobs'][0]
    df = pd.read_csv(f'datasets/{dataset}/{dataset}.csv')
    X = df.values[:, :-1]

    # Clusterize with our methods
    algorithm = LSL(1.8)
    algorithm.fit(X)
    y_pred = algorithm.predict()

    print('Clusters LSL:', np.unique(y_pred))
    print('Number of clusters:', len(np.unique(y_pred)))
    print('Number of points per cluster:', [len(y_pred[y_pred == x]) for x in np.unique(y_pred)])
    print('')

    # Clusterize with our methods (second time)
    algorithm = ALSL(1.7)
    algorithm.fit(X)
    y_pred2 = algorithm.predict()

    print('Clusters ALSL:', np.unique(y_pred2))
    print('Number of clusters:', len(np.unique(y_pred2)))
    print('Number of points per cluster:', [len(y_pred2[y_pred2 == x]) for x in np.unique(y_pred2)])
    print('')

    print('rand_index of both clusterings:', rand(y_pred, y_pred2))

    do_pca = False
    if do_pca:
        print('PCA...')
        pca = PCA(n_components=2)
        pca.fit(X)
        X = pca.transform(X)

    do_pca_3d = False
    if do_pca_3d:
        print('PCA...')
        pca = PCA(n_components=3)
        pca.fit(X)
        X = pca.transform(X)

        # Create a 3D scatter plot of the PCA-transformed data
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], y=y_pred)
        plt.show()
    else:
        # Create a subplot with two plots
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the first clusterization result
        axs[0].scatter(X[:, 0], X[:, 1], c=y_pred, s=3)
        axs[0].set_title('y_pred')

        # Plot the second clusterization result
        axs[1].scatter(X[:, 0], X[:, 1], c=y_pred2, s=3)
        axs[1].set_title('y_pred2')

        # Show the plot
        plt.show()
