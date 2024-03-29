import numpy as np
import time

# Function to calculate the distance between two patterns
def distance(x, y):
    return np.sqrt(np.sum(np.square(x - y)))


def distance_matrix(x, mat):
    return np.sqrt(np.sum((mat - x) ** 2, axis=1))


# Class implementing the Leaders clustering method
class Leaders:
    def __init__(self, tau):
        self.X = None
        self.tau = tau  # Threshold distance for leader identification

    def fit(self, X):
        self.X = X

    def predict(self, return_leaders=False):

        start_time = time.time()

        # Initialize an array to hold cluster labels for each pattern
        labels = np.empty((np.shape(self.X)[0],))
        # Initialize a list to hold the indices of the leaders in the X array
        leaders_idxs = []

        # Loop over each pattern in X
        for i, x in enumerate(self.X):

            # Compute the distance between the current pattern and all the leaders
            distances = distance_matrix(x, self.X[leaders_idxs])
            small_distances = np.where(distances <= self.tau)[0]
            if len(distances) > 0 and len(small_distances) > 0:
                labels[i] = small_distances[0]
            else:
                leaders_idxs.append(i)
                labels[i] = len(leaders_idxs) - 1

        end_time = time.time()

        print('Time to predict Leaders:', end_time - start_time)


        if return_leaders:
            # Return the cluster labels for each pattern and the index of each leader
            return labels, leaders_idxs

        else:
            # Return the cluster labels for each pattern
            return labels
