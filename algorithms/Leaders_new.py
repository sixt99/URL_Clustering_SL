import numpy as np


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

        # Initialize an array to hold cluster labels for each pattern
        labels = np.empty((np.shape(self.X)[0],))
        # Initialize a list to hold the indices of the leaders in the X array
        leaders_idxs = np.empty(0, dtype=int)

        # Loop over each pattern in X
        for i, x in enumerate(self.X):
            print(i)

            arr = distance_matrix(x, self.X[leaders_idxs])
            print(arr)
            if leaders_idxs.size != 0 and np.min(arr) <= self.tau:
                labels[i] = np.argmin(arr)

            # If the current pattern has no leader, create a new leader and assign it to a new cluster
            else:
                leaders_idxs = np.concatenate((leaders_idxs, np.array([i])))
                labels[i] = len(leaders_idxs) - 1

        if return_leaders:
            # Return the cluster labels for each pattern and the index of each leader
            return labels, leaders_idxs

        else:
            # Return the cluster labels for each pattern
            return labels
