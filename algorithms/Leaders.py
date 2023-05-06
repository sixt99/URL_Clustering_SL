import numpy as np


# Function to calculate the distance between two patterns
def distance(x, y):
    return np.sqrt(np.sum(np.square(x - y)))


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
        leaders_idxs = []

        # Loop over each pattern in X
        for i, x in enumerate(self.X):
            is_follower = False
            # Loop over each leader index in the leaders_idxs list
            for j, leader_idx in enumerate(leaders_idxs):
                # Check if the distance between the current pattern and the leader is within the threshold
                if distance(self.X[leader_idx], x) <= self.tau:
                    # If the current pattern has a leader, assign it to the corresponding cluster
                    labels[i] = j
                    is_follower = True
                    break

            # If the current pattern has no leader, create a new leader and assign it to a new cluster
            if not is_follower:
                leaders_idxs.append(i)
                labels[i] = len(leaders_idxs) - 1

        if return_leaders:
            # Return the cluster labels for each pattern and the index of each leader
            return labels, leaders_idxs

        else:
            # Return the cluster labels for each pattern
            return labels
