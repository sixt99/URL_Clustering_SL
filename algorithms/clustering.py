import numpy as np


# Function to calculate the distance between two patterns
def distance(x, y):
    return np.linalg.norm(y - x)


# Class implementing the Leaders clustering method
class Leaders:
    def __init__(self, tau):
        self.tau = tau  # Threshold distance for leader identification

    def predict(self, X):

        # Initialize an array to hold cluster labels for each pattern
        labels = np.empty((np.shape(X)[0],))
        # Initialize a list to hold the indices of the leaders in the X array
        leaders_idxs = []

        # Loop over each pattern in X
        for i, x in enumerate(X):
            is_follower = False
            # Loop over each leader index in the leaders_idxs list
            for j, leader_idx in enumerate(leaders_idxs):
                # Check if the distance between the current pattern and the leader is within the threshold
                if distance(X[leader_idx], x) <= self.tau:
                    # If the current pattern has a leader, assign it to the corresponding cluster
                    labels[i] = j
                    is_follower = True
                    break

            # If the current pattern has no leader, create a new leader and assign it to a new cluster
            if not is_follower:
                leaders_idxs.append(i)
                labels[i] = len(leaders_idxs) - 1

        # Return the cluster labels for each pattern
        return labels
