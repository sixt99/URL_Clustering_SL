import numpy as np

import time


# Define a function to compute the pairwise distance matrix between a set of samples and itself
def samples_distance_matrix(mat):
    return np.sqrt(np.sum(np.square(mat[:, np.newaxis] - mat[np.newaxis]), axis=2))


# Define a function to find the argmin of a matrix, but only considering positive elements
def positive_argmin(mat):
    # Create a boolean mask for the positive elements
    mask = mat > 0

    # Find the argmin of the non-zero elements
    argmin = np.argmin(mat[mask])

    # Adjust the argmin to account for the indices of the zero elements
    indices = np.arange(mat.size).reshape(mat.shape)
    argmin_adjusted = indices[mask][argmin]

    # Convert the adjusted index to matrix coordinates
    argmin_row, argmin_col = np.unravel_index(argmin_adjusted, mat.shape)

    return argmin_row, argmin_col


class SL:
    # Initialize the single linkage clustering object with a threshold distance
    def __init__(self, h):
        self.X = None
        self.h = h  # Threshold distance for leader identification

    def fit(self, X):
        self.X = X

    # Perform single linkage clustering on a set of samples
    def predict(self, return_time=False):

        start_time = time.time()

        # Initialize the cluster labels to be the indices of the samples
        labels = np.arange(np.shape(self.X)[0])

        # Compute the pairwise distance matrix between the samples
        distance_matrix = samples_distance_matrix(self.X)

        # Perform single linkage clustering until the threshold distance is reached
        while True:

            if not np.any(distance_matrix > 0):
                print('SL came across a null distance matrix. Cannot move forward.')
                exit()

            # Find the minimum distance between any two samples
            min_dist = np.min(distance_matrix[distance_matrix > 0])

            # If the minimum distance is greater than or equal to the threshold distance, stop clustering
            if min_dist >= self.h:
                break

            # Find the indices of the closest pair of samples
            min_idx = positive_argmin(distance_matrix)

            # Merge the two closest clusters into a single cluster
            distance_matrix[min_idx[0]] = np.minimum(distance_matrix[min_idx[0]], distance_matrix[min_idx[1]])
            distance_matrix[:, min_idx[0]] = np.minimum(distance_matrix[:, min_idx[0]], distance_matrix[:, min_idx[1]])
            distance_matrix = np.delete(distance_matrix, min_idx[1], axis=0)
            distance_matrix = np.delete(distance_matrix, min_idx[1], axis=1)

            # Update the cluster labels
            labels[labels == min_idx[1]] = min_idx[0]
            labels[labels > min_idx[1]] -= 1

        end_time = time.time()

        elapsed_time = end_time - start_time

        # print('Time to predict SL:', elapsed_time)

        if return_time:
            return labels, elapsed_time

        # Return the final cluster labels
        return labels
