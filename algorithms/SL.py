import numpy as np


# Define a function to compute the pairwise distance matrix between two sets of samples
def samples_distance_matrix(A, B):
    return np.sqrt(np.sum(np.square(A[:, np.newaxis] - B[np.newaxis]), axis=2))


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
        self.h = h  # Threshold distance for leader identification

    # Perform single linkage clustering on a set of samples
    def predict(self, X):
        # Initialize the cluster labels to be the indices of the samples
        labels = np.arange(np.shape(X)[0])

        # Compute the pairwise distance matrix between the samples
        distance_matrix = samples_distance_matrix(X, X)

        # Perform single linkage clustering until the threshold distance is reached
        while True:
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

        # Return the final cluster labels
        return labels
