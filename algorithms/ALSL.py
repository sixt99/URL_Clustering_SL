from algorithms.Leaders import Leaders
from algorithms.SL import SL
import numpy as np
import time


def distance_matrix(mat_a, mat_b):
    return np.sqrt(np.sum(np.square(mat_a[:, np.newaxis] - mat_b[np.newaxis]), axis=2))


def distance(a, b):
    return np.sqrt(np.sum(np.square(a - b)))


def return_row_index(matrix, sample):
    return np.where(np.all(matrix == sample, axis=1))[0][0]


class ALSL:
    def __init__(self, h):
        self.X = None
        self.h = h

    def fit(self, X):
        self.X = X

    def predict(self):

        start_time = time.time()

        # Compute leaders on X
        leads = Leaders(self.h / 2)
        leads.fit(self.X)
        labels, leaders_idxs = leads.predict(return_leaders=True)
        leaders = self.X[leaders_idxs]

        print('Number of samples:', self.X.shape[0])
        print('Number of leaders:', len(leaders))

        # Compute clusters on the set of leaders
        sl = SL(self.h)
        sl.fit(leaders)
        labels_lead = sl.predict()
        num_leader_clusters = len(np.unique(labels_lead))

        print('Number of clusters of leaders:', num_leader_clusters)

        leaders_partition = [leaders[labels_lead == i] for i in range(num_leader_clusters)]

        # Start merging process
        S = {}
        for i in range(num_leader_clusters):
            for j in range(i + 1, num_leader_clusters):

                Bli = leaders_partition[i]
                Blj = leaders_partition[j]

                dist_mat = distance_matrix(Bli, Blj)
                if np.min(dist_mat) > 2 * self.h:
                    continue

                arg = np.argmin(dist_mat)
                li = Bli[int(arg / dist_mat.shape[1])]
                lj = Blj[arg % dist_mat.shape[1]]

                L_i = np.array([lx for lx in Bli if distance(lx, lj) <= 2 * self.h])
                L_j = np.array([ly for ly in Blj if distance(ly, li) <= 2 * self.h])

                S[(i, j)] = L_i, L_j

        to_merge = np.empty((0, 2), dtype=int)

        for i, j in S.keys():

            Bli = leaders_partition[i]
            Blj = leaders_partition[j]

            L_i, L_j = S[(i, j)]

            finish_loop = False
            for la in L_i:

                if finish_loop:
                    break

                la_idx = return_row_index(leaders, la)
                followers_a = self.X[labels == la_idx]

                for lb in L_j:

                    lb_idx = return_row_index(leaders, lb)
                    followers_b = self.X[labels == lb_idx]

                    matrix = distance_matrix(followers_a, followers_b)
                    if np.min(matrix) <= self.h:
                        to_merge = np.vstack([to_merge, [i, j]])
                        finish_loop = True
                        break

        # Merge labels_lead
        for i, j in to_merge:
            labels_lead[labels_lead == j] = i
            to_merge[to_merge == j] = i

        new_labels = np.zeros(labels.shape, dtype=int)
        for i in range(len(leaders)):
            new_labels[labels == i] = labels_lead[i]

        end_time = time.time()

        elapsed_time = end_time - start_time

        print('Time to train ALSL:', elapsed_time)

        return new_labels
