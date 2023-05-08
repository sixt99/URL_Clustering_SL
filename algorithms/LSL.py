import numpy as np

from algorithms.Leaders import Leaders
from algorithms.SL import SL
import time

class LSL:
    def __init__(self, h):
        self.X = None
        self.h = h

    def fit(self, X):
        self.X = X

    def predict(self, return_time=False):

        start_time = time.time()

        leaders = Leaders(self.h / 2)
        leaders.fit(self.X)
        labels, leaders_idxs = leaders.predict(return_leaders=True)
        sl = SL(self.h)
        sl.fit(self.X[leaders_idxs])
        pred = sl.predict()

        updated_labels = np.empty(labels.shape, dtype=int)
        for i in range(len(leaders_idxs)):
            updated_labels[labels == i] = pred[i]

        end_time = time.time()

        # print('Time to predict LSL:', end_time - start_time)

        if return_time:
            return labels, end_time - start_time

        return updated_labels
