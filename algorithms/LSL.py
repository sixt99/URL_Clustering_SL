import numpy as np

from algorithms.Leaders import Leaders
from algorithms.SL import SL


class LSL:
    def __init__(self, h):
        self.X = None
        self.h = h

    def fit(self, X):
        self.X = X

    def predict(self):
        leaders = Leaders(self.h / 2)
        leaders.fit(self.X)
        labels, leaders_idxs = leaders.predict(return_leaders=True)
        sl = SL(self.h)
        sl.fit(self.X[leaders_idxs])
        pred = sl.predict()

        updated_labels = np.empty(labels.shape)
        for i in range(len(leaders_idxs)):
            updated_labels[labels == i] = pred[i]

        return updated_labels
