import numpy as np
from sklearn.preprocessing import LabelEncoder
from ml.dataset.mappers_prepare import *


def equal_over_axis(y_true, y_pred, axis=1):
    def process(row):
        true_row = row[:y_true.shape[1]]
        pred_row = row[y_true.shape[1]:]
        return np.array_equal(true_row, pred_row)

    return np.apply_along_axis(process, axis, np.hstack([y_true, y_pred]))


def accuracy_over_axis(y_test, y_pred, axis=1):
    return sum(equal_over_axis(y_test, y_pred, 1)) / y_pred.shape[np.abs(1 - axis)]


class BasePredictor:
    def __init__(self):
        pass

    def encode(self):
        raise NotImplementedError()

    def fit(self, clf, X_train, y_train):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()

    def load_dicts(self, filename):
        self.ccm = ClassifyChordsMapper()
        self.ccm.load_dicts(filename)

    def predict_song(self, song_c, preview=1, measure_length=8):
        raise NotImplementedError()

    def dump_clf(self, filename):
        with open(filename, 'wb') as fid:
            pickle.dump(self.clf, fid)
