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
        self.clf = clf
        self.clf.fit(X_train, y_train)

    def predict(self, X):
        return self.clf.predict(X).astype(np.int64)

    def load_dicts(self, filename):
        self.ccm = ClassifyChordsMapper()
        self.ccm.load_dicts(filename)

    def predict_song(self, song_c, preview=1, measure_length=8):
        song = self.ccm.process(song_c)[0]

        predicted = ['']*(measure_length*preview)
        for i in range(preview, len(song[0]) // measure_length):
            # melody history + chords history + current melody
            x = np.array(song[0][(i - preview) * measure_length:i * measure_length] + \
                         song[1][(i - preview) * measure_length:i * measure_length])
            # current chords
            y = np.array(song[1][i * measure_length:(i + 1) * measure_length])
            x_cat = x[self.X_cat_mask]
            x_num = x[self.X_num_mask]
            for i in range(sum(self.X_cat_mask)):
                x_cat[i] = self.enc.transform([x_cat[i]])[0]
                y[i] = self.enc.transform([y[i]])[0]
            x = np.hstack([x_num, x_cat])

            y_pred = self.clf.predict([x]).astype(np.int64)[0]
            predicted += list(self.enc.inverse_transform(y_pred))

        predicted_song = self.ccm.inverse_process([[], predicted])
        predicted_song.tracks[0] = deepcopy(song_c.tracks[0])
        return predicted_song

    def dump_clf(self, filename):
        with open(filename, 'wb') as fid:
            pickle.dump(self.clf, fid)
