import unittest
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ml.dataset.model import *
from sklearn.model_selection import train_test_split

DATASET_HOME = '../datasets/'


class TestPredictions(unittest.TestCase):
    def test(self):
        pred = Predictor()
        X = np.load(DATASET_HOME + 'simple/X.npy')
        y = np.load(DATASET_HOME + 'simple/y.npy')

        pred.X = X
        pred.y = y
        pred.load_dicts(DATASET_HOME + 'simple/simple_dataset_dicts.pickle')

        pred.encode()

        X_train, X_test, y_train, y_test = train_test_split(pred.X, pred.y)

        pred.fit(RandomForestClassifier(n_estimators=20), X_train, y_train)

        y_pred = pred.predict(X_test)

        print(accuracy_over_axis(y_test, y_pred))
