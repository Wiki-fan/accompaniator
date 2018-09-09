from ml.dataset.base_processor import BaseProcessor
import numpy as np

from ml.dataset.corpus import get_progressbar


class HistoryDatasetProcessor(BaseProcessor):
    def process(self, songs, preview=1, measure_length=8):
        X = []
        Y = []

        for song in songs:
            for i in range(preview, len(song[0]) // measure_length - 1):
                # melody history + chords history + current melody
                x = [song[0][(i - preview) * measure_length:i * measure_length],
                     song[1][(i - preview) * measure_length:i * measure_length]]
                # current chords
                y = song[1][i * measure_length:(i + 1) * measure_length]
                X.append(x)
                Y.append(y)
        return np.array(X, dtype=object), np.array(Y, dtype=object)
