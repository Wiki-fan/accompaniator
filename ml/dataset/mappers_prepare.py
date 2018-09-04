import music21

from ml.dataset.base_mapper import BaseMapper, MapperError
import numpy as np
from ml.structures import *

from ml.structures import Chord


class VarietyMapper(BaseMapper):
    """Removes songs with too many repeating chords."""

    def __init__(self, min_variety=0.5, **kwargs):
        super().__init__(**kwargs)
        self.min_variety = min_variety
        self.stats['variety'] = dict()

    def process(self, song):
        equal, non_equal = 0, 0
        for track in song.tracks:
            for i in range(len(track.chords) - 1):
                if track.chords[i] == track.chords[i + 1]:
                    equal += 1
                else:
                    non_equal += 1

        variety = non_equal / (equal + non_equal)
        self.increment_stat(variety, self.stats['variety'])
        if variety < self.min_variety:
            raise MapperError('Small variety')

        return song


class SimplifyChordsMapper(BaseMapper):

    @staticmethod
    def map_many(iterable, function, *other):
        if other:
            return ClassifyChordsMapper.map_many(map(function, iterable), *other)
        return map(function, iterable)

    @staticmethod
    def apply_many(elem, function, *other):
        if other:
            return ClassifyChordsMapper.apply_many(function(elem), *other)
        return function(elem)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.root_to_simple_chord_dict = dict()
        self.chord_to_root_dict = dict()

    def process(self, song):
        new_chords = []
        for chord in song.chord_track.chords:
            if chord.notes:
                if tuple(chord.notes) not in self.chord_to_root_dict:
                    music21_chord = chord.get_music21_repr()
                    root = music21_chord.root().name
                    simple_chord = Chord.from_music21_repr(music21.harmony.ChordSymbol(root))

                    self.chord_to_root_dict[tuple(chord.notes)] = root
                    self.root_to_simple_chord_dict[root] = simple_chord
                    chord.notes = simple_chord.notes
                    new_chords.append(chord)
                else:
                    chord.notes = self.root_to_simple_chord_dict[self.chord_to_root_dict[tuple(chord.notes)]].notes
                    new_chords.append(chord)
            else:
                new_chords.append(chord)

        song.chord_track.chords = new_chords
        return song


# TODO: this mapper changes data format dramatically. May it be better to convert it to Processor?
class ClassifyChordsMapper(BaseMapper):

    @staticmethod
    def map_many(iterable, function, *other):
        if other:
            return ClassifyChordsMapper.map_many(map(function, iterable), *other)
        return map(function, iterable)

    @staticmethod
    def apply_many(elem, function, *other):
        if other:
            return ClassifyChordsMapper.apply_many(function(elem), *other)
        return function(elem)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chord_to_root_dict = dict()
        self.reverse_chord_to_root_dict = dict()

    def process(self, song):
        melody = [0 if not chord.notes else chord.notes[0].number for chord in song.melody_track.chords]

        rhythm = []
        for chord in song.chord_track.chords:
            if chord.is_repeat:
                rhythm.append('-')
            elif chord.notes:
                if tuple(chord.notes) not in self.chord_to_root_dict:
                    music21_chord = chord.get_music21_repr()
                    root = music21_chord.root().name

                    self.chord_to_root_dict[tuple(chord.notes)] = root
                    cs = music21.harmony.ChordSymbol(root)
                    self.reverse_chord_to_root_dict[root] = tuple(Chord.from_music21_repr(cs).notes)

                rhythm.append(self.chord_to_root_dict[tuple(chord.notes)])
            else:
                rhythm.append('')

        return [[melody, rhythm]]

    def inverse_process(self, song):
        melody_track = Track([Chord([Note(number)], 128 / 8, 64) for number in song[0]])

        chord_track = []
        for symbol in song[1]:
            if symbol == '':
                chord_track.append(Chord([], 128 / 8, 64))
            elif symbol == '-':
                chord_track.append(deepcopy(chord_track[-1]))
            else:
                notes = list(self.reverse_chord_to_root_dict[symbol])
                chord_track.append(Chord(notes, 128 / 8, 64))
        chord_track = Track(chord_track)

        return Song([melody_track, chord_track])

    def dump_dicts(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.chord_to_root_dict, f)
            pickle.dump(self.reverse_chord_to_root_dict, f)

    def load_dicts(self, filename):
        with open(filename, 'rb') as f:
            self.chord_to_root_dict = pickle.load(f)
            self.reverse_chord_to_root_dict = pickle.load(f)
