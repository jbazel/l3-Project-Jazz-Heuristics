import music21.converter
from music21 import *
import sqlite3
from chord_corpus_builder import buildCorpus
from chord_interval_corpus import build_chord_interval_corpus
import matplotlib.pyplot as plt
import numpy as np

us = environment.UserSettings()
us['musicxmlPath'] = '../../../../../Applications/MuseScore 3.app'
us['midiPath'] = '../../../../../Applications/GarageBand.app'
db = sqlite3.connect('EWLD/EWLD.db')
cursor = db.cursor()
cursor.execute('SELECT DISTINCT path_leadsheet FROM works, work_genres WHERE work_genres.genre = "Jazz"')
paths = cursor.fetchall()

NOTES = {
    "C--": -1, "C-": 0, "C": 1, "C#": 2, "C##": 3,
    "D--": 1, "D-": 2, "D": 3, "D#": 4, "D##": 5,
    "E--": 3, "E-": 4, "E": 5, "E#": 6, "E##": 7,
    "F--": 4, "F-": 5, "F": 6, "F#": 7, "F##": 8,
    "G--": 6, "G-": 7, "G": 8, "G#": 9, "G##": 10,
    "A--": 8, "A-": 9, "A": 10, "A#": 11, "A##": 12,
    "B--": 10, "B-": 11, "B": 12, "B#": 1, "B##": 2,
}

CHORD_EQUIVELANCE_DICT = dict()


class Leadsheet:

    class WeightedNote:
        def __init__(self, pitch, pitch_stability, interval_salience, beat_strength, durational_accent):
            self.pitch = pitch
            self.pitch_stability = pitch_stability
            self.interval_salience = interval_salience
            self.beat_strength = beat_strength
            self.durational_accent = durational_accent
            self.weight = self.pitch_stability + self.interval_salience + self.beat_strength + self.durational_accent

    def __init__(self, source):
        self.source = source
        self.score_parsed = music21.converter.parse(source)
        self.chords = []
        self.chord_profiles = []
        self.chord_intervals = []
        self.chord_common_names = []
        self.chord_pitched_names = []
        self.melodies_intervals = []
        self.melodies_exact = []
        self.weighted_melodies = []
        self.melodies = []

    def lyric_removal(self):
        for n in self.score_parsed.recurse().notesAndRests:
            if n.lyric:
                n.lyric = ""

    def chord_feature_extraction(self):
        profiles = []
        intervals = []
        for i in self.chords:
            profile = [0] * 12
            interval = [0] * 12
            normal_order = i.normalOrder
            for j in normal_order:
                profile[j] = 1

            first_pitch = normal_order[0]
            rotated_normal = [(pc - first_pitch) % 12 for pc in normal_order]
            for j in rotated_normal:
                interval[j] = 1

            profiles.append(profile)
            intervals.append(interval)

        self.chord_intervals = intervals[1:]
        self.chord_profiles = profiles[1:]
        self.chord_common_names = [chord.commonName for chord in self.chords]
        self.chord_pitched_names = [chord.pitchedCommonName for chord in self.chords]

    def chord_melody_separation(self):
        score = [n for n in self.score_parsed.recurse().notes]
        separation = [x for x in range(len(score)) if type(score[x]) == music21.harmony.ChordSymbol]
        self.chords = [score[i] for i in separation]
        separation.append(len(score))
        for i in range(len(separation) - 1):
            self.melodies.append(score[separation[i] + 1:separation[i + 1]])

    def melody_pitch_encoding(self):
        exact = []
        for melody in self.melodies:
            pitches = []
            for i in range(len(melody)):
                note = str(melody[i].pitch)
                note_separated = [note[:-1], note[-1]]
                octave = int(note_separated[1]) - 3
                note_value = NOTES[note_separated[0]] + (12 * octave)
                pitches.append(note_value)
            exact.append(pitches)
        self.melodies_exact = exact

    def melody_interval_encoding(self):
        relative = []
        for melody in self.melodies_exact:
            temp_interval = []
            for i in range(len(melody) - 1):
                temp_interval.append(melody[i+1] - melody[i])
            relative.append(temp_interval)
        self.melodies_intervals = relative

    def weighted_pitch_encoding(self):
        pitch_weights = []
        for melody_index, melody in enumerate(self.melodies):
            note_dict = dict()
            total_notes = 0
            weighted_melody = []
            # created pitch class vector
            for note_index, note in enumerate(melody):
                pitch = str(note.pitch)
                if pitch in note_dict:
                    note_dict[pitch] += 1
                else:
                    note_dict[pitch] = 1
                total_notes += 1

            # scale it for number of notes
            for key in note_dict:
                note_dict[key] /= total_notes

            for note in melody:
                key = str(note.pitch)
                weighted_melody.append(note_dict[key] * note.duration.quarterLength)

            pitch_weights.append(weighted_melody)

        self.weighted_melodies = pitch_weights





    def melody_reduction(self):
        for melody_index, melody in enumerate(self.melodies):
            for note_index, note in enumerate(melody):

                # get interval salience
                if note_index == 0:
                    interval_salience = 0

                else:
                    interval_salience = self.melodies_intervals[melody_index][note_index - 1]

                pitch_stability = self.weighted_melodies[melody_index][note_index]
                beat_strength = note.beatStrength
                durational_accent = note.duration.quarterLength
                weighted = self.WeightedNote(note.pitch, interval_salience, pitch_stability, beat_strength, durational_accent)
                print(weighted.weight)




# def chord_progression_extractor(progression, key):
#     # return [roman.romanNumeralFromChord(i, key).figure for i in progression]
#     return [roman.romanNumeralFromChord(i, key).romanNumeralAlone for i in progression]

# def get_key(sheet):
#     for i in (sheet.score_parsed.recurse()):
#         if type(i) == music21.key.KeySignature:
#             print(i)

def stringify(val):
    return ','.join([str(x) for x in val])


def flatten(arr):
    return [i for j in arr for i in j]


def update_corpus(corp, key):
    if key in corp:
        corp[key] += 1
    else:
        corp[key] = 1
    return corp


def build_weighted_note_corpus():
    pass

def build_duo_corpi(c1, m1, c2, m2):

    """
    METHOD 1:

    this corpus considers simple relation to chords and melodies, both pitched and un-pitched

    for optimal use, pattern matching must be applied as exact matches are unlikely

    takes in 2 pairs of chord-melody arrays and builds a 2D dictionary of melody occurrences
    in a both pitched and un-pitched representation.

    function will return the respective corpi in the order in which the pairs are passed
    into the function
    """

    # for each of chord; build corpus of melodies, then combine to make multi-layerd dict.
    # iteration only necessary for either c1, m1 as mapped directly to c2, m2
    c1_lookup = dict()
    c2_lookup = dict()
    corp1 = []
    corp2 = []

    for index, chord in enumerate(c1):
        chord = stringify(chord)

        # if chord currently has a corpus built from it; find from lookup table
        if chord in c1_lookup:
            lookup_index = c1_lookup[chord]
            m1_key = stringify(m1[index])
            m2_key = stringify(m2[index])

            corp1[lookup_index] = update_corpus(corp1[lookup_index], m1_key)
            corp2[lookup_index] = update_corpus(corp2[lookup_index], m2_key)

        # if chord does not currently have a corpus; create corp
        else:

            # create corp
            n = len(c1_lookup)
            c1_lookup[chord] = n
            c2_lookup[stringify(c2[index])] = n
            corp1.append(dict())
            corp2.append(dict())

            m1_key = stringify(m1[index])
            m2_key = stringify(m2[index])

            corp1[n] = update_corpus(corp1[n], m1_key)
            corp2[n] = update_corpus(corp2[n], m2_key)

    c1_corpus = dict()
    c2_corpus = dict()
    for index, key in enumerate(c1_lookup):
        c1_corpus[key] = corp1[index]

    for index, key in enumerate(c2_lookup):
        c2_corpus[key] = corp2[index]

    return c1_corpus, c2_corpus


leadsheets = []
chord_dict = {}

chord_intervals = []
chord_profiles = []
melody_intervals = []
melody_exact = []


# main processing section
for i in range(10):
    path = paths[i]
    # print('parsing ', paths)
    path = "EWLD/" + path[0]
    sheet = Leadsheet(path)
    sheet.chord_melody_separation()
    sheet.chord_feature_extraction()
    sheet.melody_pitch_encoding()
    sheet.melody_interval_encoding()

    sheet.weighted_pitch_encoding()
    sheet.melody_reduction()

    chord_intervals.append(sheet.chord_intervals)
    chord_profiles.append(sheet.chord_profiles)

    melody_intervals.append(sheet.melodies_intervals)
    melody_exact.append(sheet.melodies_exact)

chord_intervals = flatten(chord_intervals)
chord_profiles = flatten(chord_profiles)
melody_intervals = flatten(melody_intervals)
melody_exact = flatten(melody_exact)

print(build_duo_corpi(chord_intervals, melody_intervals, chord_profiles, melody_exact))

# THIS SECTION FOR VISUALIZING A MUDDY DATA FIELD WITH NON-USEFUL CHORD SYMBOLS
# plt.bar(list(corp.keys()), list(corp.values()))
# plt.xticks(list(corp.keys()), rotation=90)
# plt.show()


# # chord removal
# for n in c.recurse().getElementsByClass(chord.Chord):
#     meas = n.getContextByClass(stream.Measure)
#     print(meas.measureNumber, meas.offset, n.getOffsetInHierarchy(c), n.offset)
#     meas.remove(n)


# c.show()


# db = "/OpenEWLD-master/OpenEWLD.db"
# db = sqlite3.connect(db)
