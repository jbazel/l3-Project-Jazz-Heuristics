import music21.converter
from music21 import *
import sqlite3
from chord_corpus_builder import build_duo_corp
from chord_corpus_builder import *
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
    def __init__(self, source):
        self.source = source
        self.score_parsed = music21.converter.parse(source)
        self.chords = []
        self.chord_pitch_class = []
        self.chord_profile = []
        self.chord_common_names = []
        self.chord_pitched_names = []
        self.melodies_intervals = []
        self.melodies_pitched = []
        self.pitch_weights = []
        self.weighted_melodies = []
        self.reduced_melodies = []
        self.normalized_reduction = []
        self.reduced_stream = None
        self.melodies = []

    # --------------------------------------------------
    def lyric_removal(self):
        for n in self.score_parsed.recurse().notesAndRests:
            if n.lyric:
                n.lyric = ""

    # --------------------------------------------------
    # --------------------------------------------------
    def chord_melody_separation(self):

        """
        Chord Melody Separation:

        Works by recursing through all "notes" within the score as
        both traditional notes and chord symbols are categorized as this.

        By marking chord location by type-checking, the chords and melody
        can be separated.
        """

        score = [n for n in self.score_parsed.recurse().notes]
        separation = [x for x in range(len(score)) if type(score[x]) == music21.harmony.ChordSymbol]
        self.chords = [score[i] for i in separation]
        separation.append(len(score))
        for i in range(len(separation) - 1):
            self.melodies.append(score[separation[i] + 1:separation[i + 1]])

    # --------------------------------------------------
    # --------------------------------------------------
    def chord_feature_extraction(self):
        """

        Extracts features from chord:

        - Pitch Class(represented as a 12-vector of pitches in 12TET)
        - Profile (pitch class however rotated (transposed), be in key C)
        - Common names (chord type name from Music21)
        - Pitched name (pitched chord name from Music21)
        :return:
        """
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

            print(rotated_normal, i.pitchedCommonName, i.commonName)


            for j in rotated_normal:
                interval[j] = 1
            # need justification on this method used
            profiles.append(profile)
            intervals.append(interval)

        self.chord_profile = intervals[1:]
        self.chord_pitch_class = profiles[1:]
        self.chord_common_names = [chord.commonName for chord in self.chords]
        self.chord_pitched_names = [chord.pitchedCommonName for chord in self.chords]

    # --------------------------------------------------
    # --------------------------------------------------

    def melody_pitch_encoding(self):
        exact = []
        for melody in self.melodies:
            pitches = []
            for i in range(len(melody)):
                # note = str(melody[i].pitch)
                # note_separated = [note[:-1], note[-1]]
                # octave = int(note_separated[1]) - 3
                # note_value = NOTES[note_separated[0]] + (12 * octave)
                # pitches.append(note_value)
                pitches.append(melody[i].pitch.midi)
            exact.append(pitches)
        self.melodies_pitched = exact

    # --------------------------------------------------
    # --------------------------------------------------
    def melody_interval_encoding(self):
        relative = []
        for melody in self.melodies_pitched:
            temp_interval = []
            for i in range(len(melody) - 1):
                temp_interval.append(melody[i + 1] - melody[i])
            relative.append(temp_interval)
        self.melodies_intervals = relative

    # --------------------------------------------------
    # --------------------------------------------------
    def weighted_pitch_encoding(self):
        """
        Weighted melody dictionary generated on a per-chord
        basis.

        Weighting is based on the note duration.

        Possibility to extend this to be a more comprehensive
        weighting metric.
        """
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

        self.pitch_weights = pitch_weights

    # --------------------------------------------------
    # --------------------------------------------------

    def parsons_encoding(self):
        parson_melodies = []
        for m in self.melodies_intervals:
            pars = ["*"]
            for n in m:
                if n < 0:
                    pars.append("d")
                elif n == 0:
                    pars.append("r")
                else:
                    pars.append("u")
            parson_melodies.append(pars)

    # --------------------------------------------------
    # --------------------------------------------------

    def melody_reduction(self, ratio):

        """
        calculate interval salience, pitch stability, beat strength and durational accent
        for each note. These values are then combined in order to calculate a melodic
        weight for each note

        using a threshold value, reduce melody based on weight of each not
        threshold parameter can be used to adjust sensitivity of reduction
        """
        original_count = 0
        for melody_index, melody in enumerate(self.melodies):
            m = []
            for note_index, note in enumerate(melody):
                # get interval salience
                if note_index == 0:
                    interval_salience = 0
                else:
                    interval_salience = np.abs(self.melodies_intervals[melody_index][note_index - 1])

                # RENAME PITCH STABILITY
                pitch_stability = self.pitch_weights[melody_index][note_index]


                beat_strength = note.beatStrength
                durational_accent = note.duration.quarterLength
                weight = interval_salience + pitch_stability + beat_strength + durational_accent
                m.append(weight)

                original_count += 1

            self.weighted_melodies.append(m)

        new_count = original_count

        # here reduce by a compression ratio -> eg. 0.5
        for i in range(20):
            threshold = i * 0.25
            temporary_reduction = []
            for index, mel in enumerate(self.weighted_melodies):
                m2 = []
                for j, w in enumerate(mel):

                    if w < threshold:
                        m2.append(music21.note.Rest(quarterLength=self.melodies[index][j].duration.quarterLength))
                        original_count -= 1
                    else:
                        m2.append(self.melodies[index][j])

                temporary_reduction.append(m2)

            if -0.1 < (new_count / original_count) - ratio < 0.1:
                self.reduced_melodies = temporary_reduction
                return

    # --------------------------------------------------
    # --------------------------------------------------

    def normalize_reduction(self):
        norm = []
        for m in self.reduced_melodies:
            if len(m) > 1:
                temp = []
                for n in range(len(m) - 1):
                    temp.append(m[n+1].pitch.midi - m[n].pitch.midi)
                norm.append(temp)
            else:
                norm.append([0])
        self.normalized_reduction = norm
        return


    # --------------------------------------------------
    # --------------------------------------------------

    def reconstruct(self):
        """
        Reconstructs the melody and chord structure
        Works by recursing through the notes of the original score
        and replacing the notes with the reduced melody
        indexed by a counter
        """

        score = [n for n in self.score_parsed.recurse().notesAndRests]
        flat_melody = [n for m in self.reduced_melodies for n in m]
        for index, note in enumerate(score):
            if type(note) is music21.note.Note and type(note) is not music21.note.Rest:
                if flat_melody:
                    n = flat_melody.pop(0)
                else:
                    n = music21.note.Rest(quarterLength=note.duration.quarterLength)
                score[index] = n

        S = stream.Score()
        for i in score:
            S.append(i)
        self.reduced_stream = S

    # --------------------------------------------------
    # --------------------------------------------------

leadsheets = []
chord_dict = {}

chord_profile = []
chord_pitch_class = []
chord_common_names = []
melody_intervals = []
melody_pitched = []


def main():
    # main processing section
    for i in range(100):
        path = paths[i]
        path = "EWLD/" + path[0]
        sheet = Leadsheet(path)

        # THIS IS IN PLACE FOR A REASON:
        # ERRORS OCCUR WITH NO TIME SIG ENCODING
        # CANNOT CALCULATE BEAT STRENGTH
        if len(sheet.score_parsed.recurse().getElementsByClass(meter.TimeSignature)) == 0:
            continue

        sheet.chord_melody_separation()
        sheet.chord_feature_extraction()
        sheet.melody_pitch_encoding()
        sheet.melody_interval_encoding()

        sheet.weighted_pitch_encoding()
        sheet.melody_reduction(threshold=0.5)

        chord_profile.append(sheet.chord_profile)
        chord_pitch_class.append(sheet.chord_pitch_class)
        chord_common_names.append(sheet.chord_pitched_names)
        melody_intervals.append(sheet.melodies_intervals)
        melody_pitched.append(sheet.melodies_pitched)

    # chord_intervals = flatten(chord_intervals)
    # chord_profiles = flatten(chord_profiles)
    # melody_intervals = flatten(melody_intervals)
    # melody_exact = flatten(melody_exact)

    # print(build_duo_corp(chord_profile, melody_intervals))

main()

def flatten(arr):
    return [i for j in arr for i in j]


# KEEP THIS
# def melodic_reduction_test():
#     for s in range(5):
#         path = paths[s]
#         path = "EWLD/" + path[0]
#         sheet = Leadsheet(path)
#         if len(sheet.score_parsed.recurse().getElementsByClass(meter.TimeSignature)) == 0:
#             print("no time signature")
#             continue
#         # sheet.score_parsed.show()
#         sheet.chord_melody_separation()
#         sheet.chord_feature_extraction()
#         sheet.melody_pitch_encoding()
#         sheet.melody_interval_encoding()

#         sheet.weighted_pitch_encoding()

#         n = []
#         unique = []
#         for i in range(5):

#             sheet.melody_reduction(threshold=i)
#             sheet.reconstruct()
#             corp = build_corp_reduced(sheet.chord_profile, sheet.reduced_melodies)

#             # plot distinct melodies for each threshold
#             # sheet.reduced_stream.show()
#             # input()
#             n.append(i)
#             total = 0
#             for key in corp:
#                 total += len(corp[key].keys())
#             unique.append(total - len(corp.keys()))

#         print(n)
#         plt.plot(n, unique)
#         plt.title(("sheet: ", s))
#         plt.xlabel("Threshold")
#         plt.ylabel("Unique Melodies")
#         plt.show()


# melodic_reduction_test()


# def count_missing_time_sig():
#     count = 0
#     for path in paths:
#         path = "EWLD/" + path[0]
#         sheet = Leadsheet(path)
#         if len(sheet.score_parsed.recurse().getElementsByClass(meter.TimeSignature)) == 0:
#             count += 1
#             print(path)
#             print(count)
#             continue
#
#     print(count)
#
# count_missing_time_sig()

# # THIS SECTION FOR VISUALIZING A MUDDY DATA FIELD WITH NON-USEFUL CHORD SYMBOLS
# main()
# corp = build_single_corp(chord_common_names)
# plt.bar(list(corp.keys()), list(corp.values()))
# plt.xticks(list(corp.keys()), rotation=90)
# plt.title("Chord Pitched Common Names")
# plt.xlabel("chord name")
# plt.ylabel("count")
# plt.show()
# print(len(corp.keys()))
#
# corp = build_single_corp(chord_pitch_class)
# plt.bar(list(corp.keys()), list(corp.values()))
# plt.xticks(list(corp.keys()), rotation=90)
# plt.title("Chord Pitch Class")
# plt.xlabel("pitch class")
# plt.ylabel("count")
# plt.show()
# print(len(corp.keys()))
#
# corp = build_single_corp(chord_profile)
# plt.bar(list(corp.keys()), list(corp.values()))
# plt.xticks(list(corp.keys()), rotation=90)
# plt.title("Chord Profile")
# plt.xlabel("profile")
# plt.ylabel("count")
# plt.show()
# print(len(corp.keys()))

# # chord removal
# for n in c.recurse().getElementsByClass(chord.Chord):
#     meas = n.getContextByClass(stream.Measure)
#     print(meas.measureNumber, meas.offset, n.getOffsetInHierarchy(c), n.offset)
#     meas.remove(n)


# c.show()


# db = "/OpenEWLD-master/OpenEWLD.db"
# db = sqlite3.connect(db)
