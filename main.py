import music21.converter
from music21 import *
import sqlite3

us = environment.UserSettings()
us['musicxmlPath'] = '../../../../../Applications/MuseScore 3.app'
us['midiPath'] = '../../../../../Applications/GarageBand.app'
db = sqlite3.connect('EWLD/EWLD.db')
cursor = db.cursor()
cursor.execute('SELECT DISTINCT path_leadsheet FROM works, work_genres WHERE work_genres.genre = "Jazz"')
paths = cursor.fetchall()

class Leadsheet:

    def __init__(self, source):
        self.source = source
        self.score_parsed = music21.converter.parse(source)
        self.chords = []
        self.chord_profiles = []
        self.chord_intervals = []
        self.chord_common_names = []
        self.melodies = []

    def lyric_removal(self):
        for n in self.score_parsed.recurse().notesAndRests:
            if n.lyric:
                n.lyric = ""

    def chord_feature_extraction(self):
        profiles = []
        intervals = []
        for i in self.chords:
            profile = [0]*12
            interval = [0]*12
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

    def chord_melody_separation(self):
        score = [n for n in self.score_parsed.recurse().notes]
        separation = [x for x in range(len(score)) if type(score[x]) == music21.harmony.ChordSymbol]
        self.chords = [score[i] for i in separation]
        separation.append(len(score))
        for i in range(len(separation) - 1):
            self.melodies.append(score[separation[i] + 1:separation[i + 1]])



leadsheets = []
chord_dict = {}

for i in range(100):
    path = paths[i]
    # print('parsing ', paths)
    path = "EWLD/" + path[0]
    sheet = Leadsheet(path)
    sheet.chord_melody_separation()
    sheet.chord_feature_extraction()
    leadsheets.append(sheet)
    for j in sheet.chord_common_names:
        if j in chord_dict:
            chord_dict[j] += 1
        else:
            chord_dict[j] = 1
    print(sheet.chord_profiles)

#
# print(leadsheets)
# print(chord_dict)


# # chord removal
# for n in c.recurse().getElementsByClass(chord.Chord):
#     meas = n.getContextByClass(stream.Measure)
#     print(meas.measureNumber, meas.offset, n.getOffsetInHierarchy(c), n.offset)
#     meas.remove(n)


# c.show()


# db = "/OpenEWLD-master/OpenEWLD.db"
# db = sqlite3.connect(db)
