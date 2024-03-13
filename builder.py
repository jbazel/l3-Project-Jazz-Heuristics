import music21.converter
from music21 import *
import sqlite3
import json

from feature_extraction import extract
from chord_corpus_builder import build_duo_corp, build_single_corp, update_duo_corp, convert_corpus_to_probabilities, update_pitch_vec
# us = environment.UserSettings()
# us['musicxmlPath'] = '../../../../../Applications/MuseScore 3.app'
# us['midiPath'] = '../../../../../Applications/GarageBand.app'

from search import search
db = sqlite3.connect('EWLD/EWLD.db')
cursor = db.cursor()
cursor.execute('SELECT DISTINCT t.path_leadsheet FROM works t INNER JOIN work_genres w ON t.id = w.id WHERE w.genre = "Jazz"')
paths = cursor.fetchall()

pitched_corp = dict()
relative_corp = dict()
reduced_pitched_corp = dict()
reduced_relative_corp = dict()
pitch_vec = dict()


counter = 0

for path in paths:
    counter += 1
    print("Processing: ", counter, " of ", len(paths))
    print(path)
    print(counter/ len(paths) * 100, "%")
    path = "EWLD/" + path[0]
    score = music21.converter.parse(path)
    k = score.analyze('key')
    print(k)
    if len(score.recurse().getElementsByClass(meter.TimeSignature)) == 0:
        print("ERROR: Invalid Time Signature - Skipping")
        continue

    chords, \
    melodies, \
    normal_order, \
    pc0, \
    pitched, \
    intervals, \
    pitch_weights, \
    reduction, \
    interval_reduction = extract(score)

    pitch_vec = update_duo_corp(normal_order, melodies, pitch_vec)
    pitched_corp = update_duo_corp(normal_order, pitched, pitched_corp)
    relative_corp = update_duo_corp(pc0, intervals, relative_corp)
    reduced_pitched_corp = update_duo_corp(normal_order, reduction, reduced_pitched_corp)
    reduced_relative_corp = update_duo_corp(pc0, interval_reduction, reduced_relative_corp)


pitch_vec = convert_corpus_to_probabilities(pitch_vec)
print(pitch_vec)
pitched_corp = convert_corpus_to_probabilities(pitched_corp)
relative_corp = convert_corpus_to_probabilities(relative_corp)
reduced_pitched_corp = convert_corpus_to_probabilities(reduced_pitched_corp)
reduced_relative_corp = convert_corpus_to_probabilities(reduced_relative_corp)

with open("corpi/pitched_corpus.json", "w") as f:
    json.dump(pitched_corp, f)

with open("corpi/relative_corpus.json", "w") as f:
    json.dump(relative_corp, f)

with open("corpi/reduced_pitched_corpus.json", "w") as f:
    json.dump(reduced_pitched_corp, f)

with open("corpi/reduced_relative_corpus.json", "w") as f:
    json.dump(reduced_relative_corp, f)
