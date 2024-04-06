import music21.converter
from music21 import *
import sqlite3
import json
from feature_extraction import extract
from chord_corpus_builder import build_duo_corp, build_single_corp, update_duo_corp, convert_corpus_to_probabilities, \
    update_pitch_vec, update_key_vec
# us = environment.UserSettings()
# us['musicxmlPath'] = '../../../../../Applications/MuseScore 3.app'
# us['midiPath'] = '../../../../../Applications/GarageBand.app'

from search import three_gram_search

db = sqlite3.connect('EWLD/EWLD.db')
cursor = db.cursor()
cursor.execute(
    'SELECT DISTINCT t.path_leadsheet FROM works t INNER JOIN work_genres w ON t.id = w.id WHERE w.genre = "Jazz"')
paths = cursor.fetchall()

pitched_corp = dict()
relative_corp = dict()
reduced_pitched_corp = dict()
reduced_relative_corp = dict()
pitch_vec = dict()
key_pitch_vec = dict()
counter = 0

paths = paths[:int(len(paths) - len(paths) / 10)]
for path in paths:
    counter += 1
    print("Processing: ", counter, " of ", len(paths))
    print(counter / len(paths) * 100, "%")
    path = "EWLD/" + path[0]
    score = music21.converter.parse(path)
    key = score.analyze('key')
    key = key.tonicPitchNameWithCase
    if len(score.recurse().getElementsByClass(meter.TimeSignature)) == 0:
        print("ERROR: Invalid Time Signature - Skipping")
        continue

    chords, \
        melodies, \
        normal_order, \
        pc0, \
        numerals, \
        pitched, \
        intervals, \
        pitch_weights, \
        reduction, \
        interval_reduction = extract(score)

    key_pitch_vec = update_key_vec(key, melodies, key_pitch_vec)
    pitch_vec = update_pitch_vec(normal_order, melodies, pitch_vec)
    pitched_corp = update_duo_corp(normal_order, pitched, pitched_corp)
    relative_corp = update_duo_corp(numerals, intervals, relative_corp)
    reduced_pitched_corp = update_duo_corp(normal_order, reduction, reduced_pitched_corp)
    reduced_relative_corp = update_duo_corp(numerals, interval_reduction, reduced_relative_corp)

key_pitch_vec = convert_corpus_to_probabilities(key_pitch_vec)
pitch_vec = convert_corpus_to_probabilities(pitch_vec)
pitched_corp = convert_corpus_to_probabilities(pitched_corp)
relative_corp = convert_corpus_to_probabilities(relative_corp)
reduced_pitched_corp = convert_corpus_to_probabilities(reduced_pitched_corp)
reduced_relative_corp = convert_corpus_to_probabilities(reduced_relative_corp)

print(key_pitch_vec)
with open("corpi/pitched_corpus.json", "w") as f:
    json.dump(pitched_corp, f)

with open("corpi/relative_corpus.json", "w") as f:
    json.dump(relative_corp, f)

with open("corpi/reduced_pitched_corpus.json", "w") as f:
    json.dump(reduced_pitched_corp, f)

with open("corpi/reduced_relative_corpus.json", "w") as f:
    json.dump(reduced_relative_corp, f)

with open("corpi/pitch_vec.json", "w") as f:
    json.dump(pitch_vec, f)

with open("corpi/key_pitch_vec.json", "w") as f:
    json.dump(key_pitch_vec, f)
