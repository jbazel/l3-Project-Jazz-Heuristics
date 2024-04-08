import matplotlib.pyplot as plt
import json
import music21
import sqlite3

from nltk.corpus.reader.reviews import TITLE
from feature_extraction import extract, reconstruct, melodic_reduction_test
pitch_vec = json.load(open("corpi/pitch_vec.json"))

from search import three_gram_search

db = sqlite3.connect('EWLD/EWLD.db')
cursor = db.cursor()
cursor.execute(
    'SELECT DISTINCT t.path_leadsheet FROM works t INNER JOIN work_genres w ON t.id = w.id WHERE w.genre = "Jazz"')
paths = cursor.fetchall()

path = "EWLD/" + paths[10][0]
score = music21.converter.parse(path)

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
# score.show()
reduction = melodic_reduction_test(melodies, pitched, intervals, pitch_weights, ratio=0.75)
stream = reconstruct(score, reduction)
stream.plot(title="Reduction to 75%")
stream.show()

reduction = melodic_reduction_test(melodies, pitched, intervals, pitch_weights, ratio=0.5)
stream = reconstruct(score, reduction)
stream.plot(title="Reduction to 50%")
stream.show()

reduction = melodic_reduction_test(melodies, pitched, intervals, pitch_weights, ratio=0.25)
stream = reconstruct(score, reduction)
stream.plot(title="Reduction to 25%")
stream.show()
