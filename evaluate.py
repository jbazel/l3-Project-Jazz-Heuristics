import music21
import sqlite3
import random
import os
import json
from feature_extraction import omr_extract
from search import three_gram_search_test, analyse_prob_dist
from builder import build
from utils import flatten

def get_samples():
    db = sqlite3.connect('EWLD/EWLD.db')
    cursor = db.cursor()
    cursor.execute(
        'SELECT DISTINCT t.path_leadsheet FROM works t INNER JOIN work_genres w ON t.id = w.id WHERE w.genre = "Jazz"')
    paths = cursor.fetchall()
    db.close()

    samples = random.sample(paths, 10)
    return samples

def open_for_photo(samples):
    paths = []
    for i in range(len(samples)):
        path = "EWLD/" + samples[i][0]
        score = music21.converter.parse(path)
        path = score.write('musicxml.pdf')
        os.rename(str(path), 'omr_evaluation_images/pdf{}.pdf'.format(i))
# samples = get_samples()
# open_for_photo(samples)

def get_omr_outputs():
    # build(0.25)
    paths = os.listdir('omr_outputs/')
    paths = sorted(paths)
    for path in paths:
        if path[-4:] == 'mscz':
            continue
        print("Processing: ", path)
        score = music21.converter.parse("omr_outputs/" + path)
        key = score.analyze('key')
        key = key.tonicPitchNameWithCase
        chords, \
            melodies, \
            normal_order, \
            pc0, \
            numerals, \
            pitched, \
            intervals, \
            pitch_weights = omr_extract(score, 0.25)


        corpus = json.load(open("corpi/reduced_pitched_corpus.json"))
        note_probabilities = json.load(open("corpi/key_pitch_vec.json"))
        n = len(flatten(pitched))
        flags = analyse_prob_dist(three_gram_search_test(pitched, intervals, normal_order, numerals, corpus, note_probabilities, 0, key), n)
        print(flags)
get_omr_outputs()
