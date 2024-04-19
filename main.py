import music21.converter
from music21 import *
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import sys
from feature_extraction import omr_extract
import json
from utils import flatten
from search import three_gram_search_test, analyse_prob_dist


def main():
    with open("corpi/reduced_pitched_corpus.json") as f:
        search_corpus = json.load(f)
        path = sys.argv[1]
        score = music21.converter.parse(path)
        key = score.analyze('key')
        key = key.tonicPitchNameWithCase
        chords, \
            melodies, \
            normal_order, \
            pc0, \
            numerals, \
            pitched, \
            intervals, \
            pitch_weights = omr_extract(score)

        note_probabilities = json.load(open("corpi/key_pitch_vec.json"))
        n = len(flatten(pitched))
        dist = three_gram_search_test(pitched, intervals, normal_order, numerals, search_corpus, note_probabilities, 0, key)
        flags = analyse_prob_dist(dist, n)
        print(flags)


if __name__ == "__main__":
    main()
