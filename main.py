import music21.converter
from music21 import *
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import sys
import seaborn as sns
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
        min_flags = min(flags)
        max_flags = max(flags)
        normalised = [(x - min_flags) / (max_flags - min_flags) for x in flags]
        flags = np.array(normalised).reshape(1, -1)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title("Uncertainty Flags", fontsize=12)
        sns.heatmap(flags, cmap='plasma', ax=ax, cbar=False)
        color = fig.colorbar(ax.collections[0], ax=ax, orientation='horizontal', pad=0.2, location = 'top')
        plt.yticks([])
        plt.xticks([i for i in range(0, len(normalised), 10)], [str(i) for i in range(0, len(normalised), 10)])
        plt.xlabel("Note Displacement")
        plt.title("KDC")
        plt.show()

        dist = three_gram_search_test(pitched, intervals, normal_order, numerals, search_corpus, note_probabilities, 1, key)
        flags = analyse_prob_dist(dist, n)
        min_flags = min(flags)
        max_flags = max(flags)
        normalised = [(x - min_flags) / (max_flags - min_flags) for x in flags]
        flags = np.array(normalised).reshape(1, -1)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title("Uncertainty Flags", fontsize=12)
        sns.heatmap(flags, cmap='plasma', ax=ax, cbar=False)
        color = fig.colorbar(ax.collections[0], ax=ax, orientation='horizontal', pad=0.2, location = 'top')
        plt.yticks([])
        plt.xticks([i for i in range(0, len(normalised), 10)], [str(i) for i in range(0, len(normalised), 10)])
        plt.xlabel("Note Displacement")
        plt.title("KIC")
        plt.show()



if __name__ == "__main__":
    main()
