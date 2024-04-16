import matplotlib.pyplot as plt
import json
import music21
import sqlite3
import numpy as np
import pandas as pd
import seaborn as sns
from builder import build
from nltk.corpus.reader.reviews import TITLE
from feature_extraction import extract, reconstruct, melodic_reduction_test
pitch_vec = json.load(open("corpi/pitch_vec.json"))

from search import three_gram_search



db = sqlite3.connect('EWLD/EWLD.db')
cursor = db.cursor()
cursor.execute(
    'SELECT DISTINCT t.path_leadsheet FROM works t INNER JOIN work_genres w ON t.id = w.id WHERE w.genre = "Jazz"')
paths = cursor.fetchall()

def reduction_vis():
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

def flag_vis():
    # test1_data = json.load(open("results/3gram_test1-seen.json"))
    # test2_data = json.load(open("results/3gram_test2-seen-0.75.json"))
    # test3_data = json.load(open("results/3gram_test3-seen.json"))
    # test4_data = json.load(open("results/3gram_test4-seen-0.75.json"))

    # flags = test4_data["test_flag_arr"]
    # real = test4_data["comparison_flag_arr"]
    # print(len(flags[30]))
    # print(len(real[30]))
    # plt.plot(flags[30])
    # plt.plot(real[30])

    # plt.show()

    # test 1-seen:
    with open("results/3gram_test1-unseen.json") as f:
        test1_data = json.load(f)
        real = test1_data["test_flag_arr"][25]
        flags = np.array(test1_data["comparison_flag_arr"][25])
        rounded_flags = [round(x) for x in flags]
        true = [1 if (real[i] == 1 and rounded_flags[i] == 1) else 0 for i in range(len(real))]
        flags = flags.reshape(1, len(flags))
        fig, (ax,ax2, ax3, ax4) = plt.subplots(nrows=4, sharex=True, figsize=(10, 10))
        sns.heatmap(flags, cmap='plasma', ax=ax, cbar=False)
        color = fig.colorbar(ax.collections[0], ax=ax, orientation='horizontal', pad=0.2, location = 'top')
        ax2.bar(range(len(rounded_flags)), rounded_flags, width=1)
        ax3.bar(range(len(real)), real,width=1)
        ax4.bar(range(len(true)), true, width=1)
        plt.yticks([0])
        ax.set_title("Uncertainty Flags", fontsize=12)
        ax2.set_title("Rounded Uncertainties", fontsize=12)
        ax3.set_title("True Errors", fontsize=12)
        ax4.set_title("Correctly Flagged Errors", fontsize=12)
        ax.set_yticks([])
        ax2.set_yticks([])
        ax3.set_yticks([])
        ax4.set_yticks([])
        ax4.set_xlabel("Note Displacement")
        fig.tight_layout(pad=0.5)
        plt.show()

    with open("results/3gram_test3-unseen.json") as f:
        test1_data = json.load(f)
        real = test1_data["test_flag_arr"][25]
        flags = np.array(test1_data["comparison_flag_arr"][25])
        rounded_flags = [round(x) for x in flags]
        true = [1 if (real[i] == 1 and rounded_flags[i] == 1) else 0 for i in range(len(real))]
        false = [1 if (real[i] == 0 and rounded_flags[i] == 1) else 0 for i in range(len(real))]
        flags = flags.reshape(1, len(flags))


        fig, (ax,ax2, ax3, ax4) = plt.subplots(nrows=4, sharex=True, figsize=(10, 10))
        sns.heatmap(flags, cmap='plasma', ax=ax, cbar=False)
        color = fig.colorbar(ax.collections[0], ax=ax, orientation='horizontal', pad=0.2, location = 'top')
        ax2.bar(range(len(rounded_flags)), rounded_flags, width=1)
        ax3.bar(range(len(real)), real,width=1)
        ax4.bar(range(len(true)), true, width=1)
        plt.yticks([0])
        ax.set_title("Uncertainty Flags", fontsize=12)
        ax2.set_title("Rounded Uncertainties", fontsize=12)
        ax3.set_title("True Errors", fontsize=12)
        ax4.set_title("True Positives", fontsize=12)
        ax.set_yticks([])
        ax2.set_yticks([])
        ax3.set_yticks([])
        ax4.set_yticks([])
        ax4.set_xlabel("Note Displacement")
        fig.tight_layout(pad=0.5)
        plt.show()



def visualize_reduction_against_corp():
    reductions = []
    counts = []
    for i in range(10):
        build(1 - i * 0.1)
        with open ("corpi/reduced_relative_corpus.json") as f:
            reduction = json.load(f)
            count = 0
            for key in reduction:
                for key2 in reduction[key]:
                    count += 1

        reductions.append(str(100 - (i*10)) + "%")
        counts.append(count)

    plt.plot(reductions, counts)
    plt.xlabel("Reduction")
    plt.ylabel("Number of Unique Melodies")
    plt.title("Reduction Threshold vs. Number of Unique Melodies")

    plt.show()

# visualize_reduction_against_corp()
flag_vis()
