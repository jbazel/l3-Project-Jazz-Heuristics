import matplotlib.pyplot as plt
import json
import music21
import sqlite3
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
    test1_data = json.load(open("results/3gram_test1.json"))
    test2_data = json.load(open("results/3gram_test2.json"))
    test3_data = json.load(open("results/3gram_test3.json"))
    test4_data = json.load(open("results/3gram_test4.json"))

    flags = test4_data["test_flag_arr"]
    real = test4_data["comparison_flag_arr"]
    print(len(flags[30]))
    print(len(real[30]))
    plt.plot(flags[30])
    plt.plot(real[30])

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

visualize_reduction_against_corp()
