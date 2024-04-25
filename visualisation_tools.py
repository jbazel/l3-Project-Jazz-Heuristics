import matplotlib.pyplot as plt
import json
import music21
from music21 import *
from sklearn.metrics import RocCurveDisplay
import sqlite3
import numpy as np
import pandas as pd
from pandas.core.reshape.pivot import Index
import seaborn as sns
from builder import build
from nltk.corpus.reader.reviews import TITLE
from feature_extraction import extract, reconstruct, melodic_reduction_test
pitch_vec = json.load(open("corpi/pitch_vec.json"))
from test_data_creation import add_errors_for_vis
from search import three_gram_search

from utils import stringify

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

def reduction_flag_vis():
    with open("results/3gram_test1-seen.json") as f1, open("results/3gram_test2-seen-0.75.json") as f2, \
            open("results/3gram_test2-seen-0.5.json") as f3, open("results/3gram_test2-seen-0.25.json") as f4:
        test1_data = json.load(f1)
        test2_data_25 = json.load(f2)
        test2_data_50 = json.load(f3)
        test2_data_75 = json.load(f4)
        real1 = test1_data["test_flag_arr"][21]
        flags1 = np.array(test1_data["comparison_flag_arr"][21])
        flags1 = flags1.reshape(1, len(flags1))

        flags2 = np.array(test2_data_25["comparison_flag_arr"][21])
        flags2 = flags2.reshape(1, len(flags2))

        flags3 = np.array(test2_data_50["comparison_flag_arr"][21])
        flags3 = flags3.reshape(1, len(flags3))

        flags4 = np.array(test2_data_75["comparison_flag_arr"][20])
        flags4 = flags4.reshape(1, len(flags4))

        fig, (ax,ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, sharex=True, figsize=(10, 10))

        sns.heatmap(flags1, cmap='plasma', ax=ax, cbar=False)
        sns.heatmap(flags2, cmap='plasma', ax=ax2, cbar=False)
        sns.heatmap(flags3, cmap='plasma', ax=ax3, cbar=False)
        sns.heatmap(flags4, cmap='plasma', ax=ax4, cbar=False)
        color = fig.colorbar(ax.collections[0], ax=ax, orientation='horizontal', pad=0.2, location = 'top')
        ax5.bar(range(len(real1)), real1,width=1)
        plt.yticks([0])
        ax.set_title("Uncertainty Flags 0% reduction", fontsize=12)
        ax2.set_title("Uncertainty Flags 25% reduction", fontsize=12)
        ax3.set_title("Uncertainty Flags 50% reduction", fontsize=12)
        ax4.set_title("Uncertainty Flags 75% reduction", fontsize=12)
        ax5.set_title("True Errors", fontsize=12)
        ax.set_yticks([])
        ax2.set_yticks([])
        ax3.set_yticks([])
        ax4.set_yticks([])
        ax5.set_yticks([])
        ax5.set_xlabel("Note Displacement")
        fig.tight_layout(pad=0.5)
        plt.show()

    with open("results/3gram_test3-unseen.json") as f1, open("results/3gram_test4-unseen-0.75.json") as f2, \
            open("results/3gram_test4-unseen-0.5.json") as f3, open("results/3gram_test4-unseen-0.25.json") as f4:
        test1_data = json.load(f1)
        test2_data_25 = json.load(f2)
        test2_data_50 = json.load(f3)
        test2_data_75 = json.load(f4)
        real1 = test1_data["test_flag_arr"][61]
        flags1 = np.array(test1_data["comparison_flag_arr"][61])
        flags1 = flags1.reshape(1, len(flags1))

        flags2 = np.array(test2_data_25["comparison_flag_arr"][61])
        flags2 = flags2.reshape(1, len(flags2))

        flags3 = np.array(test2_data_50["comparison_flag_arr"][61])
        flags3 = flags3.reshape(1, len(flags3))

        flags4 = np.array(test2_data_75["comparison_flag_arr"][61])
        flags4 = flags4.reshape(1, len(flags4))

        fig, (ax,ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, sharex=True, figsize=(10, 10))

        sns.heatmap(flags1, cmap='plasma', ax=ax, cbar=False)
        sns.heatmap(flags2, cmap='plasma', ax=ax2, cbar=False)
        sns.heatmap(flags3, cmap='plasma', ax=ax3, cbar=False)
        sns.heatmap(flags4, cmap='plasma', ax=ax4, cbar=False)
        color = fig.colorbar(ax.collections[0], ax=ax, orientation='horizontal', pad=0.2, location = 'top')
        ax5.bar(range(len(real1)), real1,width=1)
        plt.yticks([0])
        ax.set_title("Uncertainty Flags 0% reduction", fontsize=12)
        ax2.set_title("Uncertainty Flags 25% reduction", fontsize=12)
        ax3.set_title("Uncertainty Flags 50% reduction", fontsize=12)
        ax4.set_title("Uncertainty Flags 75% reduction", fontsize=12)
        ax5.set_title("True Errors", fontsize=12)
        ax.set_yticks([])
        ax2.set_yticks([])
        ax3.set_yticks([])
        ax4.set_yticks([])
        ax5.set_yticks([])
        ax5.set_xlabel("Note Displacement")
        fig.tight_layout(pad=0.5)
        plt.show()



def get_most_acc_test():
    with open("results/stats-3gram_test1-seen.json") as f1, \
    open("results/stats-3gram_test1-unseen.json") as f2, \
    open("results/stats-3gram_test2-seen-0.25.json") as f3, \
    open("results/stats-3gram_test2-unseen-0.25.json") as f4, \
    open("results/stats-3gram_test2-seen-0.5.json") as f5, \
    open("results/stats-3gram_test2-unseen-0.5.json") as f6, \
    open("results/stats-3gram_test2-seen-0.75.json") as f7, \
    open("results/stats-3gram_test2-unseen-0.75.json") as f8:

        data1 = json.load(f1)
        data2 = json.load(f2)
        data3 = json.load(f3)
        data4 = json.load(f4)
        data5 = json.load(f5)
        data6 = json.load(f6)
        data7 = json.load(f7)
        data8 = json.load(f8)

        datas = [[data1, "test-1-seen"], \
        [data2, "test-1-unseen"],\
        [data3, "test-2-0.25-seen"], \
        [data4, "test-2-0.25-unseen"],\
        [data5, "test-2-0.5-seen"], \
        [data6, "test-2-0.5-unseen"],\
        [data7, "test-2-0.75-seen"], \
        [data8, "test-2-0.75-unseen"]]

        best_f1 = 0
        best_test_pres = 0
        best_test_recall = 0
        best_test_acc = 0
        best_test = ""
        best_index = None
        average_accuracies = []
        average_f1s = []
        for data in datas:
            count = 0
            avg_f1 = 0
            avg_acc = 0
            for index, test in enumerate(data[0].values()):
                pres = test["precision"]
                recall = test["recall"]
                if pres + recall == 0:
                    continue
                acc = test["accuracy"]
                f1 = 2 * (pres * recall) / (pres + recall)

                avg_f1 += f1
                avg_acc += acc
                count += 1


                if f1 > best_f1:
                    best_f1 = f1
                    best_test_pres = pres
                    best_test_recall = recall
                    best_test_acc = acc
                    best_test = data[1]
                    best_index = index

            average_accuracies.append(avg_acc/count)
            average_f1s.append(avg_f1/count)

        average_accuracies = [(average_accuracies[i] + average_accuracies[i+1])/2 for i in range(0, len(average_accuracies), 2)]
        average_f1s = [(average_f1s[i] + average_f1s[i+1])/2 for i in range(0, len(average_f1s), 2)]
        print("Best Test: ", best_test)
        print("Best F1: ", best_f1)
        print("Best Precision: ", best_test_pres)
        print("Best Recall: ", best_test_recall)
        print("Best Accuracy: ", best_test_acc)
        print("Best Index: ", best_index)
        print("Average Accuracies: ", average_accuracies)
        print("Average F1s: ", average_f1s)

    with open("results/stats-3gram_test3-seen.json") as f1, \
    open("results/stats-3gram_test3-unseen.json") as f2, \
    open("results/stats-3gram_test4-seen-0.25.json") as f3, \
    open("results/stats-3gram_test4-unseen-0.25.json") as f4, \
    open("results/stats-3gram_test4-seen-0.5.json") as f5, \
    open("results/stats-3gram_test4-unseen-0.5.json") as f6, \
    open("results/stats-3gram_test4-seen-0.75.json") as f7, \
    open("results/stats-3gram_test4-unseen-0.75.json") as f8:

        data1 = json.load(f1)
        data2 = json.load(f2)
        data3 = json.load(f3)
        data4 = json.load(f4)
        data5 = json.load(f5)
        data6 = json.load(f6)
        data7 = json.load(f7)
        data8 = json.load(f8)

        datas = [[data1, "test-3-seen"], \
        [data2, "test-3-unseen"],\
        [data3, "test-4-0.25-seen"], \
        [data4, "test-4-0.25-unseen"],\
        [data5, "test-4-0.5-seen"], \
        [data6, "test-4-0.5-unseen"],\
        [data7, "test-4-0.75-seen"], \
        [data8, "test-4-0.75-unseen"]]

        best_f1 = 0
        best_test_pres = 0
        best_test_recall = 0
        best_test_acc = 0
        best_test = ""
        best_index = None
        average_accuracies = []
        average_f1s = []


        for data in datas:
            count = 0
            avg_f1 = 0
            avg_acc = 0
            for index, test in enumerate(data[0].values()):
                pres = test["precision"]
                recall = test["recall"]
                if pres + recall == 0:
                    continue
                acc = test["accuracy"]
                f1 = 2 * (pres * recall) / (pres + recall)
                avg_f1 += f1
                avg_acc += acc
                count += 1
                if f1 > best_f1:
                    best_f1 = f1
                    best_test_pres = pres
                    best_test_recall = recall
                    best_test_acc = acc
                    best_test = data[1]
                    best_index = index

            average_accuracies.append(avg_acc/count)
            average_f1s.append(avg_f1/count)


        print("--------------------")
        average_accuracies = [(average_accuracies[i] + average_accuracies[i+1])/2 for i in range(0, len(average_accuracies), 2)]
        average_f1s = [(average_f1s[i] + average_f1s[i+1])/2 for i in range(0, len(average_f1s), 2)]
        print("Best Test: ", best_test)
        print("Best F1: ", best_f1)
        print("Best Precision: ", best_test_pres)
        print("Best Recall: ", best_test_recall)
        print("Best Accuracy: ", best_test_acc)
        print("Best Index: ", best_index)
        print("Average Accuracies: ", average_accuracies)
        print("Average F1s: ", average_f1s)












def flag_vis():

    # test 1-seen:
    with open("results/3gram_test1-unseen.json") as f:
        test1_data = json.load(f)
        real = test1_data["test_flag_arr"][30]
        flags = np.array(test1_data["comparison_flag_arr"][30])
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
        real = test1_data["test_flag_arr"][30]
        flags = np.array(test1_data["comparison_flag_arr"][30])
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



def visualize_accuracy_with_reduction():
    acc_dep = [0.5295856835393198, 0.643415787338521, 0.6246888086186316, 0.6008383963851374]
    f1_dep = [0.26393193198987125, 0.22624082867681677, 0.2403772647531024, 0.25627506854725296]

    acc_indep = [0.5313309185971187, 0.572707010671565, 0.5811873699708818, 0.5816035407609089]
    f1_indep = [0.38328484742376623, 0.23241898685051549, 0.2726489419538305, 0.30310579756469275]
    plt.plot([0, 25, 50, 75], acc_dep, label="KDC Accuracy")
    plt.plot([0, 25, 50, 75], acc_indep, label="KIC Accuracy")

    plt.legend()
    plt.xlabel("Percentage of Reduction")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Percentage of Reduction")
    plt.show()

    plt.plot([0, 25, 50, 75], f1_dep, label="KDC f1 Score")
    plt.plot([0, 25, 50, 75], f1_indep, label="KIC f1 Score")

    plt.legend()
    plt.xlabel("Percentage of Reduction")
    plt.ylabel("f1 Score")
    plt.title("f1 Score vs. Percentage of Reduction")
    plt.show()


def vis_best():
    with open("results/3gram_test2-seen-0.75.json") as f1, open("results/3gram_test3-unseen.json") as f2:
        data_kd = json.load(f1)
        data_ki = json.load(f2)

        uncertainties_kd = data_kd["comparison_flag_arr"][51]
        true_kd = data_kd["test_flag_arr"][51]
        rounded_flags = [round(x) for x in uncertainties_kd]
        uncertainties_kd = np.array(uncertainties_kd).reshape(1, len(uncertainties_kd))
        true = [1 if (true_kd[i] == 1 and rounded_flags[i] == 1) else 0 for i in range(len(true_kd))]

        fig, (ax, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(10, 7))
        sns.heatmap(uncertainties_kd, cmap='plasma', ax=ax, cbar=False)
        color = fig.colorbar(ax.collections[0], ax=ax, orientation='horizontal', pad=0.2, location = 'top')
        ax2.bar(range(len(true_kd)), true_kd, width=1)
        ax3.bar(range(len(true)), true, width=1)
        plt.yticks([0])
        ax.set_title("Uncertainty Flags", fontsize=12)
        ax2.set_title("True Errors", fontsize=12)
        ax3.set_title("True Positives", fontsize=12)
        ax.set_yticks([])
        ax2.set_yticks([])
        ax3.set_yticks([])
        ax3.set_xlabel("Note Displacement")
        plt.xticks(ticks = [i for i in range(0, len(true), 5)], labels =  [str(i) for i in range(0, len(true), 5)])
        fig.tight_layout(pad=0.5)
        plt.show()

        uncertainties_ki = data_ki["comparison_flag_arr"][61]
        true_ki = data_ki["test_flag_arr"][61]
        rounded_flags = [round(x) for x in uncertainties_ki]
        uncertainties_ki = np.array(uncertainties_ki).reshape(1, len(uncertainties_ki))
        true = [1 if (true_ki[i] == 1 and rounded_flags[i] == 1) else 0 for i in range(len(true_ki))]

        fig, (ax,ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(10, 7))
        sns.heatmap(uncertainties_ki, cmap='plasma', ax=ax, cbar=False)
        color = fig.colorbar(ax.collections[0], ax=ax, orientation='horizontal', pad=0.2, location = 'top')
        ax2.bar(range(len(true_ki)), true_ki, width=1)
        ax3.bar(range(len(true)), true, width=1)
        plt.yticks([0])
        ax.set_title("Uncertainty Flags", fontsize=12)
        ax2.set_title("True Errors", fontsize=12)
        ax3.set_title("True Positives", fontsize=12)
        ax.set_yticks([])
        ax2.set_yticks([])
        ax3.set_yticks([])
        ax3.set_xlabel("Note Displacement")
        plt.xticks(ticks = [i for i in range(0, len(true), 5)], labels =  [str(i) for i in range(0, len(true),5)])
        fig.tight_layout(pad=0.5)
        plt.show()


def visualize_reduction_against_corp():
    reductions = []
    counts_dep = []
    counts_indep = []
    for i in range(10):
        build(1 - i * 0.1)
        with open ("corpi/reduced_relative_corpus.json") as f:
            reduction = json.load(f)
            count1 = 0
            for key in reduction:
                for key2 in reduction[key]:
                    count1 += 1
        with open("corpi/reduced_pitched_corpus.json") as f:
            reduction = json.load(f)
            count2 = 0
            for key in reduction:
                for key2 in reduction[key]:
                    count2 += 1


        reductions.append(i*10)
        counts_indep.append(count1)
        counts_dep.append(count2)



    plt.plot(reductions, counts_dep, label="Key Dependent")
    plt.plot(reductions, counts_indep, label="Key Independent")
    plt.xlabel("Percentage of Reduction")
    plt.ylabel("Number of Unique Melodies")
    plt.title("Reduction Threshold vs. Number of Unique Melodies")

    plt.show()

def get_corp_sizes():
    with open("corpi/relative_corpus.json") as f:
        corp = json.load(f)
        num_chords = 0
        num_mels = 0
        for key in corp:
            print(key)
            num_chords += 1
            for key2 in corp[key]:
                num_mels += 1
        print("relative")
        print(num_chords)
        print(num_mels)

    with open("corpi/pitched_corpus.json") as f:
        corp = json.load(f)
        num_chords = 0
        num_mels = 0
        for key in corp:
            num_chords += 1
            for key2 in corp[key]:
                num_mels += 1

        print("pitched")
        print(num_chords)
        print(num_mels)

def vis_pitch_vec():
    with open("corpi/key_pitch_vec.json") as f:
        corp = json.load(f)
        pitches = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        probs = []
        c = corp['B-']
        for i in range(12):
            key = str(i)
            if key in c:
                print(key)
                val = c[key]
                probs.append(val)
            else:
                probs.append(0)

        plt.plot(probs)
        plt.xticks(range(12), pitches)
        plt.xlabel("Pitch")
        plt.ylabel("Probability")
        plt.title("Pitch Vector for Key of B Flat Major")
        plt.show()






def visualize_test_data():
    path = "EWLD/" + paths[10][0]
    score = music21.converter.parse(path)
    key = score.analyze('key')
    error_melodies, flags = add_errors_for_vis(score, 0.1)

    stream = music21.stream.Stream()
    notes = [n for n in score.recurse().notes if n.isNote]
    for note in notes:
        n = music21.note.Note()
        n.pitch.midi = note.pitch.midi
        stream.append(n)

    stream.show()


    print(flags)
    stream = music21.stream.Stream()
    for melody in error_melodies:
        for note in melody:
            n = music21.note.Note()
            n.pitch.midi = note
            stream.append(n)
    stream.show()








def get_percent_missing_meter():
    total_scores = 0
    total_missing = 0
    for path in paths:
        total_scores += 1
        path = "EWLD/" + path[0]
        score = music21.converter.parse(path)
        if len(score.recurse().getElementsByClass(meter.TimeSignature)) == 0:
            print("ERROR: Invalid Time Signature - Skipping")
            total_missing += 1

    print(total_missing, total_scores)
    print(total_missing / total_scores)


def get_chord_encoding_counts():
    numerals = []
    norm_order = []
    pc0 = []
    pitchedNames = []
    for path in paths:
        score = music21.converter.parse("EWLD/" + path[0])
        key = score.analyze('key')
        chords = [n for n in score.recurse().notes if n.isChord]
        for chord in chords:
            num = music21.roman.romanNumeralFromChord(chord, key)
            num = num.figure
            numerals.append(num)
            fp = chord.normalOrder[0]
            rotate = [(n - fp) % 12 for n in chord.normalOrder]

            norm_order.append(stringify(chord.normalOrder))
            pc0.append(stringify(rotate))

            pitchedNames.append(chord.pitchedCommonName)

    norm_order = set(norm_order)
    numerals = set(numerals)
    pc0 = set(pc0)
    pitchedNames = set(pitchedNames)
    print("norm order: ", len(norm_order))
    print("numerals: ", len(numerals))
    print("pc0: ", len(pc0))
    print("pitchedNames: ", len(pitchedNames))




# vis_pitch_vec()
# get_corp_sizes()
# visualize_reduction_against_corp()
# flag_vis()
# visualize_accuracy_with_reduction()
# reduction_flag_vis()
# get_most_acc_test()
# visualize_test_data()
# get_percent_missing_meter()
# get_chord_encoding_counts()

# get_most_acc_test()
# vis_best()
print(len(paths))
