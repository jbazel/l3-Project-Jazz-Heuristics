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

def reduction_flag_vis():
    with open("results/3gram_test1-seen.json") as f1, open("results/3gram_test2-seen-0.25.json") as f2, \
            open("results/3gram_test2-seen-0.5.json") as f3, open("results/3gram_test2-seen-0.75.json") as f4:
        test1_data = json.load(f1)
        test2_data_25 = json.load(f2)
        test2_data_50 = json.load(f3)
        test2_data_75 = json.load(f4)
        real1 = test1_data["test_flag_arr"][51]
        flags1 = np.array(test1_data["comparison_flag_arr"][51])
        flags1 = flags1.reshape(1, len(flags1))

        flags2 = np.array(test2_data_25["comparison_flag_arr"][51])
        flags2 = flags2.reshape(1, len(flags2))

        flags3 = np.array(test2_data_50["comparison_flag_arr"][51])
        flags3 = flags3.reshape(1, len(flags3))

        flags4 = np.array(test2_data_75["comparison_flag_arr"][51])
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

    with open("results/3gram_test3-seen.json") as f1, open("results/3gram_test4-seen-0.25.json") as f2, \
            open("results/3gram_test4-seen-0.5.json") as f3, open("results/3gram_test4-seen-0.75.json") as f4:
        test1_data = json.load(f1)
        test2_data_25 = json.load(f2)
        test2_data_50 = json.load(f3)
        test2_data_75 = json.load(f4)
        real1 = test1_data["test_flag_arr"][51]
        flags1 = np.array(test1_data["comparison_flag_arr"][51])
        flags1 = flags1.reshape(1, len(flags1))

        flags2 = np.array(test2_data_25["comparison_flag_arr"][51])
        flags2 = flags2.reshape(1, len(flags2))

        flags3 = np.array(test2_data_50["comparison_flag_arr"][51])
        flags3 = flags3.reshape(1, len(flags3))

        flags4 = np.array(test2_data_75["comparison_flag_arr"][51])
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
    with open("results/stats-3gram_test2-seen-0.75.json") as f:
        data = json.load(f)
        max_acc = 0
        for i in data.keys():
            acc = data[i]['precision']
            if acc > max_acc:
                max_acc = acc
                best = i

        print(best)


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
    key_dep_100 = open("results/stats-3gram_test1-seen.json")
    key_dep_75 = open("results/stats-3gram_test2-seen-0.75.json")
    key_dep_50 = open("results/stats-3gram_test2-seen-0.5.json")
    key_dep_25 = open("results/stats-3gram_test2-seen-0.25.json")

    key_dep_100_data = json.load(key_dep_100)
    key_dep_75_data = json.load(key_dep_75)
    key_dep_50_data = json.load(key_dep_50)
    key_dep_25_data = json.load(key_dep_25)

    key_indep_100 = open("results/stats-3gram_test3-seen.json")
    key_indep_75 = open("results/stats-3gram_test4-seen-0.75.json")
    key_indep_50 = open("results/stats-3gram_test4-seen-0.5.json")
    key_indep_25 = open("results/stats-3gram_test4-seen-0.25.json")

    key_indep_100_data = json.load(key_indep_100)
    key_indep_75_data = json.load(key_indep_75)
    key_indep_50_data = json.load(key_indep_50)
    key_indep_25_data = json.load(key_indep_25)

    acc_dep_100 = np.mean([i['accuracy'] for i in key_dep_100_data.values()])
    acc_dep_75 = np.mean([i['accuracy'] for i in key_dep_75_data.values()])
    acc_dep_50 = np.mean([i['accuracy'] for i in key_dep_50_data.values()])
    acc_dep_25 = np.mean([i['accuracy'] for i in key_dep_25_data.values()])

    acc_indep_100 = np.mean([i['accuracy'] for i in key_indep_100_data.values()])
    acc_indep_75 = np.mean([i['accuracy'] for i in key_indep_75_data.values()])
    acc_indep_50 = np.mean([i['accuracy'] for i in key_indep_50_data.values()])
    acc_indep_25 = np.mean([i['accuracy'] for i in key_indep_25_data.values()])

    acc_dep = [acc_dep_100, acc_dep_75, acc_dep_50, acc_dep_25]
    acc_indep = [acc_indep_100, acc_indep_75, acc_indep_50, acc_indep_25]

    print(acc_dep)
    print(acc_indep)
    plt.plot([0, 25, 50, 75], acc_dep, label="Key Dependent")
    plt.plot([0, 25, 50, 75], acc_indep, label="Key Independent")
    plt.legend()
    plt.xlabel("Percentage of Reduction")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Percentage of Reduction")


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






# vis_pitch_vec()
# get_corp_sizes()
# visualize_reduction_against_corp()
# flag_vis()
# visualize_accuracy_with_reduction()
# reduction_flag_vis()
# get_most_acc_test()
