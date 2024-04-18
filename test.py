import json
import os

from nltk.tag.api import accuracy
import music21
from builder import build
from search import three_gram_search_test
from utils import *
import numpy as np
from search import analyse_prob_dist


def get_array_sim(arr1, arr2):
    return np.abs(np.sum(np.array(arr1) - np.array(arr2)))

paths = os.listdir("test_files")
n = len(paths)
print(paths)
paths_seen = [str(i) + ".json" for i in range(1, n//2)]
paths_unseen = [str(i) + ".json" for i in range(n//2, n)]


def get_num_notes(score):
    notes = [i for i in score.recurse().notes]
    return len(notes)

def test1(p):
    comparison_flag_arr = []
    test_flag_arr = []
    for path in p:
        if path[-5:] != ".json":
            continue
        print("Testing: ", path)
        test_data = json.load(open("test_files/" + path))
        pitched = test_data["pitched"]
        intervals = test_data["intervals"]
        normal_order = test_data["normal_order"]
        pc0 = test_data["pc0"]
        key = test_data["key"]
        flags = flatten(test_data["flags"])

        corpus = json.load(open("corpi/pitched_corpus.json"))
        note_probabilities = json.load(open("corpi/key_pitch_vec.json"))

        dist = three_gram_search_test(pitched, intervals, normal_order, pc0, corpus, note_probabilities, 0, key)
        # print(n)
        n = len(flatten(pitched))

        comparison_flags = analyse_prob_dist(dist, n)
        comparison_flags = (comparison_flags-np.min(comparison_flags))/(np.max(comparison_flags)-np.min(comparison_flags))
        if np.NAN in comparison_flags:
            comparison_flags = np.zeros(len(comparison_flags))
        test_flag_arr.append(flags)
        comparison_flag_arr.append(comparison_flags.tolist())

        if(len(flags) != len(comparison_flags)):
            print("Lengths don't match")
            print(len(flags))
            print(len(comparison_flags))

    return test_flag_arr, comparison_flag_arr

def test2(p):
    comparison_flag_arr = []
    test_flag_arr = []
    for path in p:
        if path[-5:] != ".json":
            continue
        print("Testing: ", path)
        test_data = json.load(open("test_files/" + path))
        pitched = test_data["pitched"]
        intervals = test_data["intervals"]
        normal_order = test_data["normal_order"]
        pc0 = test_data["pc0"]
        key = test_data["key"]
        flags = flatten(test_data["flags"])

        corpus = json.load(open("corpi/reduced_pitched_corpus.json"))
        note_probabilities = json.load(open("corpi/key_pitch_vec.json"))

        dist = three_gram_search_test(pitched, intervals, normal_order, pc0, corpus, note_probabilities, 0, key)
        n = len(flatten(pitched))
        comparison_flags = analyse_prob_dist(dist, n)
        comparison_flags = (comparison_flags-np.min(comparison_flags))/(np.max(comparison_flags)-np.min(comparison_flags))
        if np.NAN in comparison_flags:
            comparison_flags = np.zeros(len(comparison_flags))
        test_flag_arr.append(flags)
        comparison_flag_arr.append(comparison_flags.tolist())

        if(len(flags) != len(comparison_flags)):
            print("Lengths don't match")
            print(len(flags))
            print(len(comparison_flags))

    return test_flag_arr, comparison_flag_arr


def test3(p):
    comparison_flag_arr = []
    test_flag_arr = []

    for path in p:
        if path[-5:] != ".json":
            continue
        print("Testing: ", path)
        test_data = json.load(open("test_files/" + path))
        pitched = test_data["pitched"]
        intervals = test_data["intervals"]
        normal_order = test_data["normal_order"]
        numerals = test_data["numerals"]
        key = test_data["key"]
        flags = flatten(test_data["interval_flags"])

        corpus = json.load(open("corpi/relative_corpus.json"))
        note_probabilities = json.load(open("corpi/key_pitch_vec.json"))

        dist = three_gram_search_test(pitched, intervals, normal_order, numerals, corpus, note_probabilities, 1, key)
        n = len(flatten(intervals))
        comparison_flags = analyse_prob_dist(dist, n)
        comparison_flags = (comparison_flags-np.min(comparison_flags))/(np.max(comparison_flags)-np.min(comparison_flags))
        if np.NAN in comparison_flags:
            comparison_flags = np.zeros(len(comparison_flags))

        test_flag_arr.append(flags)
        comparison_flag_arr.append(comparison_flags.tolist())

        if(len(flags) != len(comparison_flags)):
            print("Lengths don't match")
            print(len(flags))
            print(len(comparison_flags))

    return test_flag_arr, comparison_flag_arr

def test4(p):
    comparison_flag_arr = []
    test_flag_arr = []
    for path in p:
        if path[-5:] != ".json":
            continue
        print("Testing: ", path)
        test_data = json.load(open("test_files/" + path))
        pitched = test_data["pitched"]
        intervals = test_data["intervals"]
        normal_order = test_data["normal_order"]
        numerals = test_data["numerals"]
        key = test_data["key"]
        flags = flatten(test_data["interval_flags"])

        corpus = json.load(open("corpi/reduced_relative_corpus.json"))
        note_probabilities = json.load(open("corpi/key_pitch_vec.json"))

        dist = three_gram_search_test(pitched, intervals, normal_order, numerals, corpus, note_probabilities, 1, key)
        n = len(flatten(intervals))
        comparison_flags = analyse_prob_dist(dist, n)
        comparison_flags = (comparison_flags-np.min(comparison_flags))/(np.max(comparison_flags)-np.min(comparison_flags))
        if np.NAN in comparison_flags:
            comparison_flags = np.zeros(len(comparison_flags))

        test_flag_arr.append(flags)
        comparison_flag_arr.append(comparison_flags.tolist())

        if(len(flags) != len(comparison_flags)):
            print("Lengths don't match")
            print(len(flags))
            print(len(comparison_flags))


    return test_flag_arr, comparison_flag_arr

# ----------------------------
# SEEN
# ----------------------------

def forward():
    test_flag_arr, comparison_flag_arr = test1(paths_seen)
    with open("results/3gram_test1-seen.json", "w") as f:
        json.dump({"test_flag_arr": test_flag_arr, "comparison_flag_arr": comparison_flag_arr}, f)


    test_flag_arr, comparison_flag_arr = test3(paths_seen)
    with open("results/3gram_test3-seen.json", "w") as f:
        json.dump({"test_flag_arr": test_flag_arr, "comparison_flag_arr": comparison_flag_arr}, f)

    for i in range(3):
        t = 0.25 + 0.25*i
        print(t)
        build(t)


        test_flag_arr, comparison_flag_arr = test2(paths_seen)
        with open("results/3gram_test2-seen-{}.json".format(t), "w") as f:
            json.dump({"test_flag_arr": test_flag_arr, "comparison_flag_arr": comparison_flag_arr}, f)

        test_flag_arr, comparison_flag_arr = test4(paths_seen)
        with open("results/3gram_test4-seen-{}.json".format(t), "w") as f:
            json.dump({"test_flag_arr": test_flag_arr, "comparison_flag_arr": comparison_flag_arr}, f)

    # ----------------------------
    # UNSEEN
    # ----------------------------

    test_flag_arr, comparison_flag_arr = test1(paths_unseen)
    with open("results/3gram_test1-unseen.json", "w") as f:
        json.dump({"test_flag_arr": test_flag_arr, "comparison_flag_arr": comparison_flag_arr}, f)


    test_flag_arr, comparison_flag_arr = test3(paths_unseen)
    with open("results/3gram_test3-unseen.json", "w") as f:
        json.dump({"test_flag_arr": test_flag_arr, "comparison_flag_arr": comparison_flag_arr}, f)

    for i in range(3):
        t = 0.25 + 0.25*i
        print(t)
        build(t)


        test_flag_arr, comparison_flag_arr = test2(paths_unseen)
        with open("results/3gram_test2-unseen-{}.json".format(t), "w") as f:
            json.dump({"test_flag_arr": test_flag_arr, "comparison_flag_arr": comparison_flag_arr}, f)

        test_flag_arr, comparison_flag_arr = test4(paths_unseen)
        with open("results/3gram_test4-unseen-{}.json".format(t), "w") as f:
            json.dump({"test_flag_arr": test_flag_arr, "comparison_flag_arr": comparison_flag_arr}, f)

def gen_statistics():
    paths = os.listdir("results")
    paths = sorted(paths)
    for path in paths:
        if path[-5:] != ".json":
            continue
        print("Processing: ", path)
        data = json.load(open("results/" + path))
        test_flag_arr = data["test_flag_arr"]
        comparison_flag_arr = data["comparison_flag_arr"]

        stats = {}

        for i in range(len(test_flag_arr)):
            test_flags = test_flag_arr[i]
            comparison_flags = np.nan_to_num(comparison_flag_arr[i])

            comparison_flags = [round(x) for x in comparison_flags]
            stats[i] = {}
            n = len(test_flags)
            tp = sum([1 if test_flags[i] == 1 and comparison_flags[i]== 1 else 0 for i in range(n)]) / n
            tn = sum([1 if test_flags[i] == 0 and comparison_flags[i] == 0 else 0 for i in range(n)]) / n
            fp = sum([1 if test_flags[i] == 0 and comparison_flags[i] == 1 else 0 for i in range(n)]) / n
            fn = sum([1 if test_flags[i] == 1 and comparison_flags[i] == 0 else 0 for i in range(n)]) / n
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            try:
                precision = tp / (tp + fp)

            except:
                precision = 0

            try:
                recall = tp / (tp + fn)

            except:
                recall = 0

            similarity = sum([np.abs(test_flags[i] - comparison_flags[i]) for i in range(n)]) / n

            stats[i]["tp"] = tp
            stats[i]["tn"] = tn
            stats[i]["fp"] = fp
            stats[i]["fn"] = fn
            stats[i]["accuracy"] = accuracy
            stats[i]["precision"] = precision
            stats[i]["recall"] = recall
            stats[i]["similarity"] = similarity

        with open("results/stats-" + path, "w") as f:
            json.dump(stats, f)

def collate_stats():
    def pitched_unpitched():
        fpitched_seen = open("results/stats-3gram_test1-seen.json")
        fpitched_unseen = open("results/stats-3gram_test1-unseen.json")

        funpitched_seen = open("results/stats-3gram_test3-seen.json")
        funpitched_unseen = open("results/stats-3gram_test3-unseen.json")

        pitched_seen = json.load(fpitched_seen)
        pitched_unseen = json.load(fpitched_unseen)

        unpitched_seen = json.load(funpitched_seen)
        unpitched_unseen = json.load(funpitched_unseen)

        acc_arr = []
        precision_arr = []
        recall_arr = []
        sim_arr = []
        for i in pitched_seen.values():
            acc = i["accuracy"]
            precision = i["precision"]
            recall = i["recall"]
            sim = i["similarity"]

            acc_arr.append(acc)
            precision_arr.append(precision)
            recall_arr.append(recall)
            sim_arr.append(sim)

        average_pitched_seen_acc = np.mean(acc_arr)
        average_pitched_seen_precision = np.mean(precision_arr)
        average_pitched_seen_recall = np.mean(recall_arr)
        average_pitched_seen_sim = np.mean(sim_arr)

        acc_arr = []
        precision_arr = []
        recall_arr = []
        sim_arr = []

        for i in pitched_unseen.values():
            acc = i["accuracy"]
            precision = i["precision"]
            recall = i["recall"]
            sim = i["similarity"]

            acc_arr.append(acc)
            precision_arr.append(precision)
            recall_arr.append(recall)
            sim_arr.append(sim)

        average_pitched_unseen_acc = np.mean(acc_arr)
        average_pitched_unseen_precision = np.mean(precision_arr)
        average_pitched_unseen_recall = np.mean(recall_arr)
        average_pitched_unseen_sim = np.mean(sim_arr)

        acc_arr = []
        precision_arr = []
        recall_arr = []
        sim_arr = []

        for i in unpitched_seen.values():
            acc = i["accuracy"]
            precision = i["precision"]
            recall = i["recall"]
            sim = i["similarity"]

            acc_arr.append(acc)
            precision_arr.append(precision)
            recall_arr.append(recall)
            sim_arr.append(sim)

        average_unpitched_seen_acc = np.mean(acc_arr)
        average_unpitched_seen_precision = np.mean(precision_arr)
        average_unpitched_seen_recall = np.mean(recall_arr)
        average_unpitched_seen_sim = np.mean(sim_arr)

        acc_arr = []
        precision_arr = []
        recall_arr = []
        sim_arr = []

        for i in unpitched_unseen.values():

            acc = i["accuracy"]
            precision = i["precision"]
            recall = i["recall"]
            sim = i["similarity"]

            acc_arr.append(acc)
            precision_arr.append(precision)
            recall_arr.append(recall)
            sim_arr.append(sim)

        average_unpitched_unseen_acc = np.mean(acc_arr)
        average_unpitched_unseen_precision = np.mean(precision_arr)
        average_unpitched_unseen_recall = np.mean(recall_arr)
        average_unpitched_unseen_sim = np.mean(sim_arr)

        average_pitched = (average_pitched_seen_acc + average_pitched_unseen_acc) / 2
        average_unpitched = (average_unpitched_seen_acc + average_unpitched_unseen_acc) / 2

        print("Pitched Seen: ", average_pitched_seen_acc, average_pitched_seen_precision, average_pitched_seen_recall, average_pitched_seen_sim)
        print("Pitched Unseen: ", average_pitched_unseen_acc, average_pitched_unseen_precision, average_pitched_unseen_recall, average_pitched_unseen_sim)
        print("Unpitched Seen: ", average_unpitched_seen_acc, average_unpitched_seen_precision, average_unpitched_seen_recall, average_unpitched_seen_sim)
        print("Unpitched Unseen: ", average_unpitched_unseen_acc, average_unpitched_unseen_precision, average_unpitched_unseen_recall, average_unpitched_unseen_sim)

        print("Pitched: ", average_pitched)
        print("Unpitched: ", average_unpitched)


    pitched_unpitched()

def music21chordtest():
    c1 = music21.chord.Chord(['c4', 'e4', 'g4', 'a5'])
    c2 = music21.chord.Chord(['a4', 'c4', 'e4', 'g4'])

    # print(c1.pitchedCommonName)
    # print(c2.pitchedCommonName)
    # c1.root = 'C'
    # norm1 = c1.normalOrder
    # norm1.insert(0, c1.root().name)
    # fp1 = norm1[0]
    # print(fp1)
    # # rotate1 = [(n - fp1) % 12 for n in norm1]
    # # print(rotate1)

    # norm2 = c2.normalOrder
    # norm2.insert(0, c2.root().name)
    # fp2 = norm2[0]
    # print(fp2)
    # # rotate2 = [(n - fp2) % 12 for n in norm2]
    # # print(rotate2)

    # print(norm1, norm2)

# gen_statistics()
# forward()
# collate_stats()
# music21chordtest()
