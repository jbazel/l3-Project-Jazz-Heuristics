import json
import os

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
        pc0 = test_data["pc0"]
        key = test_data["key"]
        flags = flatten(test_data["interval_flags"])

        corpus = json.load(open("corpi/relative_corpus.json"))
        note_probabilities = json.load(open("corpi/key_pitch_vec.json"))

        dist = three_gram_search_test(pitched, intervals, normal_order, pc0, corpus, note_probabilities, 1, key)
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
        pc0 = test_data["pc0"]
        key = test_data["key"]
        flags = flatten(test_data["interval_flags"])

        corpus = json.load(open("corpi/reduced_relative_corpus.json"))
        note_probabilities = json.load(open("corpi/key_pitch_vec.json"))

        dist = three_gram_search_test(pitched, intervals, normal_order, pc0, corpus, note_probabilities, 1, key)
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
    # test_flag_arr, comparison_flag_arr = test1(paths_seen)
    # with open("results/3gram_test1-seen.json", "w") as f:
    #     json.dump({"test_flag_arr": test_flag_arr, "comparison_flag_arr": comparison_flag_arr}, f)


    # test_flag_arr, comparison_flag_arr = test3(paths_seen)
    # with open("results/3gram_test3-seen.json", "w") as f:
    #     json.dump({"test_flag_arr": test_flag_arr, "comparison_flag_arr": comparison_flag_arr}, f)

    # for i in range(3):
    #     t = 0.25 + 0.25*i
    #     print(t)
    #     build(t)


    #     test_flag_arr, comparison_flag_arr = test2(paths_seen)
    #     with open("results/3gram_test2-seen-{}.json".format(t), "w") as f:
    #         json.dump({"test_flag_arr": test_flag_arr, "comparison_flag_arr": comparison_flag_arr}, f)

    #     test_flag_arr, comparison_flag_arr = test4(paths_seen)
    #     with open("results/3gram_test4-seen-{}.json".format(t), "w") as f:
    #         json.dump({"test_flag_arr": test_flag_arr, "comparison_flag_arr": comparison_flag_arr}, f)

    # ----------------------------
    # UNSEEN
    # ----------------------------

    # test_flag_arr, comparison_flag_arr = test1(paths_unseen)
    # with open("results/3gram_test1-unseen.json", "w") as f:
    #     json.dump({"test_flag_arr": test_flag_arr, "comparison_flag_arr": comparison_flag_arr}, f)


    # test_flag_arr, comparison_flag_arr = test3(paths_unseen)
    # with open("results/3gram_test3-unseen.json", "w") as f:
    #     json.dump({"test_flag_arr": test_flag_arr, "comparison_flag_arr": comparison_flag_arr}, f)

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
            # precision = tp / (tp + fp)
            # recall = tp / (tp + fn)

            similarity = sum([np.abs(test_flags[i] - comparison_flags[i]) for i in range(n)]) / n



# gen_statistics()
forward()
