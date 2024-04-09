import json
import os

from nltk.classify.megam import numpy
from numpy.lib.nanfunctions import nanprod
from search import three_gram_search_test
from utils import *
import numpy as np
from search import analyse_prob_dist


def get_array_sim(arr1, arr2):
    return np.abs(np.sum(np.array(arr1) - np.array(arr2)))

paths = os.listdir("test_files")

def test1():
    comparison_flag_arr = []
    test_flag_arr = []
    accuracies = []
    similarities = []
    tp = []
    tn = []
    fp = []
    fn = []
    for path in paths:

        print("Testing: ", path)
        test_data = json.load(open("test_files/" + path))
        pitched = test_data["pitched"]
        intervals = test_data["intervals"]
        normal_order = test_data["normal_order"]
        pc0 = test_data["pc0"]
        flags = flatten(test_data["flags"])

        corpus = json.load(open("corpi/pitched_corpus.json"))
        note_probabilities = json.load(open("corpi/pitch_vec.json"))

        dist, n = three_gram_search_test(pitched, intervals, normal_order, pc0, corpus, note_probabilities, 0)
        # print(n)
        comparison_flags = analyse_prob_dist(dist, n)
        comparison_flags = (comparison_flags-np.min(comparison_flags))/(np.max(comparison_flags)-np.min(comparison_flags))

        rounded_flags = [np.round(flag) for flag in comparison_flags]
        # print(rounded_flags)

        false_pos = (sum([1 for i in range(len(flags)) if flags[i] == 0 and rounded_flags[i] == 1]) / len(flags)) * 100
        false_neg = (sum([1 for i in range(len(flags)) if flags[i] == 1 and rounded_flags[i] == 0]) / len(flags)) * 100
        true_pos = (sum([1 for i in range(len(flags)) if flags[i] == 1 and rounded_flags[i] == 1]) / len(flags)) * 100
        true_neg = (sum([1 for i in range(len(flags)) if flags[i] == 0 and rounded_flags[i] == 0]) / len(flags)) * 100

        # print("False Positives: ", false_pos, "%")
        # print("False Negatives: ", false_neg, "%")
        # print("True Positives: ", true_pos, "%")
        # print("True Negatives: ", true_neg, "%")
        try:
            accuracy = ((true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)) * 100

        except ZeroDivisionError:
            print("Zero Division Error")
            accuracy = 0
        # print("Accuracy: ", accuracy, "%")

        similarity = 100 - (get_array_sim(flags, comparison_flags) / len(flags)) * 100
        # print("Distance: ", similarity, "%")

        test_flag_arr.append(flags)
        comparison_flag_arr.append(comparison_flags.tolist())
        accuracies.append(accuracy)
        similarities.append(similarity)
        tp.append(true_pos)
        tn.append(true_neg)
        fp.append(false_pos)
        fn.append(false_neg)

    avg_accuracy = np.mean(accuracies)
    avg_similarity = np.mean(similarities)
    avg_tp = np.mean(tp)
    avg_tn = np.mean(tn)
    avg_fp = np.mean(fp)
    avg_fn = np.mean(fn)
    print("Average Accuracy: ", avg_accuracy, "%")
    return avg_accuracy, avg_similarity, avg_tp, avg_tn, avg_fp, avg_fn, test_flag_arr, comparison_flag_arr

def test2():
    comparison_flag_arr = []
    test_flag_arr = []
    accuracies = []
    similarities = []
    tp = []
    tn = []
    fp = []
    fn = []
    for path in paths:
        print("Testing: ", path)
        test_data = json.load(open("test_files/" + path))
        pitched = test_data["pitched"]
        intervals = test_data["intervals"]
        normal_order = test_data["normal_order"]
        pc0 = test_data["pc0"]
        flags = flatten(test_data["flags"])

        corpus = json.load(open("corpi/reduced_pitched_corpus.json"))
        note_probabilities = json.load(open("corpi/pitch_vec.json"))

        dist, n = three_gram_search_test(pitched, intervals, normal_order, pc0, corpus, note_probabilities, 0)
        # print(n)
        comparison_flags = analyse_prob_dist(dist, n)
        comparison_flags = (comparison_flags-np.min(comparison_flags))/(np.max(comparison_flags)-np.min(comparison_flags))
        rounded_flags = [np.round(flag) for flag in comparison_flags]
        # print(rounded_flags)

        false_pos = (sum([1 for i in range(len(flags)) if flags[i] == 0 and rounded_flags[i] == 1]) / len(flags)) * 100
        false_neg = (sum([1 for i in range(len(flags)) if flags[i] == 1 and rounded_flags[i] == 0]) / len(flags)) * 100
        true_pos = (sum([1 for i in range(len(flags)) if flags[i] == 1 and rounded_flags[i] == 1]) / len(flags)) * 100
        true_neg = (sum([1 for i in range(len(flags)) if flags[i] == 0 and rounded_flags[i] == 0]) / len(flags)) * 100

        # print("False Positives: ", false_pos, "%")
        # print("False Negatives: ", false_neg, "%")
        # print("True Positives: ", true_pos, "%")
        # print("True Negatives: ", true_neg, "%")
        try:
            accuracy = ((true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)) * 100
        except:
            accuracy = 0
        # print("Accuracy: ", accuracy, "%")

        similarity = 100 - (get_array_sim(flags, comparison_flags) / len(flags)) * 100
        # print("Distance: ", similarity, "%")

        test_flag_arr.append(flags)
        comparison_flag_arr.append(comparison_flags.tolist())
        accuracies.append(accuracy)
        similarities.append(similarity)
        tp.append(true_pos)
        tn.append(true_neg)
        fp.append(false_pos)
        fn.append(false_neg)

    avg_accuracy = np.mean(accuracies)
    avg_similarity = np.mean(similarities)
    avg_tp = np.mean(tp)
    avg_tn = np.mean(tn)
    avg_fp = np.mean(fp)
    avg_fn = np.mean(fn)

    return avg_accuracy, avg_similarity, avg_tp, avg_tn, avg_fp, avg_fn, test_flag_arr, comparison_flag_arr

    # print(get_array_sim(comparison_flags, flags))
    #

def test3():
    comparison_flag_arr = []
    test_flag_arr = []
    accuracies = []
    similarities = []
    tp = []
    tn = []
    fp = []
    fn = []
    for path in paths:

        print("Testing: ", path)
        test_data = json.load(open("test_files/" + path))
        pitched = test_data["pitched"]
        intervals = test_data["intervals"]
        normal_order = test_data["normal_order"]
        pc0 = test_data["pc0"]
        flags = flatten(test_data["flags"])

        corpus = json.load(open("corpi/relative_corpus.json"))
        note_probabilities = json.load(open("corpi/pitch_vec.json"))

        dist, n = three_gram_search_test(pitched, intervals, normal_order, pc0, corpus, note_probabilities, 1)
        # print(n)
        comparison_flags = analyse_prob_dist(dist, n)
        comparison_flags = (comparison_flags-np.min(comparison_flags))/(np.max(comparison_flags)-np.min(comparison_flags))
        rounded_flags = [np.round(flag) for flag in comparison_flags]
        # print(rounded_flags)

        false_pos = (sum([1 for i in range(len(flags)) if flags[i] == 0 and rounded_flags[i] == 1]) / len(flags)) * 100
        false_neg = (sum([1 for i in range(len(flags)) if flags[i] == 1 and rounded_flags[i] == 0]) / len(flags)) * 100
        true_pos = (sum([1 for i in range(len(flags)) if flags[i] == 1 and rounded_flags[i] == 1]) / len(flags)) * 100
        true_neg = (sum([1 for i in range(len(flags)) if flags[i] == 0 and rounded_flags[i] == 0]) / len(flags)) * 100

        # print("False Positives: ", false_pos, "%")
        # print("False Negatives: ", false_neg, "%")
        # print("True Positives: ", true_pos, "%")
        # print("True Negatives: ", true_neg, "%")
        try:
            accuracy = ((true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)) * 100

        except ZeroDivisionError:
            print("Zero Division Error")
            accuracy = 0
        # print("Accuracy: ", accuracy, "%")

        similarity = 100 - (get_array_sim(flags, comparison_flags) / len(flags)) * 100
        # print("Distance: ", similarity, "%")

        test_flag_arr.append(flags)
        comparison_flag_arr.append(comparison_flags.tolist())
        accuracies.append(accuracy)
        similarities.append(similarity)
        tp.append(true_pos)
        tn.append(true_neg)
        fp.append(false_pos)
        fn.append(false_neg)

    avg_accuracy = np.mean(accuracies)
    avg_similarity = np.mean(similarities)
    avg_tp = np.mean(tp)
    avg_tn = np.mean(tn)
    avg_fp = np.mean(fp)
    avg_fn = np.mean(fn)

    print("Average Accuracy: ", avg_accuracy, "%")
    return avg_accuracy, avg_similarity, avg_tp, avg_tn, avg_fp, avg_fn, test_flag_arr, comparison_flag_arr

def test4():
    comparison_flag_arr = []
    test_flag_arr = []
    accuracies = []
    similarities = []
    tp = []
    tn = []
    fp = []
    fn = []
    for path in paths:

        print("Testing: ", path)
        test_data = json.load(open("test_files/" + path))
        pitched = test_data["pitched"]
        intervals = test_data["intervals"]
        normal_order = test_data["normal_order"]
        pc0 = test_data["pc0"]
        flags = flatten(test_data["flags"])

        corpus = json.load(open("corpi/reduced_relative_corpus.json"))
        note_probabilities = json.load(open("corpi/pitch_vec.json"))

        dist, n = three_gram_search_test(pitched, intervals, normal_order, pc0, corpus, note_probabilities, 1)
        # print(n)
        comparison_flags = analyse_prob_dist(dist, n)
        comparison_flags = (comparison_flags-np.min(comparison_flags))/(np.max(comparison_flags)-np.min(comparison_flags))
        rounded_flags = [np.round(flag) for flag in comparison_flags]
        # print(rounded_flags)

        false_pos = (sum([1 for i in range(len(flags)) if flags[i] == 0 and rounded_flags[i] == 1]) / len(flags)) * 100
        false_neg = (sum([1 for i in range(len(flags)) if flags[i] == 1 and rounded_flags[i] == 0]) / len(flags)) * 100
        true_pos = (sum([1 for i in range(len(flags)) if flags[i] == 1 and rounded_flags[i] == 1]) / len(flags)) * 100
        true_neg = (sum([1 for i in range(len(flags)) if flags[i] == 0 and rounded_flags[i] == 0]) / len(flags)) * 100

        # print("False Positives: ", false_pos, "%")
        # print("False Negatives: ", false_neg, "%")
        # print("True Positives: ", true_pos, "%")
        # print("True Negatives: ", true_neg, "%")
        try:
            accuracy = ((true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)) * 100

        except ZeroDivisionError:
            print("Zero Division Error")
            accuracy = 0
        # print("Accuracy: ", accuracy, "%")

        similarity = 100 - (get_array_sim(flags, comparison_flags) / len(flags)) * 100
        # print("Distance: ", similarity, "%")

        test_flag_arr.append(flags)
        comparison_flag_arr.append(comparison_flags.tolist())
        accuracies.append(accuracy)
        similarities.append(similarity)
        tp.append(true_pos)
        tn.append(true_neg)
        fp.append(false_pos)
        fn.append(false_neg)

    avg_accuracy = np.mean(accuracies)
    avg_similarity = np.mean(similarities)
    avg_tp = np.mean(tp)
    avg_tn = np.mean(tn)
    avg_fp = np.mean(fp)
    avg_fn = np.mean(fn)

    print("Average Accuracy: ", avg_accuracy, "%")
    return avg_accuracy, avg_similarity, avg_tp, avg_tn, avg_fp, avg_fn, test_flag_arr, comparison_flag_arr


avg_accuracy, avg_similarity, avg_tp, avg_tn, avg_fp, avg_fn, test_flag_arr, comparison_flag_arr = test1()



with open("results/3gram_test1.json", "w") as f:
    json.dump({"avg_accuracy": avg_accuracy, "avg_similarity": avg_similarity, "avg_tp": avg_tp, "avg_tn": avg_tn, "avg_fp": avg_fp, "avg_fn": avg_fn, "test_flag_arr": test_flag_arr, "comparison_flag_arr": comparison_flag_arr}, f)

avg_accuracy, avg_similarity, avg_tp, avg_tn, avg_fp, avg_fn, test_flag_arr, comparison_flag_arr = test2()
with open("results/3gram_test2.json", "w") as f:
    json.dump({"avg_accuracy": avg_accuracy, "avg_similarity": avg_similarity, "avg_tp": avg_tp, "avg_tn": avg_tn, "avg_fp": avg_fp, "avg_fn": avg_fn, "test_flag_arr": test_flag_arr, "comparison_flag_arr": comparison_flag_arr}, f)

avg_accuracy, avg_similarity, avg_tp, avg_tn, avg_fp, avg_fn, test_flag_arr, comparison_flag_arr = test3()
with open("results/3gram_test3.json", "w") as f:
    json.dump({"avg_accuracy": avg_accuracy, "avg_similarity": avg_similarity, "avg_tp": avg_tp, "avg_tn": avg_tn, "avg_fp": avg_fp, "avg_fn": avg_fn, "test_flag_arr": test_flag_arr, "comparison_flag_arr": comparison_flag_arr}, f)

avg_accuracy, avg_similarity, avg_tp, avg_tn, avg_fp, avg_fn, test_flag_arr, comparison_flag_arr = test4()
with open("results/3gram_test4.json", "w") as f:
    json.dump({"avg_accuracy": avg_accuracy, "avg_similarity": avg_similarity, "avg_tp": avg_tp, "avg_tn": avg_tn, "avg_fp": avg_fp, "avg_fn": avg_fn, "test_flag_arr": test_flag_arr, "comparison_flag_arr": comparison_flag_arr}, f)
