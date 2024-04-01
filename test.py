import json
import os
from search import three_gram_search_test

from search import analyse_prob_dist

paths = os.listdir("test_files")
for path in paths:
    test_data = json.load(open("test_files/" + path))
    pitched = test_data["pitched"]
    intervals = test_data["intervals"]
    normal_order = test_data["normal_order"]
    pc0 = test_data["pc0"]
    flags = test_data["flags"]

    corpus = json.load(open("corpi/pitched_corpus.json"))
    note_probabilities = json.load(open("corpi/pitch_vec.json"))

    dist, n = three_gram_search_test(pitched, intervals, normal_order, pc0, corpus, note_probabilities, 0)
    comparison_flags = analyse_prob_dist(dist, n)

    print("Comparison flags: ", comparison_flags, "Expected flags: ", flags)
