import numpy as np
import music21
from feature_extraction import extract
import json
from nltk import edit_distance
import statistics


def stringify(val):
    return ','.join([str(x) for x in val])


def three_gram_search(score, corpus, note_probabilities):
    reduced_pitched_corp = json.load(open("corpi/reduced_pitched_corpus.json"))
    reduced_relative_corp = json.load(open("corpi/reduced_relative_corpus.json"))
    corpus = reduced_relative_corp
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

    probs = []
    for index, m in enumerate(intervals):
        length = len(m)
        if length > 3:
            for i in range(length - 2):
                search_gram = m[i:i + 3]
                current_chord = pc0[index]
                current_chord = stringify(current_chord)
                if current_chord in corpus:
                    mindx = 100000
                    for comparison_gram in corpus[current_chord].keys():
                        dx = edit_distance(search_gram, comparison_gram)
                        # if dx != 0:
                        #     prob = corpus[current_chord][comparison_gram] * (1 - (0.1 / dx))
                        # else:
                        #     prob = corpus[current_chord][comparison_gram]
                        if dx < mindx:
                            mindx = dx
                            prob = corpus[current_chord][comparison_gram]

                    print("current search gram {} has probability: {}".format(search_gram, prob))

                else:
                    average_note_prob = 0
                    for note in search_gram:
                        average_note_prob += note_probabilities[current_chord][note]
                    average_note_prob /= len(search_gram)
                    prob = average_note_prob
                    print("current search gram {} has probability: {}".format(search_gram, prob))
            probs.append([search_gram, prob])

        else:
            if not m:
                continue
            search_gram = m
            current_chord = pc0[index]
            current_chord = stringify(current_chord)

            if current_chord in corpus:
                mindx = 100000
                for comparison_gram in corpus[current_chord].keys():
                    dx = edit_distance(search_gram, comparison_gram)
                    # if dx != 0:
                    #     prob = corpus[current_chord][comparison_gram] * (1 / dx)
                    # else:
                    #     prob = corpus[current_chord][comparison_gram]

                    if dx < mindx:
                        mindx = dx
                        prob = corpus[current_chord][comparison_gram]

                print("current search gram {} has probability: {}".format(search_gram, prob))

            else:
                average_note_prob = 0
                for note in search_gram:
                    average_note_prob += note_probabilities[current_chord][note]
                average_note_prob /= len(search_gram)
                prob = average_note_prob
                print("current search gram {} has probability: {}".format(search_gram, prob))

            probs.append([search_gram, prob])

    return probs


def analyse_prob_dist(dist):
    probs = [x[1] for x in dist]
    mean = statistics.mean(probs)
    variance = statistics.variance(probs)
    z_scores = [(x - mean) / variance for x in probs]
    for i in range(len(dist)):
        print("Z-score for {} is: {}".format(dist[i][0], z_scores[i]))


probs = three_gram_search(
    music21.converter.parse("EWLD/dataset/Adam_Anders-Nikki_Hassman-Peer_Åström/If_Only/If_Only.mxl"), None, None)
print(probs)

analyse_prob_dist(probs)


def search(score, method_flag=0):
    reduced_pitched_corp = json.load(open("corpi/reduced_pitched_corpus.json"))
    reduced_relative_corp = json.load(open("corpi/reduced_relative_corpus.json"))
    corpus = reduced_relative_corp
    chords, \
        melodies, \
        normal_order, \
        pc0, \
        pitched, \
        intervals, \
        pitch_weights, \
        reduction, \
        interval_reduction = extract(score)

    bar_flags = [False for i in range(len(reduction))]

    for i, m in enumerate(interval_reduction):
        prob = 0
        min_edit = 100000
        chord = pc0[i]
        if chord in corpus:
            current = corpus[chord]
            for comparison_melody in current.keys:
                # distance is levenstein
                edit = edit_distance(reduction[i], comparison_melody)
                if edit < min_edit:
                    min_edit = edit
                    prob = current[comparison_melody]

        print(prob)


def search_witout_chord(search_melody, corpus):
    pass
